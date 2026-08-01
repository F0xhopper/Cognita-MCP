"""The reranker must improve ordering when it works and vanish when it does not.

Returning None means "keep your own order" — search then falls back to the
fusion ranking rather than failing.
"""

from types import SimpleNamespace

import pytest

from cognita.core.config import settings
from cognita.infrastructure import reranker


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "RERANK_BATCH_SIZE", 20)


def _docs(n: int) -> list[tuple[str, str]]:
    return [(f"passage {i}", "") for i in range(n)]


def _client_returning(*batches: dict[int, float]):
    """A stub Anthropic client that answers each call from `batches` in turn."""
    calls = {"n": 0}

    class Messages:
        async def create(self, **kwargs):
            rankings = [
                {"index": i, "relevance": s} for i, s in batches[calls["n"]].items()
            ]
            calls["n"] += 1
            block = SimpleNamespace(type="tool_use", input={"rankings": rankings})
            return SimpleNamespace(content=[block])

    return SimpleNamespace(messages=Messages()), calls


# ── Disabled paths ────────────────────────────────────────────────────────────

async def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    assert await reranker.rerank("q", _docs(5), top_n=3) is None


async def test_missing_key_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    assert await reranker.rerank("q", _docs(5), top_n=3) is None


async def test_single_document_never_calls_the_model(monkeypatch):
    def boom():
        raise AssertionError("the model must not be called for one document")

    monkeypatch.setattr(reranker, "get_anthropic_client", boom)
    assert await reranker.rerank("q", _docs(1), top_n=5) is None


# ── Successful reranking ──────────────────────────────────────────────────────

async def test_reorders_by_model_score(monkeypatch):
    client, _ = _client_returning({0: 0.2, 1: 0.95, 2: 0.5})
    monkeypatch.setattr(reranker, "get_anthropic_client", lambda: client)

    result = await reranker.rerank("q", _docs(3), top_n=3)

    assert [index for index, _ in result] == [1, 2, 0]
    assert result[0][1] == 0.95


async def test_result_is_truncated_to_top_n(monkeypatch):
    client, _ = _client_returning({0: 0.1, 1: 0.9, 2: 0.5, 3: 0.4})
    monkeypatch.setattr(reranker, "get_anthropic_client", lambda: client)

    result = await reranker.rerank("q", _docs(4), top_n=2)

    assert [index for index, _ in result] == [1, 2]


async def test_scores_are_clamped_to_the_unit_range(monkeypatch):
    client, _ = _client_returning({0: 4.2, 1: -3.0})
    monkeypatch.setattr(reranker, "get_anthropic_client", lambda: client)

    result = await reranker.rerank("q", _docs(2), top_n=2)

    assert all(0.0 <= score <= 1.0 for _, score in result)


async def test_a_skipped_passage_is_kept_at_the_back(monkeypatch):
    client, _ = _client_returning({0: 0.8})  # index 1 never scored
    monkeypatch.setattr(reranker, "get_anthropic_client", lambda: client)

    result = await reranker.rerank("q", _docs(2), top_n=5)

    assert [index for index, _ in result] == [0, 1]
    assert result[1][1] == 0.0


async def test_out_of_range_indices_are_discarded(monkeypatch):
    client, _ = _client_returning({0: 0.5, 99: 1.0})
    monkeypatch.setattr(reranker, "get_anthropic_client", lambda: client)

    result = await reranker.rerank("q", _docs(2), top_n=5)

    assert {index for index, _ in result} == {0, 1}


# ── Batching ──────────────────────────────────────────────────────────────────

async def test_candidates_are_split_into_batches(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_BATCH_SIZE", 2)
    client, calls = _client_returning({0: 0.4, 1: 0.9}, {2: 0.7, 3: 0.1}, {4: 0.6})
    monkeypatch.setattr(reranker, "get_anthropic_client", lambda: client)

    result = await reranker.rerank("q", _docs(5), top_n=5)

    assert calls["n"] == 3, "five documents in batches of two is three calls"
    assert [index for index, _ in result] == [1, 2, 4, 0, 3]


async def test_every_document_survives_batching(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_BATCH_SIZE", 2)
    client, _ = _client_returning({0: 0.4, 1: 0.9}, {2: 0.7, 3: 0.1})
    monkeypatch.setattr(reranker, "get_anthropic_client", lambda: client)

    result = await reranker.rerank("q", _docs(4), top_n=10)

    assert sorted(index for index, _ in result) == [0, 1, 2, 3]


# ── Failure ───────────────────────────────────────────────────────────────────

async def test_total_failure_returns_none(monkeypatch):
    class Messages:
        async def create(self, **kwargs):
            raise RuntimeError("API down")

    monkeypatch.setattr(
        reranker, "get_anthropic_client", lambda: SimpleNamespace(messages=Messages())
    )

    assert await reranker.rerank("q", _docs(3), top_n=3) is None


async def test_one_failed_batch_does_not_lose_the_others(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_BATCH_SIZE", 2)
    calls = {"n": 0}

    class Messages:
        async def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first batch failed")
            block = SimpleNamespace(
                type="tool_use", input={"rankings": [{"index": 2, "relevance": 0.9}]}
            )
            return SimpleNamespace(content=[block])

    monkeypatch.setattr(
        reranker, "get_anthropic_client", lambda: SimpleNamespace(messages=Messages())
    )

    result = await reranker.rerank("q", _docs(4), top_n=4)

    assert result is not None
    assert result[0][0] == 2, "the scored passage leads"
    assert sorted(index for index, _ in result) == [0, 1, 2, 3]
