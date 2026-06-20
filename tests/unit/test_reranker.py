"""Unit tests for the reranker's graceful-fallback behaviour.

The reranker must never break search: when disabled, unconfigured, or given a
trivial input it returns the candidates in their original (RRF) order without
calling the model.
"""

from cognita.core.config import settings
from cognita.infrastructure import reranker


async def test_returns_identity_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    out = await reranker.rerank("q", ["a", "b", "c"], top_n=2)

    assert out == [(0, 0.0), (1, 0.0)]


async def test_returns_identity_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

    out = await reranker.rerank("q", ["a", "b", "c", "d"], top_n=10)

    # capped at the number of documents, original order preserved
    assert out == [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]


async def test_single_document_skips_model(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    def _boom():
        raise AssertionError("client should not be constructed for n<=1")

    monkeypatch.setattr(reranker, "get_anthropic_client", _boom)

    out = await reranker.rerank("q", ["only one"], top_n=5)

    assert out == [(0, 0.0)]
