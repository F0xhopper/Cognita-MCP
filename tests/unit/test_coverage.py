"""Unit tests for coverage assessment.

Tests cover:
- Fallback when the Anthropic API key is not configured
- Fallback when the model call raises an exception
- Happy-path: satisfied=true with empty gaps/suggestions
- Happy-path: satisfied=false with gaps and follow-up queries
- Passage capping (only _MAX_PASSAGES passages sent to the model)
- Passage text truncation (_MAX_PASSAGE_CHARS applied per passage)
"""

from cognita.core.config import settings
from cognita.tools import coverage
from cognita.tools.coverage import _MAX_PASSAGE_CHARS, _MAX_PASSAGES


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tool_use_block(input_data: dict):
    class _Block:
        type = "tool_use"
        input = input_data
    return _Block()


def _make_response(*blocks):
    class _Resp:
        content = list(blocks)
    return _Resp()


def _passages(n: int, text: str = "some passage text") -> list[dict]:
    return [{"text": text, "citation": f"Book › Ch {i}"} for i in range(n)]


# ── fallback paths ────────────────────────────────────────────────────────────

async def test_fallback_when_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

    result = await coverage.assess_coverage("What is justice?", _passages(3))

    assert result["satisfied"] is False
    assert result["gaps"] == ["Coverage assessment unavailable"]
    assert result["suggested_queries"] == []


async def test_fallback_when_client_raises(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    class _BrokenClient:
        class messages:
            @staticmethod
            async def create(**_kwargs):
                raise ConnectionError("timeout")

    monkeypatch.setattr(coverage, "get_anthropic_client", lambda: _BrokenClient())

    result = await coverage.assess_coverage("q", _passages(2))

    assert result["satisfied"] is False
    assert result["suggested_queries"] == []


# ── happy path: satisfied ─────────────────────────────────────────────────────

async def test_satisfied_returns_empty_gaps(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    assessment = {
        "satisfied": True,
        "gaps": [],
        "suggested_queries": [],
    }

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**_kwargs):
                return _make_response(_make_tool_use_block(assessment))

    monkeypatch.setattr(coverage, "get_anthropic_client", lambda: _FakeClient())

    result = await coverage.assess_coverage(
        "What is the Stoic view of virtue?", _passages(5)
    )

    assert result["satisfied"] is True
    assert result["gaps"] == []
    assert result["suggested_queries"] == []


# ── happy path: gaps ──────────────────────────────────────────────────────────

async def test_gaps_returned_with_follow_up_queries(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    assessment = {
        "satisfied": False,
        "gaps": [
            "No passages on the role of reason in virtue",
            "Missing discussion of the mean doctrine",
        ],
        "suggested_queries": [
            "reason virtue rational activity Aristotle",
            "doctrine of the mean excess deficiency character",
        ],
    }

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**_kwargs):
                return _make_response(_make_tool_use_block(assessment))

    monkeypatch.setattr(coverage, "get_anthropic_client", lambda: _FakeClient())

    result = await coverage.assess_coverage(
        "Explain Aristotle's doctrine of the mean", _passages(3)
    )

    assert result["satisfied"] is False
    assert len(result["gaps"]) == 2
    assert len(result["suggested_queries"]) == 2


# ── passage capping ───────────────────────────────────────────────────────────

async def test_only_max_passages_sent_to_model(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    captured_prompts: list[str] = []

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                captured_prompts.append(kwargs["messages"][0]["content"])
                raise RuntimeError("stop after capture")

    monkeypatch.setattr(coverage, "get_anthropic_client", lambda: _FakeClient())

    # Send more passages than the cap
    await coverage.assess_coverage("q", _passages(_MAX_PASSAGES + 5))

    assert captured_prompts, "expected a messages.create call"
    prompt = captured_prompts[0]
    # Only indices 0 .. _MAX_PASSAGES-1 should appear
    assert f"[{_MAX_PASSAGES - 1}]" in prompt
    assert f"[{_MAX_PASSAGES}]" not in prompt


async def test_passage_text_is_truncated(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    captured_prompts: list[str] = []

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                captured_prompts.append(kwargs["messages"][0]["content"])
                raise RuntimeError("stop after capture")

    monkeypatch.setattr(coverage, "get_anthropic_client", lambda: _FakeClient())

    long_text = "x" * (_MAX_PASSAGE_CHARS + 500)
    await coverage.assess_coverage("q", [{"text": long_text, "citation": "Book › Ch 1"}])

    prompt = captured_prompts[0]
    # The truncated passage text must not exceed the cap
    assert "x" * (_MAX_PASSAGE_CHARS + 1) not in prompt
    assert "x" * _MAX_PASSAGE_CHARS in prompt


# ── fallback structure ────────────────────────────────────────────────────────

def test_fallback_structure():
    result = coverage._fallback()

    assert isinstance(result["satisfied"], bool)
    assert result["satisfied"] is False
    assert isinstance(result["gaps"], list)
    assert isinstance(result["suggested_queries"], list)
