"""Unit tests for the query planner.

Tests cover:
- Fallback when the Anthropic API key is not configured
- Fallback when the model call raises an exception
- Happy-path parsing of a well-formed tool-use response
- Specialty list formatting (empty and non-empty)
- Minimum-viable subquery guarantee (fallback always has at least one)
"""

import pytest

from cognita.core.config import settings
from cognita.tools import planner


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tool_use_block(input_data: dict):
    """Minimal stand-in for an Anthropic tool_use content block."""
    class _Block:
        type = "tool_use"
        input = input_data
    return _Block()


def _make_response(*blocks):
    class _Resp:
        content = list(blocks)
    return _Resp()


# ── fallback paths ────────────────────────────────────────────────────────────

async def test_fallback_when_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

    result = await planner.plan_research("What is virtue?", [])

    assert result["specialty_id"] is None
    assert result["subqueries"] == ["What is virtue?"]
    assert result["depth"] == "medium"
    assert result["needs_chapter_scan"] is False


async def test_fallback_when_client_raises(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    class _BrokenClient:
        class messages:
            @staticmethod
            async def create(**_kwargs):
                raise RuntimeError("network error")

    monkeypatch.setattr(planner, "get_anthropic_client", lambda: _BrokenClient())

    result = await planner.plan_research("explain causality", [{"id": 1, "name": "Physics"}])

    assert result["subqueries"] == ["explain causality"]
    assert result["specialty_id"] is None


# ── happy-path parsing ────────────────────────────────────────────────────────

async def test_parses_tool_use_response(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "PLANNER_MODEL", "claude-haiku-4-5-20251001")

    plan_input = {
        "specialty_id": 3,
        "intent": "definition",
        "subqueries": [
            "virtue as stable disposition character",
            "moral virtue habit excellence phronesis",
        ],
        "depth": "medium",
        "needs_chapter_scan": False,
    }

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**_kwargs):
                return _make_response(_make_tool_use_block(plan_input))

    monkeypatch.setattr(planner, "get_anthropic_client", lambda: _FakeClient())

    result = await planner.plan_research(
        "What is virtue according to Aristotle?",
        [{"id": 3, "name": "Aristotle Ethics", "description": "Nicomachean Ethics corpus"}],
    )

    assert result["specialty_id"] == 3
    assert result["intent"] == "definition"
    assert len(result["subqueries"]) == 2
    assert result["depth"] == "medium"
    assert result["needs_chapter_scan"] is False


async def test_deep_plan_with_chapter_scan(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    plan_input = {
        "specialty_id": None,
        "intent": "thematic",
        "subqueries": ["soul immortality Plato", "Forms theory knowledge recollection"],
        "depth": "deep",
        "needs_chapter_scan": True,
    }

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**_kwargs):
                return _make_response(_make_tool_use_block(plan_input))

    monkeypatch.setattr(planner, "get_anthropic_client", lambda: _FakeClient())

    result = await planner.plan_research("Summarise Plato's theory of Forms", [])

    assert result["needs_chapter_scan"] is True
    assert result["depth"] == "deep"
    assert result["specialty_id"] is None


# ── specialty listing ─────────────────────────────────────────────────────────

async def test_no_specialties_uses_library_wide_fallback(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    captured: list[str] = []

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                captured.append(kwargs["messages"][0]["content"])
                raise RuntimeError("stop here")

    monkeypatch.setattr(planner, "get_anthropic_client", lambda: _FakeClient())

    await planner.plan_research("anything", [])

    assert captured, "expected at least one messages.create call"
    assert "none" in captured[0].lower() or "whole library" in captured[0].lower()


async def test_specialties_included_in_prompt(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    captured: list[str] = []

    class _FakeClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                captured.append(kwargs["messages"][0]["content"])
                raise RuntimeError("stop here")

    monkeypatch.setattr(planner, "get_anthropic_client", lambda: _FakeClient())

    await planner.plan_research(
        "q",
        [
            {"id": 1, "name": "Stoics", "description": "Stoic philosophy"},
            {"id": 2, "name": "Epistemology", "description": None},
        ],
    )

    assert "Stoics" in captured[0]
    assert "Epistemology" in captured[0]
    assert "id=1" in captured[0]
    assert "id=2" in captured[0]


# ── fallback guarantees ───────────────────────────────────────────────────────

def test_fallback_preserves_question_as_subquery():
    question = "What did Kant say about free will?"
    result = planner._fallback(question)

    assert result["subqueries"] == [question]
    assert len(result["subqueries"]) >= 1
