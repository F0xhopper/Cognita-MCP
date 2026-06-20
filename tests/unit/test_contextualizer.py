"""Unit tests for contextualization's no-op paths and section reconstruction."""

from cognita.chunks.domain import Chunk, ChunkLevel, ChunkLocation
from cognita.core.config import settings
from cognita.ingestion import contextualizer


def _chunk(text: str, chapter_n: int, section_n: int, seq: int) -> Chunk:
    return Chunk(
        id=0, book_id=1, user_id="u", text=text, level=ChunkLevel.PARAGRAPH,
        sequence=seq,
        location=ChunkLocation(chapter_n=chapter_n, section_n=section_n),
    )


async def test_empty_input_returns_empty():
    assert await contextualizer.contextualize_chunks([], "Title", None) == []


async def test_disabled_returns_blank_contexts(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_ENABLED", False)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    chunks = [_chunk("a", 1, 1, 0), _chunk("b", 1, 1, 1)]
    out = await contextualizer.contextualize_chunks(chunks, "Title", "Author")

    assert out == ["", ""]


async def test_missing_api_key_returns_blank_contexts(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_ENABLED", True)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

    chunks = [_chunk("a", 1, 1, 0)]
    out = await contextualizer.contextualize_chunks(chunks, "Title", None)

    assert out == [""]


def test_section_index_groups_and_caps(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_MAX_CHARS", 100)
    chunks = [
        _chunk("alpha", 1, 1, 0),
        _chunk("beta", 1, 1, 1),
        _chunk("gamma", 2, 1, 2),
    ]
    idx = contextualizer._build_section_index(chunks)

    assert idx[(1, 1)] == "alpha\n\nbeta"
    assert idx[(2, 1)] == "gamma"
    assert all(len(v) <= 100 for v in idx.values())
