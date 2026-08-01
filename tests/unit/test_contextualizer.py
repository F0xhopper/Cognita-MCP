"""Contextualization is optional and must degrade to a no-op cleanly."""

from cognita.chunks.domain import Chunk, ChunkLocation
from cognita.core.config import settings
from cognita.ingestion import contextualizer


def _chunk(text: str, chapter_n: int = 1, section_n: int = 1, sequence: int = 0) -> Chunk:
    return Chunk(
        id=0,
        book_id=1,
        text=text,
        sequence=sequence,
        location=ChunkLocation(chapter_n=chapter_n, section_n=section_n),
    )


async def test_no_chunks_means_no_work():
    assert await contextualizer.contextualize_chunks([], "Title", None) == []


async def test_disabled_returns_blanks(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_ENABLED", False)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test")

    result = await contextualizer.contextualize_chunks([_chunk("a"), _chunk("b")], "T", "A")

    assert result == ["", ""]


async def test_missing_key_returns_blanks(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_ENABLED", True)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

    assert await contextualizer.contextualize_chunks([_chunk("a")], "T", None) == [""]


async def test_result_is_aligned_with_the_input(monkeypatch):
    """The pipeline zips contexts onto chunks, so length and order must match."""
    monkeypatch.setattr(settings, "CONTEXT_ENABLED", False)
    chunks = [_chunk(f"chunk {i}", sequence=i) for i in range(7)]

    assert len(await contextualizer.contextualize_chunks(chunks, "T", None)) == len(chunks)


def test_sections_are_reassembled_from_their_chunks(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_MAX_CHARS", 100)
    index = contextualizer._build_section_index([
        _chunk("alpha", 1, 1),
        _chunk("beta", 1, 1),
        _chunk("gamma", 2, 1),
    ])

    assert index[(1, 1)] == "alpha\n\nbeta"
    assert index[(2, 1)] == "gamma"


def test_reassembled_sections_are_capped(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_MAX_CHARS", 50)
    index = contextualizer._build_section_index([_chunk("x" * 500), _chunk("y" * 500)])

    assert all(len(section) <= 50 for section in index.values())
