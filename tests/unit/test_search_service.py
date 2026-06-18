"""Unit tests for citation building in search service."""


from cognita.chunks.domain import Chunk, ChunkLevel, ChunkLocation
from cognita.search.service import _build_citation


def _make_chunk(**kwargs) -> Chunk:
    defaults = dict(
        id=1, book_id=1, user_id="u", text="text", level=ChunkLevel.PARAGRAPH,
        sequence=0,
        location=ChunkLocation(),
        token_count=10,
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def test_citation_with_all_fields():
    chunk = _make_chunk(
        location=ChunkLocation(
            chapter_title="Chapter 1",
            section_title="Section A",
            page_start=42,
            page_end=43,
        )
    )
    cit = _build_citation(chunk, "My Book")
    s = cit.to_string()
    assert "My Book" in s
    assert "Chapter 1" in s
    assert "Section A" in s
    assert "42" in s


def test_citation_page_range_single_page():
    chunk = _make_chunk(location=ChunkLocation(page_start=10, page_end=10))
    cit = _build_citation(chunk, "Book")
    assert "p. 10" in cit.to_string()


def test_citation_page_range_multi_page():
    chunk = _make_chunk(location=ChunkLocation(page_start=10, page_end=12))
    cit = _build_citation(chunk, "Book")
    assert "pp. 10" in cit.to_string()


def test_citation_no_location():
    chunk = _make_chunk(location=ChunkLocation())
    cit = _build_citation(chunk, "Minimal Book")
    assert "Minimal Book" in cit.to_string()
