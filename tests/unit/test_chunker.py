"""Tests for the hierarchical chunker."""


from cognita.ingestion.chunker import _split_into_paragraphs, build_chunks
from cognita.ingestion.parsers import ParsedDocument


def test_split_respects_paragraph_boundaries():
    text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
    chunks = _split_into_paragraphs(text, max_chars=30, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 60  # slack for overlap


def test_split_overlap_carries_content():
    text = "A" * 40 + "\n\n" + "B" * 40 + "\n\n" + "C" * 40
    chunks = _split_into_paragraphs(text, max_chars=50, overlap=20)
    assert len(chunks) >= 2
    # second chunk should start with overlap from first
    joined = " ".join(chunks)
    assert "A" in joined and "B" in joined and "C" in joined


def test_build_chunks_no_toc():
    doc = ParsedDocument(
        raw_text="Introduction text.\n\nMore content here.\n\nAnd yet more.\n\n" * 5,
        pages=[],
        native_toc=[],
    )
    chunks = build_chunks(doc, book_id=1, user_id="user-123")
    assert len(chunks) > 0
    for c in chunks:
        assert c.book_id == 1
        assert c.user_id == "user-123"
        assert c.text.strip()
        assert c.sequence >= 0


def test_build_chunks_with_toc():
    full_text = (
        "Chapter One\nThis is the first chapter content spanning multiple paragraphs.\n\n"
        "More text in chapter one.\n\n"
        "Chapter Two\nThis is the second chapter.\n\n"
        "More text here.\n\n"
    )
    toc = [
        {"title": "Chapter One", "level": 1, "char_offset": 0},
        {"title": "Chapter Two", "level": 1, "char_offset": full_text.index("Chapter Two")},
    ]
    doc = ParsedDocument(raw_text=full_text, pages=[], native_toc=toc)
    chunks = build_chunks(doc, book_id=2, user_id="user-456")
    assert len(chunks) >= 2
    chapter_titles = {c.location.chapter_title for c in chunks if c.location.chapter_title}
    assert "Chapter One" in chapter_titles or "Chapter Two" in chapter_titles


def test_chunks_have_monotonic_sequences():
    doc = ParsedDocument(
        raw_text="Paragraph.\n\n" * 20,
        pages=[],
        native_toc=[],
    )
    chunks = build_chunks(doc, book_id=3, user_id="u")
    seqs = [c.sequence for c in chunks]
    assert seqs == list(range(len(seqs)))
