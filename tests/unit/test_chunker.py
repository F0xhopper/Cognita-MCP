"""The chunker turns headings into sections and text into citable chunks."""

from cognita.core.config import settings
from cognita.ingestion.chunker import _pack, _paragraph_spans, build_chunks
from cognita.ingestion.parsers import Heading, ParsedDocument


def _doc(text: str, headings: list[Heading] | None = None, page_starts=None) -> ParsedDocument:
    return ParsedDocument(
        raw_text=text,
        headings=headings or [],
        page_starts=page_starts or [],
    )


# ── Paragraph spans ───────────────────────────────────────────────────────────

def test_paragraph_spans_are_exact_offsets_into_the_source():
    text = "First para.\n\nSecond para.\n\nThird para."
    spans = _paragraph_spans(text, base=0)

    assert [s[0] for s in spans] == ["First para.", "Second para.", "Third para."]
    for content, start, end in spans:
        assert text[start:end].strip() == content


def test_paragraph_spans_offset_by_base():
    assert _paragraph_spans("Only para.", base=1000)[0][1] == 1000


def test_paragraph_spans_ignore_blank_runs():
    assert len(_paragraph_spans("A.\n\n\n\n\nB.", base=0)) == 2


# ── Packing ───────────────────────────────────────────────────────────────────

def test_pack_respects_the_size_budget():
    paragraphs = _paragraph_spans("\n\n".join(["word " * 20] * 10), base=0)
    packed = _pack(paragraphs, max_chars=300, overlap_chars=0)

    assert len(packed) > 1
    assert all(len(text) <= 320 for text, _, _ in packed)


def test_pack_overlaps_consecutive_chunks():
    paragraphs = _paragraph_spans("\n\n".join(f"Paragraph number {i}." for i in range(12)), base=0)
    packed = _pack(paragraphs, max_chars=60, overlap_chars=40)

    assert len(packed) > 2
    first_tail = packed[0][0].split("\n\n")[-1]
    assert first_tail in packed[1][0], "the next chunk should carry the previous tail"


def test_pack_keeps_an_oversized_paragraph_whole():
    """Splitting mid-sentence is worse than one chunk over budget."""
    long_para = "x" * 5000
    packed = _pack(_paragraph_spans(long_para, base=0), max_chars=1000, overlap_chars=100)

    assert len(packed) == 1
    assert packed[0][0] == long_para


def test_pack_spans_bracket_their_text():
    text = "\n\n".join(f"Para {i} with some words." for i in range(8))
    for chunk_text, start, end in _pack(_paragraph_spans(text, base=0), 80, 20):
        assert start < end
        assert chunk_text.split("\n\n")[0] in text[start:end]


# ── Sections ──────────────────────────────────────────────────────────────────

def test_headings_become_numbered_chapters_and_sections():
    text = (
        "Chapter One\nOpening words here.\n\n"
        "Section A\nSection body text.\n\n"
        "Chapter Two\nSecond chapter body.\n\n"
    )
    headings = [
        Heading("Chapter One", 1, text.index("Chapter One")),
        Heading("Section A", 2, text.index("Section A")),
        Heading("Chapter Two", 1, text.index("Chapter Two")),
    ]
    chunks = build_chunks(_doc(text, headings), book_id=1)

    located = {
        (c.location.chapter_n, c.location.section_n, c.location.chapter_title) for c in chunks
    }
    assert (1, 1, "Chapter One") in located
    assert (1, 2, "Chapter One") in located, "a section belongs to its chapter"
    assert (2, 1, "Chapter Two") in located


def test_section_titles_survive_onto_chunks():
    text = "Alpha\nBody of alpha.\n\nBeta\nBody of beta.\n\n"
    headings = [Heading("Alpha", 1, 0), Heading("Beta", 1, text.index("Beta"))]
    titles = {c.location.section_title for c in build_chunks(_doc(text, headings), book_id=1)}

    assert titles == {"Alpha", "Beta"}


def test_text_before_the_first_heading_is_kept():
    text = "An epigraph nobody wants to lose.\n\nChapter One\nThe body.\n\n"
    headings = [Heading("Chapter One", 1, text.index("Chapter One"))]
    chunks = build_chunks(_doc(text, headings), book_id=1)

    assert any("epigraph" in c.text for c in chunks)
    assert any(c.location.chapter_title == "Front Matter" for c in chunks)


def test_document_without_headings_gets_one_section():
    chunks = build_chunks(_doc("Some prose.\n\nMore prose.\n\n" * 5), book_id=1)

    assert chunks
    assert {c.location.chapter_title for c in chunks} == {"Full Text"}


def test_headings_outside_the_text_are_ignored():
    text = "Short body.\n\n"
    headings = [Heading("Phantom", 1, 9_999), Heading("Real", 1, 0)]
    chunks = build_chunks(_doc(text, headings), book_id=1)

    assert {c.location.section_title for c in chunks} == {"Real"}


# ── Chunk properties ──────────────────────────────────────────────────────────

def test_sequences_are_dense_and_ordered():
    chunks = build_chunks(_doc("Paragraph text here.\n\n" * 30), book_id=3)
    assert [c.sequence for c in chunks] == list(range(len(chunks)))


def test_chunks_carry_their_character_span():
    text = "\n\n".join(f"Paragraph {i} of the book." for i in range(20))
    for chunk in build_chunks(_doc(text), book_id=1):
        start, end = chunk.location.char_start, chunk.location.char_end
        assert start is not None and end > start
        # The span must actually contain the chunk's opening paragraph.
        assert chunk.text.split("\n\n")[0] in text[start:end]


def test_pages_are_derived_from_character_spans():
    pages = ["Page one text.\n\nStill page one.", "Page two text.", "Page three text."]
    raw = "\n\n".join(pages)
    starts, cursor = [], 0
    for page in pages:
        starts.append(cursor)
        cursor += len(page) + 2

    chunks = build_chunks(_doc(raw, page_starts=starts), book_id=1)

    assert all(c.location.page_start is not None for c in chunks)
    assert min(c.location.page_start for c in chunks) == 1
    assert max(c.location.page_end for c in chunks) <= len(pages)


def test_no_pages_means_no_page_citation():
    chunks = build_chunks(_doc("Body text.\n\nMore body.\n\n"), book_id=1)
    assert all(c.location.page_start is None for c in chunks)


def test_empty_document_produces_no_chunks():
    assert build_chunks(_doc("   \n\n  \n"), book_id=1) == []


def test_chunk_size_follows_settings(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE_CHARS", 200)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP_CHARS", 0)

    text = "\n\n".join("A sentence of moderate length here." for _ in range(40))
    chunks = build_chunks(_doc(text), book_id=1)

    assert len(chunks) > 4
    assert all(len(c.text) <= 240 for c in chunks)
