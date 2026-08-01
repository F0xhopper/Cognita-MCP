"""Parsers must locate headings at real character offsets.

This is load-bearing: the chunker drops any heading without a usable offset, so
a parser that returns headings at offset 0 silently collapses a whole book into
one undifferentiated section.
"""

import pytest

from cognita.core.exceptions import UnsupportedFormatError
from cognita.ingestion.parsers import (
    ParsedDocument,
    detect_headings,
    parse_document,
    parse_html,
    parse_markdown,
    parse_txt,
)


def _write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ── Markdown ──────────────────────────────────────────────────────────────────

def test_markdown_headings_point_at_their_own_text(tmp_path):
    path = _write(
        tmp_path, "notes.md", "# One\n\nBody one.\n\n## Sub\n\nBody sub.\n\n# Two\n\nBody two.\n"
    )
    doc = parse_markdown(path)

    assert [h.title for h in doc.headings] == ["One", "Sub", "Two"]
    assert [h.level for h in doc.headings] == [1, 2, 1]
    for heading in doc.headings:
        assert doc.raw_text[heading.char_offset:].startswith("#")
        assert heading.title in doc.raw_text[heading.char_offset : heading.char_offset + 40]


def test_markdown_frontmatter_is_not_part_of_the_body(tmp_path):
    path = _write(
        tmp_path,
        "post.md",
        "---\ntitle: A Post\nauthor: Someone\n---\n\n# Heading\n\nBody.\n",
    )
    doc = parse_markdown(path)

    assert "title: A Post" not in doc.raw_text
    assert doc.headings[0].title == "Heading"
    assert doc.raw_text[doc.headings[0].char_offset:].startswith("# Heading")


def test_markdown_deep_headings_are_capped_at_level_three(tmp_path):
    path = _write(tmp_path, "d.md", "###### Deep\n\nBody.\n")
    assert parse_markdown(path).headings[0].level == 3


# ── HTML ──────────────────────────────────────────────────────────────────────

def test_html_headings_resolve_in_document_order(tmp_path):
    path = _write(
        tmp_path,
        "page.html",
        "<html><body><h1>First</h1><p>Alpha.</p>"
        "<h2>Second</h2><p>Beta.</p>"
        "<h1>Third</h1><p>Gamma.</p></body></html>",
    )
    doc = parse_html(path)

    assert [h.title for h in doc.headings] == ["First", "Second", "Third"]
    offsets = [h.char_offset for h in doc.headings]
    assert offsets == sorted(offsets), "headings must advance through the text"
    for heading in doc.headings:
        assert doc.raw_text[heading.char_offset:].startswith(heading.title)


def test_html_repeated_heading_text_does_not_rewind(tmp_path):
    """A heading whose text appears twice must resolve to its own occurrence."""
    path = _write(
        tmp_path,
        "repeat.html",
        "<html><body><h1>Notes</h1><p>Body.</p><h1>Notes</h1><p>More.</p></body></html>",
    )
    doc = parse_html(path)

    assert len(doc.headings) == 2
    assert doc.headings[0].char_offset < doc.headings[1].char_offset


def test_html_scripts_and_styles_are_dropped(tmp_path):
    path = _write(
        tmp_path,
        "noisy.html",
        "<html><body><script>var x=1;</script><style>p{}</style>"
        "<h1>Real</h1><p>Content.</p></body></html>",
    )
    doc = parse_html(path)

    assert "var x" not in doc.raw_text
    assert "Content." in doc.raw_text


# ── Plain text ────────────────────────────────────────────────────────────────

def test_txt_strips_gutenberg_boilerplate(tmp_path):
    path = _write(
        tmp_path,
        "book.txt",
        "Title: A Book\n\nLicence blather that goes on.\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK A BOOK ***\n\n"
        "CHAPTER I\n\nThe real text begins.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK A BOOK ***\n\n"
        "More licence blather.\n",
    )
    doc = parse_txt(path)

    assert "Licence blather" not in doc.raw_text
    assert "More licence blather" not in doc.raw_text
    assert "The real text begins." in doc.raw_text


def test_txt_without_gutenberg_markers_is_untouched(tmp_path):
    path = _write(tmp_path, "plain.txt", "Just some notes.\n\nA second paragraph.\n")
    assert "Just some notes." in parse_txt(path).raw_text


# ── Heuristic detection ───────────────────────────────────────────────────────

def test_detect_headings_finds_chapters():
    text = "Chapter I\n\nOpening.\n\nChapter II\n\nContinuing.\n"
    headings = detect_headings(text)

    assert [h.title for h in headings] == ["Chapter I", "Chapter II"]
    assert all(text[h.char_offset:].startswith(h.title) for h in headings)


def test_detect_headings_returns_reading_order():
    text = "CHAPTER ONE\n\nBody.\n\nA SHOUTED SECTION\n\nMore body.\n\nChapter 2. Later\n\nEnd.\n"
    offsets = [h.char_offset for h in detect_headings(text)]
    assert offsets == sorted(offsets)


def test_detect_headings_on_unstructured_text_is_empty():
    assert detect_headings("just one long sentence with nothing structural in it") == []


# ── Page mapping ──────────────────────────────────────────────────────────────

def test_page_for_offset_maps_into_the_right_page():
    # Three 10-char pages joined by "\n\n": starts at 0, 12, 24.
    doc = ParsedDocument(raw_text="x" * 34, page_starts=[0, 12, 24])

    assert doc.page_for_offset(0) == 1
    assert doc.page_for_offset(11) == 1
    assert doc.page_for_offset(12) == 2
    assert doc.page_for_offset(30) == 3


def test_page_for_offset_is_none_without_pagination():
    doc = ParsedDocument(raw_text="text")
    assert doc.page_for_offset(2) is None
    assert doc.page_for_offset(None) is None


# ── Dispatch ──────────────────────────────────────────────────────────────────

def test_parse_document_dispatches_on_extension(tmp_path):
    path = _write(tmp_path, "a.md", "# Title\n\nBody.\n")
    assert parse_document(path).headings[0].title == "Title"


def test_parse_document_rejects_unknown_formats(tmp_path):
    path = _write(tmp_path, "book.docx", "nope")
    with pytest.raises(UnsupportedFormatError):
        parse_document(path)
