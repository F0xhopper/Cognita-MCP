"""Metadata extraction — what a caller does not have to type in."""

from cognita.books.domain import BookMetadata
from cognita.ingestion.metadata import (
    _clean,
    extract_metadata,
    merge_metadata,
    title_from_filename,
)


def _write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ── Filenames ─────────────────────────────────────────────────────────────────

def test_filename_becomes_a_readable_title(tmp_path):
    assert title_from_filename(tmp_path / "the_republic.pdf") == "The Republic"
    assert title_from_filename(tmp_path / "debt-the-first-5000-years.epub") == (
        "Debt The First 5000 Years"
    )


def test_filename_drops_the_download_counter(tmp_path):
    assert title_from_filename(tmp_path / "meditations (2).pdf") == "Meditations"


def test_filename_leaves_existing_capitalisation_alone(tmp_path):
    assert title_from_filename(tmp_path / "GEB An Eternal Golden Braid.pdf") == (
        "GEB An Eternal Golden Braid"
    )


# ── Markdown ──────────────────────────────────────────────────────────────────

def test_markdown_frontmatter_is_read(tmp_path):
    path = _write(
        tmp_path,
        "essay.md",
        "---\ntitle: On Certainty\nauthor: Wittgenstein\ndate: 1969-01-01\n"
        "tags: [philosophy, epistemology]\n---\n\nBody text.\n",
    )
    meta = extract_metadata(path)

    assert meta.title == "On Certainty"
    assert meta.author == "Wittgenstein"
    assert meta.year == 1969
    assert "philosophy" in meta.tags


def test_markdown_falls_back_to_the_first_heading(tmp_path):
    path = _write(tmp_path, "notes.md", "# Reading Notes\n\nSome body.\n")
    assert extract_metadata(path).title == "Reading Notes"


def test_markdown_without_either_uses_the_filename(tmp_path):
    path = _write(tmp_path, "loose_notes.md", "Just body text, no heading.\n")
    assert extract_metadata(path).title == "Loose Notes"


# ── HTML ──────────────────────────────────────────────────────────────────────

def test_html_reads_title_and_meta_tags(tmp_path):
    path = _write(
        tmp_path,
        "article.html",
        '<html><head><title>An Article</title>'
        '<meta name="author" content="A Writer">'
        '<meta name="description" content="About things.">'
        '<meta name="date" content="2021-06-01"></head><body><p>Body.</p></body></html>',
    )
    meta = extract_metadata(path)

    assert meta.title == "An Article"
    assert meta.author == "A Writer"
    assert meta.description == "About things."
    assert meta.year == 2021


def test_html_falls_back_to_the_first_h1(tmp_path):
    path = _write(tmp_path, "page.html", "<html><body><h1>Headline</h1></body></html>")
    assert extract_metadata(path).title == "Headline"


# ── Plain text / Gutenberg ────────────────────────────────────────────────────

def test_gutenberg_header_is_parsed(tmp_path):
    path = _write(
        tmp_path,
        "pg1234.txt",
        "Title: The Republic\n"
        "Author: Plato\n"
        "Release Date: October 1998 [eBook #1497]\n"
        "Language: English\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK ***\n\nBody.\n",
    )
    meta = extract_metadata(path)

    assert meta.title == "The Republic"
    assert meta.author == "Plato"
    assert meta.year == 1998
    assert meta.language == "en"


def test_plain_text_without_a_header_uses_the_filename(tmp_path):
    path = _write(tmp_path, "shopping_list.txt", "Milk.\nBread.\n")
    assert extract_metadata(path).title == "Shopping List"


# ── Merging ───────────────────────────────────────────────────────────────────

def test_supplied_metadata_overrides_extracted():
    merged = merge_metadata(
        BookMetadata(title="From File", author="File Author", year=1900),
        BookMetadata(title="From Caller", author=None),
    )

    assert merged.title == "From Caller"
    assert merged.author == "File Author", "an unset caller field keeps the extracted value"
    assert merged.year == 1900


def test_merging_with_nothing_supplied_returns_the_extraction():
    extracted = BookMetadata(title="Only Source")
    assert merge_metadata(extracted, None) is extracted


def test_merged_tags_prefer_the_caller():
    merged = merge_metadata(
        BookMetadata(title="T", tags=["from-file"]),
        BookMetadata(title="", tags=["mine"]),
    )
    assert merged.tags == ["mine"]


# ── Cleaning ──────────────────────────────────────────────────────────────────

def test_clean_rejects_producer_placeholders():
    assert _clean("untitled") == ""
    assert _clean("Microsoft Word") == ""
    assert _clean("manuscript_final.docx") == ""
    assert _clean(None) == ""


def test_clean_normalises_whitespace():
    assert _clean("  The   Real\n Title ") == "The Real Title"


# ── Robustness ────────────────────────────────────────────────────────────────

def test_a_corrupt_file_still_yields_a_title(tmp_path):
    path = tmp_path / "broken_book.pdf"
    path.write_bytes(b"this is definitely not a pdf")

    assert extract_metadata(path).title == "Broken Book"


def test_unknown_extension_falls_back_to_the_filename(tmp_path):
    path = _write(tmp_path, "mystery_file.xyz", "content")
    assert extract_metadata(path).title == "Mystery File"
