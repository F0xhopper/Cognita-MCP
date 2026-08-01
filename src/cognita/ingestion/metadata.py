"""Read a book's own metadata so the caller does not have to type it.

Every extractor is best-effort: anything it cannot determine is left as None and
filled from the filename or from whatever the caller supplied. Values the caller
passed explicitly always win — see :func:`merge_metadata`.
"""

import re
from pathlib import Path

from cognita.books.domain import BookMetadata
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_YEAR = re.compile(r"(1[0-9]{3}|20[0-9]{2})")


def _year_in(value: str | None) -> int | None:
    """First four-digit year in `value`, if there is one."""
    if not value:
        return None
    match = _YEAR.search(value)
    return int(match.group(1)) if match else None


def extract_metadata(path: Path) -> BookMetadata:
    """Pull what metadata the file itself declares. Never raises."""
    ext = path.suffix.lower().lstrip(".")
    try:
        if ext == "pdf":
            meta = _from_pdf(path)
        elif ext == "epub":
            meta = _from_epub(path)
        elif ext in ("md", "markdown"):
            meta = _from_markdown(path)
        elif ext in ("html", "htm"):
            meta = _from_html(path)
        elif ext in ("txt", "text"):
            meta = _from_txt(path)
        else:
            meta = BookMetadata(title="")
    except Exception as exc:  # noqa: BLE001 — metadata is a nicety, never a blocker
        logger.warning("Metadata extraction failed for %s: %s", path.name, exc)
        meta = BookMetadata(title="")

    if not meta.title:
        meta.title = title_from_filename(path)
    return meta


def title_from_filename(path: Path) -> str:
    """Turn 'meditations_marcus-aurelius (1).pdf' into 'Meditations Marcus Aurelius'."""
    stem = path.stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)          # trailing "(1)" from downloads
    stem = re.sub(r"[_\-.]+", " ", stem)
    stem = re.sub(r"\s{2,}", " ", stem).strip()
    return stem.title() if stem.islower() else stem or path.name


def merge_metadata(extracted: BookMetadata, supplied: BookMetadata | None) -> BookMetadata:
    """Overlay caller-supplied fields on top of extracted ones. Caller wins."""
    if supplied is None:
        return extracted
    return BookMetadata(
        title=supplied.title or extracted.title,
        author=supplied.author or extracted.author,
        year=supplied.year or extracted.year,
        publisher=supplied.publisher or extracted.publisher,
        language=supplied.language or extracted.language or "en",
        isbn=supplied.isbn or extracted.isbn,
        description=supplied.description or extracted.description,
        tags=supplied.tags or extracted.tags,
        source=supplied.source or extracted.source,
    )


# ── Per-format extractors ─────────────────────────────────────────────────────

def _from_pdf(path: Path) -> BookMetadata:
    from pypdf import PdfReader

    info = PdfReader(str(path)).metadata or {}
    return BookMetadata(
        title=_clean(info.get("/Title")),
        author=_clean(info.get("/Author")) or None,
        year=_year_in(str(info.get("/CreationDate") or "")),
        publisher=_clean(info.get("/Producer")) or None,
        description=_clean(info.get("/Subject")) or None,
    )


def _from_epub(path: Path) -> BookMetadata:
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    def dc(name: str) -> str | None:
        values = book.get_metadata("DC", name)
        return _clean(values[0][0]) if values else None

    identifier = dc("identifier") or ""
    isbn = None
    if match := re.search(r"(97[89][\d-]{10,15}|\d{9}[\dXx])", identifier):
        isbn = match.group(1).replace("-", "")

    return BookMetadata(
        title=dc("title") or "",
        author=dc("creator"),
        year=_year_in(dc("date")),
        publisher=dc("publisher"),
        language=(dc("language") or "en")[:2],
        isbn=isbn,
        description=dc("description"),
    )


_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---", re.DOTALL)
_FM_FIELD = re.compile(r"^(\w+)\s*:\s*(.+?)\s*$", re.MULTILINE)
_MD_H1 = re.compile(r"^#\s+(.+?)\s*#*$", re.MULTILINE)


def _from_markdown(path: Path) -> BookMetadata:
    text = path.read_text(encoding="utf-8", errors="replace")[:8000]

    fields: dict[str, str] = {}
    if fm := _FRONTMATTER.match(text):
        fields = {
            k.lower(): v.strip("\"'")
            for k, v in _FM_FIELD.findall(fm.group(1))
        }

    title = fields.get("title") or ""
    if not title and (h1 := _MD_H1.search(text)):
        title = h1.group(1).strip()

    tags_raw = fields.get("tags", "")
    tags = [t.strip() for t in re.split(r"[,\[\]]", tags_raw) if t.strip()]

    return BookMetadata(
        title=title,
        author=fields.get("author") or fields.get("creator"),
        year=_year_in(fields.get("date") or fields.get("year")),
        description=fields.get("description") or fields.get("summary"),
        tags=tags,
    )


def _from_html(path: Path) -> BookMetadata:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "lxml")

    def meta_content(*names: str) -> str | None:
        for name in names:
            tag = soup.find("meta", attrs={"name": name}) or soup.find(
                "meta", attrs={"property": name}
            )
            if tag and tag.get("content"):
                return _clean(tag["content"])
        return None

    title = _clean(soup.title.string) if soup.title and soup.title.string else ""
    if not title and (h1 := soup.find("h1")):
        title = h1.get_text(strip=True)

    return BookMetadata(
        title=title,
        author=meta_content("author", "article:author", "citation_author"),
        year=_year_in(
            meta_content("date", "article:published_time", "citation_publication_date")
        ),
        description=meta_content("description", "og:description"),
    )


# Project Gutenberg plain-text files open with a "Title: ... Author: ..." block.
_GUTENBERG_FIELD = re.compile(
    r"^(Title|Author|Release Date|Language|Translator)\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _from_txt(path: Path) -> BookMetadata:
    header = path.read_text(encoding="utf-8", errors="replace")[:4000]
    fields = {k.lower(): v.strip() for k, v in _GUTENBERG_FIELD.findall(header)}

    language = (fields.get("language") or "").strip().lower()
    return BookMetadata(
        title=fields.get("title", ""),
        author=fields.get("author"),
        year=_year_in(fields.get("release date")),
        language="en" if language.startswith("english") else (language[:2] or "en"),
    )


def _clean(value: object) -> str:
    """Normalise whitespace and drop the empty placeholders producers leave behind."""
    if not value:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in ("untitled", "unknown", "none", "n/a", "microsoft word"):
        return ""
    # Producer tools often leave the source filename as the title.
    if text.lower().endswith((".pdf", ".docx", ".doc", ".indd")):
        return ""
    return text
