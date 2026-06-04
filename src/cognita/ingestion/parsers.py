"""Document parsers: PDF, EPUB, TXT → (raw_text, detected_toc)."""

from dataclasses import dataclass
from pathlib import Path

from cognita.core.exceptions import UnsupportedFormatError
from cognita.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    raw_text: str
    pages: list[str]               # one string per page (empty for non-PDF)
    native_toc: list[dict]         # [{title, level, page, char_offset}] if available
    is_scanned: bool = False       # flag for Mistral OCR fallback


def parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    raw_text = "\n\n".join(pages)
    is_scanned = len(raw_text.strip()) < 200 and len(reader.pages) > 2

    native_toc: list[dict] = []
    try:
        outline = reader.outline
        if outline:
            _flatten_outline(outline, native_toc, level=1)
    except Exception:
        pass

    return ParsedDocument(
        raw_text=raw_text,
        pages=pages,
        native_toc=native_toc,
        is_scanned=is_scanned,
    )


def _flatten_outline(items, toc: list[dict], level: int) -> None:
    for item in items:
        if isinstance(item, list):
            _flatten_outline(item, toc, level + 1)
        else:
            try:
                toc.append({
                    "title": item.title,
                    "level": level,
                    "page": None,
                    "char_offset": None,
                })
            except AttributeError:
                pass


def parse_epub(path: Path) -> ParsedDocument:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters: list[str] = []
    native_toc: list[dict] = []
    chapter_n = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        headings = soup.find_all(["h1", "h2", "h3"])
        for h in headings:
            level = int(h.name[1])
            native_toc.append({
                "title": h.get_text(strip=True),
                "level": level,
                "page": None,
                "char_offset": None,
            })
        text = soup.get_text(separator="\n", strip=True)
        if text.strip():
            chapters.append(text)
            chapter_n += 1

    raw_text = "\n\n".join(chapters)
    return ParsedDocument(
        raw_text=raw_text,
        pages=[],
        native_toc=native_toc,
        is_scanned=False,
    )


def parse_txt(path: Path) -> ParsedDocument:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDocument(
        raw_text=raw_text,
        pages=[],
        native_toc=[],
        is_scanned=False,
    )


def parse_document(path: Path) -> ParsedDocument:
    ext = path.suffix.lower().lstrip(".")
    if ext == "pdf":
        return parse_pdf(path)
    if ext == "epub":
        return parse_epub(path)
    if ext == "txt":
        return parse_txt(path)
    raise UnsupportedFormatError(ext)
