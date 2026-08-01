"""Document parsers: PDF, EPUB, TXT, Markdown, HTML → text + structure.

Every parser returns a :class:`ParsedDocument` whose ``headings`` carry a real
``char_offset`` into ``raw_text``. That offset is what makes the chunker able to
split a book into chapters and sections; a heading without one is useless and is
dropped. PDFs additionally return ``page_spans`` so chunks can be mapped back to
page numbers for citation.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from cognita.core.exceptions import UnsupportedFormatError
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_PAGE_JOIN = "\n\n"


@dataclass
class Heading:
    """A structural heading located at a known point in the text."""

    title: str
    level: int  # 1 = chapter, 2 = section, 3 = subsection
    char_offset: int
    page: int | None = None


@dataclass
class ParsedDocument:
    raw_text: str
    headings: list[Heading] = field(default_factory=list)
    # Start offset of each page within raw_text; empty for non-paginated formats.
    page_starts: list[int] = field(default_factory=list)
    is_scanned: bool = False

    def page_for_offset(self, offset: int | None) -> int | None:
        """1-indexed page containing `offset`, or None if the format has no pages."""
        if offset is None or not self.page_starts:
            return None
        return bisect_right(self.page_starts, offset)


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]

    page_starts: list[int] = []
    cursor = 0
    for text in pages:
        page_starts.append(cursor)
        cursor += len(text) + len(_PAGE_JOIN)

    raw_text = _PAGE_JOIN.join(pages)
    # A PDF of scanned images extracts almost nothing; that is the OCR signal.
    is_scanned = len(raw_text.strip()) < 200 and len(pages) > 2

    headings = _pdf_headings(reader, pages, page_starts, raw_text)
    return ParsedDocument(
        raw_text=raw_text,
        headings=headings,
        page_starts=page_starts,
        is_scanned=is_scanned,
    )


def _pdf_headings(reader, pages: list[str], page_starts: list[int], raw_text: str) -> list[Heading]:
    """Turn the PDF outline into headings with real character offsets.

    An outline entry knows its destination *page*, not its character position.
    We resolve the page to a character range and then look for the entry's title
    inside it; if the title is not literally present (common — outline text is
    often reformatted), we fall back to the top of the page, which is accurate
    enough to split chapters.
    """
    try:
        outline = reader.outline
    except Exception as exc:  # noqa: BLE001 — a broken outline must not fail the parse
        logger.debug("No usable PDF outline: %s", exc)
        return detect_headings(raw_text)

    if not outline:
        return detect_headings(raw_text)

    headings: list[Heading] = []

    def walk(items, level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            title = getattr(item, "title", None)
            if not title:
                continue
            try:
                page_n = reader.get_destination_page_number(item)
            except Exception:  # noqa: BLE001 — destination may be unresolvable
                continue
            if page_n is None or not 0 <= page_n < len(pages):
                continue

            page_start = page_starts[page_n]
            local = _find_title(pages[page_n], title)
            offset = page_start + local if local is not None else page_start
            headings.append(
                Heading(
                    title=title.strip(),
                    level=min(level, 3),
                    char_offset=offset,
                    page=page_n + 1,
                )
            )

    walk(outline, 1)
    if not headings:
        return detect_headings(raw_text)

    headings.sort(key=lambda h: h.char_offset)
    return headings


def _find_title(page_text: str, title: str) -> int | None:
    """Locate `title` in `page_text`, tolerating whitespace and case differences."""
    needle = re.sub(r"\s+", " ", title).strip()
    if not needle:
        return None
    pattern = re.compile(r"\s+".join(re.escape(w) for w in needle.split()), re.IGNORECASE)
    match = pattern.search(page_text)
    return match.start() if match else None


# ── EPUB ──────────────────────────────────────────────────────────────────────

def parse_epub(path: Path) -> ParsedDocument:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    parts: list[str] = []
    headings: list[Heading] = []
    cursor = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        text = soup.get_text(separator="\n", strip=True)
        if not text.strip():
            continue

        # Headings are located by searching the *rendered* text of this document,
        # which is the same string that ends up in raw_text — so the offset is exact.
        for tag in soup.find_all(["h1", "h2", "h3"]):
            title = tag.get_text(strip=True)
            if not title:
                continue
            local = _find_title(text, title)
            headings.append(
                Heading(
                    title=title,
                    level=int(tag.name[1]),
                    char_offset=cursor + (local if local is not None else 0),
                )
            )

        parts.append(text)
        cursor += len(text) + len(_PAGE_JOIN)

    raw_text = _PAGE_JOIN.join(parts)
    headings.sort(key=lambda h: h.char_offset)
    return ParsedDocument(
        raw_text=raw_text,
        headings=headings or detect_headings(raw_text),
    )


# ── Markdown ──────────────────────────────────────────────────────────────────

_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)
_MD_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def parse_markdown(path: Path) -> ParsedDocument:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    # Strip YAML frontmatter from the body; metadata extraction reads it separately.
    raw_text = _MD_FRONTMATTER.sub("", raw_text)

    headings = [
        Heading(
            title=m.group(2).strip(),
            level=min(len(m.group(1)), 3),
            char_offset=m.start(),
        )
        for m in _MD_HEADING.finditer(raw_text)
    ]
    return ParsedDocument(raw_text=raw_text, headings=headings)


# ── HTML ──────────────────────────────────────────────────────────────────────

def parse_html(path: Path) -> ParsedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    titles = [
        (tag.get_text(strip=True), int(tag.name[1]))
        for tag in soup.find_all(["h1", "h2", "h3"])
    ]
    raw_text = soup.get_text(separator="\n", strip=True)

    headings: list[Heading] = []
    cursor = 0
    for title, level in titles:
        if not title:
            continue
        local = _find_title(raw_text[cursor:], title)
        if local is None:
            continue
        offset = cursor + local
        headings.append(Heading(title=title, level=min(level, 3), char_offset=offset))
        cursor = offset + len(title)

    return ParsedDocument(raw_text=raw_text, headings=headings or detect_headings(raw_text))


# ── Plain text ────────────────────────────────────────────────────────────────

def parse_txt(path: Path) -> ParsedDocument:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    raw_text = _strip_gutenberg_boilerplate(raw_text)
    return ParsedDocument(raw_text=raw_text, headings=detect_headings(raw_text))


_GUTENBERG_START = re.compile(r"^\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE)
_GUTENBERG_END = re.compile(r"^\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE)


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Drop the licence header and footer Gutenberg wraps around every book.

    Left in, they are several thousand words of legal text that match nothing
    anyone would ask about but still occupy chunks and embeddings.
    """
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


# ── Heuristic structure, for text with no native outline ──────────────────────

_CHAPTER_PATTERNS = [
    re.compile(r"^(?:chapter|part|book|canto|act)\s+[\dIVXLC]+\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s{0,4}[IVXLC]{1,7}\.\s+\S.*$", re.MULTILINE),
    re.compile(r"^\s{0,4}\d{1,2}\.\s+[A-Z][A-Za-z ,'-]{3,60}$", re.MULTILINE),
]

_SECTION_PATTERNS = [
    re.compile(r"^\s{0,4}#{1,3} .+$", re.MULTILINE),
    re.compile(r"^[A-Z][A-Z\s'’-]{5,60}$", re.MULTILINE),
]


def detect_headings(text: str) -> list[Heading]:
    """Best-effort structure detection for documents with no outline at all."""
    found: dict[int, Heading] = {}
    for level, patterns in ((1, _CHAPTER_PATTERNS), (2, _SECTION_PATTERNS)):
        for pattern in patterns:
            for m in pattern.finditer(text):
                title = m.group().strip()
                # A chapter match wins over a section match at the same offset.
                if title and (m.start() not in found or level < found[m.start()].level):
                    found[m.start()] = Heading(title=title, level=level, char_offset=m.start())
    return [found[k] for k in sorted(found)]


# ── Dispatch ──────────────────────────────────────────────────────────────────

_PARSERS = {
    "pdf": parse_pdf,
    "epub": parse_epub,
    "txt": parse_txt,
    "text": parse_txt,
    "md": parse_markdown,
    "markdown": parse_markdown,
    "html": parse_html,
    "htm": parse_html,
}


def parse_document(path: Path) -> ParsedDocument:
    ext = path.suffix.lower().lstrip(".")
    parser = _PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFormatError(ext)
    return parser(path)
