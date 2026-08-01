"""Split a parsed document into retrievable chunks.

Two passes:

  1. **Structure** — the document's headings carve the text into sections, each
     tagged with its chapter and section title. Headings come from the PDF
     outline, EPUB/HTML heading tags, Markdown ``#`` lines, or, failing all of
     those, pattern matching over the raw text.
  2. **Chunking** — each section is packed into chunks of at most
     ``CHUNK_SIZE_CHARS``, splitting only on paragraph boundaries so a chunk is
     never cut mid-sentence. Consecutive chunks overlap by whole trailing
     paragraphs (up to ``CHUNK_OVERLAP_CHARS``) so an idea spanning a boundary
     is still retrievable from either side.

Every chunk records the exact character span it came from, which is what lets a
citation name a real page.
"""

import re

from cognita.chunks.domain import Chunk, ChunkLocation
from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.ingestion.parsers import Heading, ParsedDocument

logger = get_logger(__name__)

# A paragraph is a run of non-blank lines.
_PARAGRAPH = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")

# Documents with no detectable structure at all still need one section to live in.
_DEFAULT_SECTION_TITLE = "Full Text"


class _Section:
    __slots__ = ("title", "chapter_title", "chapter_n", "section_n", "start", "end")

    def __init__(
        self,
        title: str,
        chapter_title: str,
        chapter_n: int,
        section_n: int,
        start: int,
        end: int,
    ) -> None:
        self.title = title
        self.chapter_title = chapter_title
        self.chapter_n = chapter_n
        self.section_n = section_n
        self.start = start
        self.end = end


def _build_sections(doc: ParsedDocument) -> list[_Section]:
    """Turn headings into non-overlapping, numbered sections covering the text."""
    text_len = len(doc.raw_text)
    headings = [h for h in doc.headings if 0 <= h.char_offset < text_len]
    headings.sort(key=lambda h: h.char_offset)

    if not headings:
        return [_Section(_DEFAULT_SECTION_TITLE, _DEFAULT_SECTION_TITLE, 1, 1, 0, text_len)]

    sections: list[_Section] = []
    chapter_n = 0
    section_n = 0
    chapter_title = ""

    # Text before the first heading is real content (preface, epigraph) — keep it.
    if headings[0].char_offset > 0:
        sections.append(_Section("Front Matter", "Front Matter", 1, 1, 0, headings[0].char_offset))
        chapter_n = 1
        chapter_title = "Front Matter"

    for i, heading in enumerate(headings):
        end = headings[i + 1].char_offset if i + 1 < len(headings) else text_len
        if end <= heading.char_offset:
            continue

        if heading.level == 1:
            chapter_n += 1
            section_n = 1
            chapter_title = heading.title
        else:
            if chapter_n == 0:  # a sub-heading before any chapter heading
                chapter_n = 1
                chapter_title = heading.title
            section_n += 1

        sections.append(
            _Section(
                title=heading.title,
                chapter_title=chapter_title or heading.title,
                chapter_n=chapter_n,
                section_n=section_n,
                start=heading.char_offset,
                end=end,
            )
        )

    return sections


def _paragraph_spans(text: str, base: int) -> list[tuple[str, int, int]]:
    """Paragraphs of `text` as (content, absolute_start, absolute_end)."""
    spans: list[tuple[str, int, int]] = []
    for match in _PARAGRAPH.finditer(text):
        content = match.group().strip()
        if content:
            spans.append((content, base + match.start(), base + match.end()))
    return spans


def _pack(
    paragraphs: list[tuple[str, int, int]],
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    """Group paragraphs into chunks of ≤ max_chars with trailing-paragraph overlap."""
    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0

    def flush() -> None:
        if current:
            chunks.append(("\n\n".join(p[0] for p in current), current[0][1], current[-1][2]))

    for para in paragraphs:
        para_len = len(para[0])

        # A single paragraph longer than the budget becomes its own chunk rather
        # than being split mid-thought.
        if para_len > max_chars:
            flush()
            chunks.append(para)
            current, current_len = [], 0
            continue

        if current and current_len + para_len > max_chars:
            flush()
            tail: list[tuple[str, int, int]] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev[0]) > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_len += len(prev[0])
            current, current_len = tail, tail_len

        current.append(para)
        current_len += para_len + 2

    flush()
    return chunks


def build_chunks(doc: ParsedDocument, book_id: int) -> list[Chunk]:
    """Convert a parsed document into ordered chunks. Embeddings come later."""
    sections = _build_sections(doc)
    chunks: list[Chunk] = []
    sequence = 0

    for section in sections:
        body = doc.raw_text[section.start : section.end]
        paragraphs = _paragraph_spans(body, section.start)
        if not paragraphs:
            continue

        packed = _pack(paragraphs, settings.CHUNK_SIZE_CHARS, settings.CHUNK_OVERLAP_CHARS)
        for paragraph_n, (text, start, end) in enumerate(packed, start=1):
            chunks.append(
                Chunk(
                    id=0,  # assigned by the database on insert
                    book_id=book_id,
                    text=text,
                    sequence=sequence,
                    location=ChunkLocation(
                        chapter_title=section.chapter_title,
                        chapter_n=section.chapter_n,
                        section_title=section.title,
                        section_n=section.section_n,
                        char_start=start,
                        char_end=end,
                        page_start=doc.page_for_offset(start),
                        page_end=doc.page_for_offset(end),
                        paragraph_n=paragraph_n,
                    ),
                    token_count=len(text) // 4,  # rough, only used for reporting
                )
            )
            sequence += 1

    logger.info(
        "Chunked book_id=%d: %d sections → %d chunks", book_id, len(sections), len(chunks)
    )
    return chunks


def heading_outline(doc: ParsedDocument) -> list[Heading]:
    """The document's headings, in reading order — used to build the stored ToC."""
    return sorted(doc.headings, key=lambda h: h.char_offset)
