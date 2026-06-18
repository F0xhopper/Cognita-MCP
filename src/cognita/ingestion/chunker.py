"""Hierarchical chunker: chapters → sections → paragraph-sized chunks with overlap.

Strategy:
  1. Use native ToC (from PDF outline / EPUB headings) to identify chapters/sections.
  2. If no native ToC, fall back to heuristic heading detection (ALL CAPS lines,
     numbered headings like "Chapter 1", markdown-style "# Heading").
  3. Within each section, split into paragraph-sized chunks respecting sentence
     boundaries. Overlap is applied by repeating the last N chars of the prev chunk.
"""

import re
from dataclasses import dataclass

from cognita.chunks.domain import Chunk, ChunkLevel, ChunkLocation
from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.ingestion.parsers import ParsedDocument

logger = get_logger(__name__)

_CHAPTER_PATTERNS = [
    re.compile(r"^(chapter|part|book)\s+\w+[\s:—–-]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s{0,4}[IVXLC]+\.\s+\w", re.MULTILINE),       # Roman numerals
    re.compile(r"^\s{0,4}\d{1,2}\.\s+[A-Z][A-Za-z ]{3,}", re.MULTILINE),
]

_SECTION_PATTERNS = [
    re.compile(r"^\s{0,4}#{1,3} .+", re.MULTILINE),              # Markdown headings
    re.compile(r"^[A-Z][A-Z\s]{5,40}$", re.MULTILINE),           # ALL CAPS lines
]


@dataclass
class _Section:
    title: str
    level: int
    text: str
    start_char: int
    chapter_n: int
    section_n: int
    chapter_title: str


def _detect_sections(text: str, native_toc: list[dict]) -> list[_Section]:
    if native_toc:
        return _sections_from_toc(text, native_toc)
    return _sections_from_heuristics(text)


def _sections_from_toc(text: str, toc: list[dict]) -> list[_Section]:
    sections: list[_Section] = []
    chapter_n = 0
    section_n = 0
    current_chapter = "Preface"

    for i, entry in enumerate(toc):
        title = entry["title"]
        level = entry.get("level", 1)
        char_offset = entry.get("char_offset")

        if level == 1:
            chapter_n += 1
            section_n = 0
            current_chapter = title

        next_offset = None
        if i + 1 < len(toc) and toc[i + 1].get("char_offset") is not None:
            next_offset = toc[i + 1]["char_offset"]

        if char_offset is not None:
            chunk_text = text[char_offset:next_offset] if next_offset else text[char_offset:]
            section_n += 1
            sections.append(_Section(
                title=title,
                level=level,
                text=chunk_text.strip(),
                start_char=char_offset,
                chapter_n=chapter_n,
                section_n=section_n,
                chapter_title=current_chapter,
            ))

    if not sections:
        sections.append(_Section(
            title="Full Text",
            level=1,
            text=text.strip(),
            start_char=0,
            chapter_n=1,
            section_n=1,
            chapter_title="Full Text",
        ))
    return sections


def _sections_from_heuristics(text: str) -> list[_Section]:
    boundaries: list[tuple[int, str, int]] = []  # (char_offset, heading_text, level)

    for pattern in _CHAPTER_PATTERNS:
        for m in pattern.finditer(text):
            heading = text[m.start():text.find("\n", m.start())].strip()
            boundaries.append((m.start(), heading, 1))
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            heading = text[m.start():text.find("\n", m.start())].strip()
            boundaries.append((m.start(), heading, 2))

    boundaries.sort(key=lambda x: x[0])
    if not boundaries:
        return [_Section("Full Text", 1, text.strip(), 0, 1, 1, "Full Text")]

    sections: list[_Section] = []
    chapter_n = 0
    section_n = 0
    current_chapter = "Introduction"

    for i, (start, title, level) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        if level == 1:
            chapter_n += 1
            section_n = 0
            current_chapter = title
        section_n += 1
        sections.append(_Section(
            title=title,
            level=level,
            text=text[start:end].strip(),
            start_char=start,
            chapter_n=chapter_n,
            section_n=section_n,
            chapter_title=current_chapter,
        ))

    return sections


def _split_into_paragraphs(
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Split `text` into chunks ≤ max_chars, respecting paragraph and sentence
    boundaries. Overlap carries the last `overlap` characters from the prev chunk."""
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in raw_paras:
        if current_len + len(para) + 1 > max_chars and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            # overlap: keep last paragraph(s) whose total is ≤ overlap chars
            tail: list[str] = []
            tail_len = 0
            for p in reversed(current):
                if tail_len + len(p) <= overlap:
                    tail.insert(0, p)
                    tail_len += len(p)
                else:
                    break
            current = tail
            current_len = tail_len

        current.append(para)
        current_len += len(para) + 1

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


def build_chunks(
    doc: ParsedDocument,
    book_id: int,
    user_id: str,
) -> list[Chunk]:
    """Convert a parsed document into a flat list of Chunk objects (no embeddings yet)."""
    sections = _detect_sections(doc.raw_text, doc.native_toc)
    logger.info("Detected %d sections for book_id=%d", len(sections), book_id)

    chunks: list[Chunk] = []
    global_seq = 0

    for sec in sections:
        if not sec.text:
            continue

        paras = _split_into_paragraphs(
            sec.text,
            settings.CHUNK_SIZE_CHARS,
            settings.CHUNK_OVERLAP_CHARS,
        )

        for para_n, para_text in enumerate(paras, start=1):
            loc = ChunkLocation(
                chapter_title=sec.chapter_title,
                chapter_n=sec.chapter_n,
                section_title=sec.title,
                section_n=sec.section_n,
                char_start=sec.start_char,
                paragraph_n=para_n,
            )
            chunks.append(Chunk(
                id=0,                   # assigned by DB on insert
                book_id=book_id,
                user_id=user_id,
                text=para_text,
                level=ChunkLevel.PARAGRAPH,
                sequence=global_seq,
                location=loc,
                token_count=len(para_text) // 4,  # rough token estimate
            ))
            global_seq += 1

    logger.info("Built %d chunks for book_id=%d", len(chunks), book_id)
    return chunks
