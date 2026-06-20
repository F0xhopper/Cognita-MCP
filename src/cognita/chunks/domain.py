from dataclasses import dataclass, field
from enum import StrEnum


class ChunkLevel(StrEnum):
    CHAPTER = "chapter"
    SECTION = "section"
    PARAGRAPH = "paragraph"


@dataclass
class ChunkLocation:
    """Precise location of a chunk within its source book."""
    chapter_title: str | None = None
    chapter_n: int | None = None
    section_title: str | None = None
    section_n: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    paragraph_n: int | None = None   # paragraph ordinal within section


@dataclass
class Chunk:
    """A text unit stored in the vector DB with its embedding."""
    id: int
    book_id: int
    user_id: str
    text: str
    level: ChunkLevel
    sequence: int                    # global ordinal across book (for neighbour expansion)
    location: ChunkLocation
    embedding: list[float] = field(default_factory=list, repr=False)
    token_count: int = 0
    context: str = ""                # contextual-retrieval blurb; embedded + indexed alongside text, never quoted


@dataclass
class Citation:
    """Human-readable citation for a chunk."""
    book_title: str
    author: str | None
    chapter: str | None
    section: str | None
    page_range: str | None          # e.g. "pp. 42-43"
    paragraph_n: int | None

    def to_string(self) -> str:
        parts = [self.book_title]
        if self.author:
            parts[0] = f"{self.author}, {self.book_title}"
        if self.chapter:
            parts.append(self.chapter)
        if self.section:
            parts.append(self.section)
        if self.page_range:
            parts.append(self.page_range)
        return " › ".join(parts)
