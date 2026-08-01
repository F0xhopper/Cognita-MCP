from dataclasses import dataclass, field


@dataclass
class ChunkLocation:
    """Where a chunk sits inside its source book."""

    chapter_title: str | None = None
    chapter_n: int | None = None
    section_title: str | None = None
    section_n: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    paragraph_n: int | None = None  # ordinal within the section


@dataclass
class Chunk:
    """A passage of a book, stored with its embedding."""

    id: int
    book_id: int
    text: str
    sequence: int  # global ordinal across the book, used for neighbour expansion
    location: ChunkLocation
    embedding: list[float] = field(default_factory=list, repr=False)
    token_count: int = 0
    # Contextual-retrieval blurb: embedded and full-text indexed alongside the
    # passage so retrieval sees it in context. Never quoted back to the caller.
    context: str = ""


@dataclass
class Citation:
    """Human-readable provenance for a passage."""

    book_title: str
    author: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_range: str | None = None
    paragraph_n: int | None = None

    def to_string(self) -> str:
        head = f"{self.author}, {self.book_title}" if self.author else self.book_title
        parts = [head]

        # Short documents often have one heading that repeats the title, which
        # would render as "Reading notes › Reading notes". Each level is only
        # worth naming when it says something the previous one did not.
        seen = {self.book_title}
        for level in (self.chapter, self.section):
            if level and level not in seen:
                parts.append(level)
                seen.add(level)

        if self.page_range:
            parts.append(self.page_range)
        return " › ".join(parts)
