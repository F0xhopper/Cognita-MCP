from dataclasses import dataclass, field

from cognita.chunks.domain import Chunk, ChunkLocation, Citation


@dataclass
class Passage:
    """One search result.

    A passage is usually a single chunk, but adjacent chunks that both matched
    are merged into one — so ``chunk_ids`` may hold several ids and ``text`` the
    joined, de-overlapped run of text.
    """

    chunk_id: int
    book_id: int
    text: str
    score: float
    citation: Citation
    location: ChunkLocation
    chunk_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chunk_ids:
            self.chunk_ids = [self.chunk_id]

    @property
    def merged(self) -> bool:
        return len(self.chunk_ids) > 1


@dataclass
class SearchResponse:
    query: str
    passages: list[Passage]
    # How the ordering was produced, so a caller can tell a reranked list from
    # a raw fusion ordering.
    ranking: str = "fusion"

    @property
    def total(self) -> int:
        return len(self.passages)


@dataclass
class PassageContext:
    """A passage widened with the chunks either side of it."""

    passage: Passage
    before: list[Chunk] = field(default_factory=list)
    after: list[Chunk] = field(default_factory=list)

    def full_text(self) -> str:
        parts = [c.text for c in self.before] + [self.passage.text] + [c.text for c in self.after]
        return join_without_overlap(parts)

    def chunk_ids(self) -> list[int]:
        return (
            [c.id for c in self.before] + self.passage.chunk_ids + [c.id for c in self.after]
        )


def join_without_overlap(parts: list[str]) -> str:
    """Join consecutive chunks, dropping the paragraphs they share.

    Chunks deliberately overlap by whole trailing paragraphs, so a naive join
    repeats them. Comparing paragraph by paragraph removes the repeat exactly.
    """
    if not parts:
        return ""

    merged: list[str] = _paragraphs(parts[0])
    for part in parts[1:]:
        incoming = _paragraphs(part)
        overlap = 0
        # Longest suffix of what we have that is a prefix of what is arriving.
        for size in range(min(len(merged), len(incoming)), 0, -1):
            if merged[-size:] == incoming[:size]:
                overlap = size
                break
        merged.extend(incoming[overlap:])
    return "\n\n".join(merged)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]
