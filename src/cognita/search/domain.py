from dataclasses import dataclass, field

from cognita.chunks.domain import Citation, Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    citation: Citation

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    total: int


@dataclass
class PassageContext:
    """A search hit expanded with surrounding chunks for richer context."""
    hit: SearchResult
    before: list[Chunk] = field(default_factory=list)
    after: list[Chunk] = field(default_factory=list)

    def full_text(self) -> str:
        parts = [c.text for c in self.before] + [self.hit.text] + [c.text for c in self.after]
        return "\n\n".join(parts)
