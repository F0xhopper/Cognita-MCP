from dataclasses import dataclass, field

from cognita.search.domain import SearchResult


@dataclass
class ResearchFinding:
    """A passage retrieved during research, numbered for inline citation."""

    n: int                     # citation marker used in the answer, e.g. [3]
    result: SearchResult

    @property
    def citation(self) -> str:
        return self.result.citation.to_string()


@dataclass
class CoverageGap:
    """A topic the specialty's corpus covers poorly or not at all."""

    topic: str
    reason: str
    suggested_reading: list[str] = field(default_factory=list)


@dataclass
class GapAnalysis:
    """Coverage audit of a specialty's corpus against its subject."""

    specialty_id: int
    specialty_name: str
    summary: str
    gaps: list[CoverageGap] = field(default_factory=list)
    books_analyzed: int = 0


@dataclass
class ResearchReport:
    """The synthesized output of a deep_research run."""

    question: str
    answer: str                          # cited prose, markers like [1] refer to citations
    citations: list[str]                 # citations[n-1] corresponds to marker [n]
    sub_queries: list[str] = field(default_factory=list)
    specialty_name: str | None = None
    findings: list[ResearchFinding] = field(default_factory=list)
