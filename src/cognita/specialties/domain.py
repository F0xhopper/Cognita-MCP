from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Specialty:
    """A named, persona-bearing slice of the user's library.

    A specialty scopes retrieval and research to a curated set of books and
    carries an optional persona/instruction block that shapes how the research
    agent presents itself (e.g. "You are an expert on Stoic philosophy…").
    """

    id: int
    user_id: str
    name: str
    description: str | None = None
    persona: str | None = None
    book_ids: list[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def book_count(self) -> int:
        return len(self.book_ids)
