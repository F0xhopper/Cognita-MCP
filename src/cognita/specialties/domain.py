from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class SourceTier(StrEnum):
    PRIMARY = "primary"
    COMMENTARY = "commentary"
    COMPETING = "competing"
    SYNTHESIS = "synthesis"


class SourceType(StrEnum):
    GUTENBERG = "gutenberg"
    STANDARD_EBOOKS = "standard_ebooks"
    OPEN_LIBRARY = "open_library"
    ARCHIVE_ORG = "archive_org"
    WIKISOURCE = "wikisource"
    USER_UPLOAD_REQUIRED = "user_upload_required"


@dataclass
class SuggestedSource:
    title: str
    author: str
    tier: SourceTier
    rationale: str
    source_url: str | None
    source_type: SourceType
    approved: bool


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
    pending_corpus: list[SuggestedSource] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def book_count(self) -> int:
        return len(self.book_ids)
