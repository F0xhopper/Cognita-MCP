from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


# ── Corpus suggestion (inline on create) ──────────────────────────────────────

class SuggestedSourceResponse(BaseModel):
    title: str
    author: str
    tier: Literal["primary", "commentary", "competing", "synthesis"]
    rationale: str
    source_url: str | None
    source_type: Literal["gutenberg", "standard_ebooks", "open_library", "archive_org", "wikisource", "user_upload_required"]
    approved: bool

    @computed_field
    @property
    def requires_upload(self) -> bool:
        return self.source_type == "user_upload_required"


class CorpusConfirmInput(BaseModel):
    approved_indices: list[int] = Field(..., min_length=1)


# ── Corpus status ──────────────────────────────────────────────────────────────

class CorpusBookStatus(BaseModel):
    book_id: int
    title: str
    author: str | None
    status: str


class CorpusStatusResponse(BaseModel):
    specialty_id: int
    total: int
    ready: int
    processing: int
    failed: int
    pending: int
    books: list[CorpusBookStatus]


# ── Specialty CRUD ─────────────────────────────────────────────────────────────

class SpecialtyCreateInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    persona: str | None = Field(
        None,
        max_length=4000,
        description="Instruction block shaping how the expert presents itself",
    )


class SpecialtyUpdateInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    persona: str | None = Field(None, max_length=4000)


class SpecialtyBooksInput(BaseModel):
    book_ids: list[int] = Field(..., min_length=1)


class SpecialtyResponse(BaseModel):
    id: int
    name: str
    description: str | None
    persona: str | None
    book_ids: list[int]
    book_count: int
    created_at: datetime
    updated_at: datetime


class SpecialtyCreateResponse(SpecialtyResponse):
    """Returned only on POST /specialties — includes the suggested corpus for user review."""
    suggestions: list[SuggestedSourceResponse]
