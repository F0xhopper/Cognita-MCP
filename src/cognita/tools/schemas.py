"""Pydantic schemas for MCP tool inputs and outputs.

These are the contracts exposed to AI agents — keep them stable and well-documented.
"""

from pydantic import BaseModel, Field


# ── Tool Inputs ──────────────────────────────────────────────────────────────

class ListBooksInput(BaseModel):
    pass  # no parameters — returns the caller's full library


class SearchLibraryInput(BaseModel):
    query: str = Field(..., description="Title, author, or keyword to search for across books")


class GetTableOfContentsInput(BaseModel):
    book_id: int = Field(..., description="The book ID returned by list_books or search_library")


class GetChapterInput(BaseModel):
    book_id: int
    chapter_n: int = Field(..., description="Chapter number (1-based)", ge=1)


class GetSectionInput(BaseModel):
    book_id: int
    chapter_n: int = Field(..., ge=1)
    section_n: int = Field(..., ge=1)


class SemanticSearchInput(BaseModel):
    query: str = Field(..., description="Natural language query to search across the library")
    book_ids: list[int] | None = Field(
        None,
        description="Restrict search to specific book IDs. Omit to search entire library.",
    )
    specialty_id: int | None = Field(
        None,
        description="Restrict search to a specialty's books. Use list_specialties to discover IDs.",
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


class CreateSpecialtyInput(BaseModel):
    name: str = Field(
        ..., min_length=1, max_length=200, description="Display name, e.g. 'Stoic Philosophy'"
    )
    description: str | None = Field(None, description="What this specialty covers")
    persona: str | None = Field(
        None,
        description="Instruction block shaping how the expert answers, e.g. 'You are an expert…'",
    )


class ConfirmCorpusInput(BaseModel):
    specialty_id: int = Field(..., description="Specialty ID returned by create_specialty")
    approved_indices: list[int] = Field(
        ...,
        description="Indices from the suggestions list to approve and ingest",
    )


class AddBooksToSpecialtyInput(BaseModel):
    specialty_id: int = Field(..., description="Specialty ID from list_specialties")
    book_ids: list[int] = Field(..., min_length=1, description="Book IDs to add")


class DeepResearchInput(BaseModel):
    question: str = Field(..., description="The research question to investigate")
    specialty_id: int | None = Field(
        None,
        description="Scope research to a specialty (its books and persona); omit for whole library",
    )
    book_ids: list[int] | None = Field(None, description="Restrict to specific book IDs")
    depth: int = Field(
        2, ge=1, le=3,
        description="1 = single retrieval pass, 2-3 = wider plan plus gap-check follow-ups",
    )


class GetPassageContextInput(BaseModel):
    chunk_id: int = Field(..., description="Chunk ID from a SemanticSearch result")
    book_id: int
    window: int = Field(2, ge=1, le=5, description="Number of chunks to expand on each side")


class GetPassageByLocationInput(BaseModel):
    book_id: int
    chapter_n: int | None = Field(None, ge=1)
    section_n: int | None = Field(None, ge=1)


# ── Tool Outputs ─────────────────────────────────────────────────────────────

class BookItem(BaseModel):
    id: int
    title: str
    author: str | None
    format: str
    chunk_count: int
    status: str


class TocItem(BaseModel):
    title: str
    level: int
    sequence: int
    page_start: int | None


class PassageResult(BaseModel):
    chunk_id: int
    book_id: int
    text: str
    score: float
    citation: str          # human-readable: "Author, Title › Chapter › Section › p. 42"
    chapter_title: str | None
    section_title: str | None
    page_start: int | None


class ExpandedPassage(BaseModel):
    full_text: str
    citation: str
    chunk_ids: list[int]   # ordered IDs of all chunks included in full_text


class CorpusSuggestionItem(BaseModel):
    index: int              # pass this in approved_indices to confirm_corpus
    title: str
    author: str
    tier: str               # primary | commentary | competing | synthesis
    rationale: str
    source_url: str | None
    source_type: str        # gutenberg | archive_org | user_upload_required
    approved: bool          # pre-selected default; user should review


class SpecialtyItem(BaseModel):
    id: int
    name: str
    description: str | None
    persona: str | None
    book_ids: list[int]
    book_count: int


class SpecialtyWithSuggestionsItem(SpecialtyItem):
    """Returned by create_specialty — includes suggested corpus for user review.

    Present the suggestions to the user; they approve or reject each one.
    Then call confirm_corpus with the approved indices to start ingestion.
    Items with source_type='user_upload_required' have no URL and must be
    uploaded manually via add_book_from_url after the user locates a copy.
    """
    suggestions: list[CorpusSuggestionItem]


class ResearchReportResult(BaseModel):
    question: str
    answer: str                # cited prose; markers like [1] refer to entries in citations
    citations: list[str]       # citations[n-1] corresponds to inline marker [n]
    sub_queries: list[str]     # the search queries the agent ran
    specialty_name: str | None
    passages: list[PassageResult]  # the numbered passages backing the citations, in order
