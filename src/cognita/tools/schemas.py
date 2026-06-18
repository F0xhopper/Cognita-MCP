"""Pydantic result models returned by the MCP tools.

Input contracts are derived by FastMCP from each tool's function signature in
server.py; these models describe the structured outputs the tools return.
Keep them stable and well-documented — they are part of the agent-facing contract.
"""

from pydantic import BaseModel


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
