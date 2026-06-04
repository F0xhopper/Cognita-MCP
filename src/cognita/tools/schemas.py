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
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


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
