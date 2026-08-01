"""Structured results returned by the MCP tools.

FastMCP derives each tool's *input* schema from its function signature; these
models describe the *outputs*. They are the agent-facing contract, so the field
names and descriptions here are read by the model at call time — keep them plain.
"""

from pydantic import BaseModel, Field


class BookItem(BaseModel):
    """A book in the library."""

    id: int
    title: str
    author: str | None = None
    format: str
    status: str = Field(description="pending | processing | ready | failed")
    chunk_count: int = Field(description="Searchable passages; 0 until ingestion finishes.")
    error: str | None = Field(default=None, description="Why ingestion failed, if it did.")


class TocItem(BaseModel):
    """One entry in a book's table of contents."""

    title: str
    level: int = Field(description="1 = chapter, 2 = section, 3 = subsection")
    chapter_n: int | None = Field(
        default=None, description="Pass to read_chapter to read this part in full."
    )
    section_n: int | None = Field(default=None, description="Pass to read_section.")
    page_start: int | None = None


class PassageResult(BaseModel):
    """A passage of a book, with everything needed to quote and cite it."""

    text: str
    citation: str = Field(description='Ready to quote: "Author, Title › Chapter › p. 42".')
    score: float = Field(description="Relevance, 0–1, relative to the best hit for this query.")
    book_id: int
    book_title: str
    chunk_id: int = Field(description="Pass to expand_passage to read around this passage.")
    chapter_title: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    chapter_n: int | None = None
    section_n: int | None = None


class SearchResult(BaseModel):
    """The response to a library search."""

    query: str
    ranking: str = Field(description="'reranked' if a model ordered these, else 'fusion'.")
    passages: list[PassageResult]


class ExpandedPassage(BaseModel):
    """A passage widened with the text either side of it."""

    text: str
    citation: str
    chunk_ids: list[int]


class AddedBook(BaseModel):
    """A book accepted into the library and queued for ingestion."""

    id: int
    title: str
    author: str | None = None
    format: str
    status: str
    source: str | None = Field(default=None, description="Where it came from.")
    note: str = Field(
        default=(
            "Queued for ingestion — searchable in a minute or two. "
            "Check library_status or list_books for progress."
        )
    )


class FolderImportResult(BaseModel):
    """Outcome of importing a directory of books."""

    added_count: int
    skipped_count: int
    added: list[AddedBook]
    skipped: list[str] = Field(description="Files not imported, each with the reason.")


class LibraryStatusResult(BaseModel):
    """What is in the library and what is still being processed."""

    total_books: int
    ready: int
    processing: int
    pending: int
    failed: int
    total_passages: int
    queue_depth: int = Field(description="Books waiting on or currently in ingestion.")
    failures: list[BookItem] = Field(
        default_factory=list, description="Recent failures, with their error messages."
    )
    search_quality: str = Field(
        description="Which optional quality features are active on this server."
    )
