from datetime import datetime

from pydantic import AnyHttpUrl, BaseModel, Field

from cognita.books.domain import BookFormat, BookStatus


class BookMetadataInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    author: str | None = None
    year: int | None = Field(None, ge=1000, le=2100)
    publisher: str | None = None
    language: str = "en"
    isbn: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class AddBookFromUrlRequest(BaseModel):
    url: AnyHttpUrl = Field(..., description="Public URL to a PDF, EPUB, or plain-text file")
    metadata: BookMetadataInput


class TocEntryResponse(BaseModel):
    title: str
    level: int
    sequence: int
    page_start: int | None = None


class BookResponse(BaseModel):
    id: int
    status: BookStatus
    format: BookFormat
    title: str
    author: str | None
    year: int | None
    language: str
    chunk_count: int
    toc: list[TocEntryResponse]
    error_message: str | None
    created_at: datetime


class BookSummaryResponse(BaseModel):
    id: int
    title: str
    author: str | None
    status: BookStatus
    format: BookFormat
    chunk_count: int
    created_at: datetime
