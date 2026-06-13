from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class BookFormat(StrEnum):
    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"


class BookStatus(StrEnum):
    PENDING = "pending"       # uploaded, waiting for worker
    PROCESSING = "processing" # worker has picked it up
    READY = "ready"           # fully ingested and searchable
    FAILED = "failed"         # ingestion failed


@dataclass
class BookMetadata:
    title: str
    author: str | None = None
    year: int | None = None
    publisher: str | None = None
    language: str = "en"
    isbn: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TocEntry:
    title: str
    level: int           # 1 = chapter, 2 = section, 3 = subsection
    sequence: int        # ordinal position in the book
    start_char: int | None = None
    page_start: int | None = None
    chunk_id: int | None = None  # points to first chunk of this section


@dataclass
class Book:
    id: int
    user_id: str
    status: BookStatus
    format: BookFormat
    file_size_bytes: int
    metadata: BookMetadata
    toc: list[TocEntry] = field(default_factory=list)
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BookSummary:
    id: int
    title: str
    author: str | None
    status: BookStatus
    format: BookFormat
    chunk_count: int
    created_at: datetime
