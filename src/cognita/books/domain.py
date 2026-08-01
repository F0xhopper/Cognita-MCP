from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class BookFormat(StrEnum):
    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"
    MD = "md"
    HTML = "html"


class BookStatus(StrEnum):
    PENDING = "pending"        # stored, waiting for the ingestion queue
    PROCESSING = "processing"  # being parsed / embedded right now
    READY = "ready"            # fully ingested and searchable
    FAILED = "failed"          # ingestion failed; see error_message


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
    source: str | None = None  # where it came from: a path, URL, or "text"


@dataclass
class TocEntry:
    title: str
    level: int  # 1 = chapter, 2 = section, 3 = subsection
    sequence: int
    start_char: int | None = None
    page_start: int | None = None
    chunk_id: int | None = None  # first chunk of this section


@dataclass
class Book:
    id: int
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
    error_message: str | None = None


@dataclass
class LibraryStatus:
    """Aggregate view of the library — what is searchable and what is still cooking."""

    total: int
    ready: int
    processing: int
    pending: int
    failed: int
    chunk_count: int
    queue_depth: int
    failures: list[BookSummary] = field(default_factory=list)
