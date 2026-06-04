from pydantic import BaseModel


class ChunkLocationResponse(BaseModel):
    chapter_title: str | None = None
    chapter_n: int | None = None
    section_title: str | None = None
    section_n: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    paragraph_n: int | None = None


class CitationResponse(BaseModel):
    text: str
    book_title: str
    author: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_range: str | None = None


class SearchResultResponse(BaseModel):
    chunk_id: int
    book_id: int
    text: str
    score: float
    location: ChunkLocationResponse
    citation: CitationResponse
