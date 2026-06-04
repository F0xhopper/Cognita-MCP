import json

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from cognita.auth.dependencies import get_current_user_id
from cognita.books.domain import BookMetadata
from cognita.books.schemas import BookResponse, BookSummaryResponse, TocEntryResponse
from cognita.books.service import BookService
from cognita.core.exceptions import NotFoundError, UnsupportedFormatError
from cognita.infrastructure.database import get_pool
from cognita.ingestion.worker import ingest_book_task

router = APIRouter(prefix="/books", tags=["books"])


def _get_service(pool=Depends(get_pool)) -> BookService:
    return BookService(pool)


@router.get("/", response_model=list[BookSummaryResponse])
async def list_books(
    user_id: str = Depends(get_current_user_id),
    svc: BookService = Depends(_get_service),
):
    books = await svc.list_books(user_id)
    return [
        BookSummaryResponse(
            id=b.id, title=b.title, author=b.author, status=b.status,
            format=b.format, chunk_count=b.chunk_count, created_at=b.created_at,
        )
        for b in books
    ]


@router.get("/search", response_model=list[BookSummaryResponse])
async def search_library(
    q: str,
    user_id: str = Depends(get_current_user_id),
    svc: BookService = Depends(_get_service),
):
    books = await svc.search_library(user_id, q)
    return [
        BookSummaryResponse(
            id=b.id, title=b.title, author=b.author, status=b.status,
            format=b.format, chunk_count=b.chunk_count, created_at=b.created_at,
        )
        for b in books
    ]


@router.post("/upload", response_model=BookSummaryResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_book(
    file: UploadFile,
    metadata: str = Form(..., description="JSON-encoded BookMetadataInput"),
    user_id: str = Depends(get_current_user_id),
    svc: BookService = Depends(_get_service),
):
    try:
        meta_dict = json.loads(metadata)
        meta = BookMetadata(**meta_dict)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    try:
        book = await svc.upload(user_id=user_id, file=file, meta=meta)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    ingest_book_task.delay(book.id)
    return BookSummaryResponse(
        id=book.id, title=book.metadata.title, author=book.metadata.author,
        status=book.status, format=book.format, chunk_count=0, created_at=book.created_at,
    )


@router.get("/{book_id}", response_model=BookResponse)
async def get_book(
    book_id: int,
    user_id: str = Depends(get_current_user_id),
    svc: BookService = Depends(_get_service),
):
    try:
        book = await svc.get_book(user_id, book_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return BookResponse(
        id=book.id,
        status=book.status,
        format=book.format,
        title=book.metadata.title,
        author=book.metadata.author,
        year=book.metadata.year,
        language=book.metadata.language,
        chunk_count=book.chunk_count,
        toc=[
            TocEntryResponse(
                title=e.title, level=e.level,
                sequence=e.sequence, page_start=e.page_start,
            )
            for e in book.toc
        ],
        error_message=book.error_message,
        created_at=book.created_at,
    )


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(
    book_id: int,
    user_id: str = Depends(get_current_user_id),
    svc: BookService = Depends(_get_service),
):
    try:
        await svc.delete_book(user_id, book_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
