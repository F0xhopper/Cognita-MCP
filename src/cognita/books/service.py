from fastapi import UploadFile

import asyncpg

from cognita.books.domain import Book, BookFormat, BookMetadata, BookStatus, BookSummary
from cognita.books.repository import BookRepository
from cognita.books.url_fetcher import fetch_book_from_url
from cognita.chunks.repository import ChunkRepository
from cognita.core.exceptions import NotFoundError, UnsupportedFormatError

_ALLOWED_FORMATS = {
    "application/pdf": BookFormat.PDF,
    "application/epub+zip": BookFormat.EPUB,
    "text/plain": BookFormat.TXT,
}
_ALLOWED_EXTENSIONS = {".pdf": BookFormat.PDF, ".epub": BookFormat.EPUB, ".txt": BookFormat.TXT}
_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


class BookService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._repo = BookRepository(pool)
        self._chunk_repo = ChunkRepository(pool)

    async def upload(
        self,
        user_id: str,
        file: UploadFile,
        meta: BookMetadata,
    ) -> Book:
        fmt = _detect_format(file)
        data = await file.read()
        if len(data) > _MAX_FILE_SIZE:
            raise ValueError(f"File exceeds {_MAX_FILE_SIZE // (1024*1024)} MB limit")

        return await self._repo.create(
            user_id=user_id,
            status=BookStatus.PENDING,
            fmt=fmt,
            file_data=data,
            file_size_bytes=len(data),
            meta=meta,
        )

    async def add_from_url(self, user_id: str, url: str, meta: BookMetadata) -> Book:
        data, fmt, _filename = await fetch_book_from_url(url)
        return await self._repo.create(
            user_id=user_id,
            status=BookStatus.PENDING,
            fmt=fmt,
            file_data=data,
            file_size_bytes=len(data),
            meta=meta,
        )

    async def list_books(self, user_id: str) -> list[BookSummary]:
        return await self._repo.list_for_user(user_id)

    async def search_library(self, user_id: str, query: str) -> list[BookSummary]:
        return await self._repo.search_library(user_id, query)

    async def get_book(self, user_id: str, book_id: int) -> Book:
        book = await self._repo.get(book_id, user_id)
        if book is None:
            raise NotFoundError("Book", book_id)
        return book

    async def delete_book(self, user_id: str, book_id: int) -> None:
        book = await self._repo.get(book_id, user_id)
        if book is None:
            raise NotFoundError("Book", book_id)
        await self._chunk_repo.delete_for_book(book_id)
        await self._repo.delete(book_id, user_id)


def _detect_format(file: UploadFile) -> BookFormat:
    import os
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in _ALLOWED_EXTENSIONS:
            return _ALLOWED_EXTENSIONS[ext]
    if file.content_type and file.content_type in _ALLOWED_FORMATS:
        return _ALLOWED_FORMATS[file.content_type]
    raise UnsupportedFormatError(file.content_type or "unknown")
