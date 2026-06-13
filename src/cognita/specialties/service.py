import asyncio

import asyncpg

from cognita.books.domain import BookMetadata
from cognita.books.repository import BookRepository
from cognita.books.service import BookService
from cognita.core.exceptions import ConflictError, NotFoundError
from cognita.ingestion.pipeline import ingest_book
from cognita.specialties.domain import Specialty, SuggestedSource
from cognita.specialties.repository import SpecialtyRepository
from cognita.specialties.schemas import CorpusBookStatus, CorpusStatusResponse
from cognita.specialties.source_resolver import resolve_sources
from cognita.specialties.suggester import suggest_corpus


class SpecialtyService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._repo = SpecialtyRepository(pool)
        self._book_repo = BookRepository(pool)

    async def create(
        self,
        user_id: str,
        name: str,
        description: str | None = None,
        persona: str | None = None,
    ) -> Specialty:
        existing = await self._repo.get_by_name(user_id, name)
        if existing is not None:
            raise ConflictError(f"Specialty already exists: {name}")

        specialty = await self._repo.create(user_id, name, description, persona)

        raw = await suggest_corpus(name, description)
        suggestions = await resolve_sources(raw)
        await self._repo.save_pending_corpus(specialty.id, suggestions)
        specialty.pending_corpus = suggestions
        return specialty

    async def get(self, user_id: str, specialty_id: int) -> Specialty:
        specialty = await self._repo.get(specialty_id, user_id)
        if specialty is None:
            raise NotFoundError("Specialty", specialty_id)
        return specialty

    async def list_specialties(self, user_id: str) -> list[Specialty]:
        return await self._repo.list_for_user(user_id)

    async def update(
        self,
        user_id: str,
        specialty_id: int,
        name: str,
        description: str | None,
        persona: str | None,
    ) -> Specialty:
        existing = await self._repo.get_by_name(user_id, name)
        if existing is not None and existing.id != specialty_id:
            raise ConflictError(f"Specialty already exists: {name}")
        updated = await self._repo.update(specialty_id, user_id, name, description, persona)
        if updated is None:
            raise NotFoundError("Specialty", specialty_id)
        return updated

    async def delete(self, user_id: str, specialty_id: int) -> None:
        deleted = await self._repo.delete(specialty_id, user_id)
        if not deleted:
            raise NotFoundError("Specialty", specialty_id)

    async def add_books(
        self, user_id: str, specialty_id: int, book_ids: list[int]
    ) -> Specialty:
        await self.get(user_id, specialty_id)
        for book_id in book_ids:
            await self._add_owned_book(user_id, specialty_id, book_id)
        return await self.get(user_id, specialty_id)

    async def remove_book(self, user_id: str, specialty_id: int, book_id: int) -> Specialty:
        await self.get(user_id, specialty_id)
        removed = await self._repo.remove_book(specialty_id, book_id)
        if not removed:
            raise NotFoundError("Book in specialty", book_id)
        return await self.get(user_id, specialty_id)

    async def confirm_corpus(
        self,
        user_id: str,
        specialty_id: int,
        approved_indices: list[int],
    ) -> Specialty:
        specialty = await self.get(user_id, specialty_id)

        if not specialty.pending_corpus:
            raise ConflictError("No pending corpus suggestions for this specialty")

        max_idx = len(specialty.pending_corpus) - 1
        invalid = [i for i in approved_indices if i < 0 or i > max_idx]
        if invalid:
            raise ConflictError(f"Invalid suggestion indices: {invalid}")

        approved: list[SuggestedSource] = [specialty.pending_corpus[i] for i in approved_indices]
        book_svc = BookService(self._pool)

        for item in approved:
            if item.source_url is None:
                continue
            try:
                meta = BookMetadata(title=item.title, author=item.author)
                book = await book_svc.add_from_url(user_id=user_id, url=item.source_url, meta=meta)
                await self._repo.add_book(specialty_id, book.id)
                asyncio.create_task(ingest_book(book.id, self._pool))
            except Exception:
                pass

        await self._repo.clear_pending_corpus(specialty_id)
        return await self.get(user_id, specialty_id)

    async def corpus_status(self, user_id: str, specialty_id: int) -> CorpusStatusResponse:
        specialty = await self.get(user_id, specialty_id)

        statuses: list[CorpusBookStatus] = []
        for book_id in specialty.book_ids:
            book = await self._book_repo.get(book_id, user_id)
            if book is None:
                continue
            statuses.append(
                CorpusBookStatus(
                    book_id=book.id,
                    title=book.metadata.title,
                    author=book.metadata.author,
                    status=book.status,
                )
            )

        counts = {s: 0 for s in ("ready", "processing", "failed", "pending")}
        for s in statuses:
            if s.status in counts:
                counts[s.status] += 1

        return CorpusStatusResponse(
            specialty_id=specialty_id,
            total=len(statuses),
            books=statuses,
            **counts,
        )

    async def resolve_scope(
        self,
        user_id: str,
        specialty_id: int | None,
        book_ids: list[int] | None = None,
    ) -> tuple[str | None, list[int] | None]:
        """Resolve a specialty into (persona, effective book_ids) for retrieval.

        With no specialty, returns the explicit book_ids unchanged (None = whole
        library). With a specialty, returns its persona and its books — intersected
        with explicit book_ids when both are given.
        """
        if specialty_id is None:
            return None, book_ids

        specialty = await self.get(user_id, specialty_id)
        if not specialty.book_ids:
            raise ConflictError(
                f"Specialty '{specialty.name}' has no books — add books before querying it"
            )
        scoped = specialty.book_ids
        if book_ids:
            scoped = [b for b in scoped if b in set(book_ids)]
            if not scoped:
                raise ConflictError(
                    f"None of the given book_ids belong to specialty '{specialty.name}'"
                )
        return specialty.persona, scoped

    async def _add_owned_book(self, user_id: str, specialty_id: int, book_id: int) -> None:
        book = await self._book_repo.get(book_id, user_id)
        if book is None:
            raise NotFoundError("Book", book_id)
        await self._repo.add_book(specialty_id, book_id)
