"""Unit tests for specialty scoping and ownership rules."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognita.core.exceptions import ConflictError, NotFoundError
from cognita.specialties.domain import Specialty
from cognita.specialties.service import SpecialtyService


def _make_specialty(**kwargs) -> Specialty:
    defaults = dict(
        id=1,
        user_id="u",
        name="Stoic Philosophy",
        description="Stoicism corpus",
        persona="You are an expert on Stoic philosophy.",
        book_ids=[1, 2, 3],
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )
    defaults.update(kwargs)
    return Specialty(**defaults)


def _make_service() -> SpecialtyService:
    svc = SpecialtyService(MagicMock())
    svc._repo = AsyncMock()
    svc._book_repo = AsyncMock()
    return svc


async def test_resolve_scope_without_specialty_passes_through():
    svc = _make_service()
    persona, book_ids = await svc.resolve_scope("u", None, [5, 6])
    assert persona is None
    assert book_ids == [5, 6]


async def test_resolve_scope_returns_persona_and_books():
    svc = _make_service()
    svc._repo.get.return_value = _make_specialty()
    persona, book_ids = await svc.resolve_scope("u", 1)
    assert persona == "You are an expert on Stoic philosophy."
    assert book_ids == [1, 2, 3]


async def test_resolve_scope_empty_specialty_raises():
    svc = _make_service()
    svc._repo.get.return_value = _make_specialty(book_ids=[])
    with pytest.raises(ConflictError):
        await svc.resolve_scope("u", 1)


async def test_resolve_scope_intersects_explicit_book_ids():
    svc = _make_service()
    svc._repo.get.return_value = _make_specialty(book_ids=[1, 2, 3])
    _, book_ids = await svc.resolve_scope("u", 1, [2, 3, 99])
    assert book_ids == [2, 3]


async def test_resolve_scope_disjoint_book_ids_raises():
    svc = _make_service()
    svc._repo.get.return_value = _make_specialty(book_ids=[1, 2])
    with pytest.raises(ConflictError):
        await svc.resolve_scope("u", 1, [99])


async def test_resolve_scope_unknown_specialty_raises():
    svc = _make_service()
    svc._repo.get.return_value = None
    with pytest.raises(NotFoundError):
        await svc.resolve_scope("u", 42)


async def test_create_duplicate_name_raises():
    svc = _make_service()
    svc._repo.get_by_name.return_value = _make_specialty()
    with pytest.raises(ConflictError):
        await svc.create("u", "Stoic Philosophy")


async def test_add_books_rejects_unowned_book():
    svc = _make_service()
    svc._repo.get.return_value = _make_specialty()
    svc._book_repo.get.return_value = None
    with pytest.raises(NotFoundError):
        await svc.add_books("u", 1, [99])


async def test_add_books_links_owned_books():
    svc = _make_service()
    svc._repo.get.return_value = _make_specialty()
    svc._book_repo.get.return_value = MagicMock()
    await svc.add_books("u", 1, [4, 5])
    linked = [call.args for call in svc._repo.add_book.call_args_list]
    assert linked == [(1, 4), (1, 5)]
