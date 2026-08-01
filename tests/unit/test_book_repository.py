"""Copying a book out of Postgres must be byte-exact and bounded in memory.

The point of write_file_to() is that a 100MB book never exists as a 100MB
Python object, so these tests assert on the slice *boundaries* as much as on
the reassembled file: a version that quietly fetched the whole column would
still produce the right bytes.
"""

from pathlib import Path

import pytest

from cognita.books import repository
from cognita.books.repository import BookRepository


class _FakeConnection:
    """Answers the two queries write_file_to() issues, over an in-memory book."""

    def __init__(self, data: bytes | None, vanish_after: int | None = None) -> None:
        self._data = data
        self._vanish_after = vanish_after
        self.slice_args: list[tuple[int, int]] = []

    async def fetchval(self, sql: str, *args):
        if "octet_length" in sql:
            return None if self._data is None else len(self._data)

        _book_id, start, count = args
        self.slice_args.append((start, count))
        if self._vanish_after is not None and len(self.slice_args) > self._vanish_after:
            return None
        return self._data[start - 1 : start - 1 + count]


class _FakePool:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Acquire:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


@pytest.fixture(autouse=True)
def _tiny_slices(monkeypatch):
    """Four-byte slices so the boundary maths is visible in the assertions."""
    monkeypatch.setattr(repository, "_SLICE_BYTES", 4)


def _repo(data: bytes | None, vanish_after: int | None = None):
    conn = _FakeConnection(data, vanish_after)
    return BookRepository(_FakePool(conn)), conn


# ── Reassembly ────────────────────────────────────────────────────────────────

async def test_writes_file_byte_exact(tmp_path: Path):
    data = bytes(range(256)) * 3
    repo, _ = _repo(data)

    await repo.write_file_to(1, tmp_path / "book.pdf")

    assert (tmp_path / "book.pdf").read_bytes() == data


async def test_reads_in_slices_not_all_at_once(tmp_path: Path):
    repo, conn = _repo(b"0123456789")

    await repo.write_file_to(1, tmp_path / "book.pdf")

    # 10 bytes at 4 per slice: offsets 1, 5, 9 — 1-indexed, and the last slice
    # runs past the end, which substring() truncates rather than erroring.
    assert conn.slice_args == [(1, 4), (5, 4), (9, 4)]


async def test_exact_multiple_of_slice_size(tmp_path: Path):
    repo, conn = _repo(b"12345678")

    await repo.write_file_to(1, tmp_path / "book.pdf")

    assert conn.slice_args == [(1, 4), (5, 4)]
    assert (tmp_path / "book.pdf").read_bytes() == b"12345678"


async def test_file_smaller_than_one_slice(tmp_path: Path):
    repo, conn = _repo(b"ab")

    await repo.write_file_to(1, tmp_path / "book.pdf")

    assert conn.slice_args == [(1, 4)]
    assert (tmp_path / "book.pdf").read_bytes() == b"ab"


# ── Edge cases ────────────────────────────────────────────────────────────────

async def test_empty_file_creates_empty_file(tmp_path: Path):
    """A book row with no bytes yet: no slice queries, but a file to parse."""
    repo, conn = _repo(b"")

    await repo.write_file_to(1, tmp_path / "book.pdf")

    assert conn.slice_args == []
    assert (tmp_path / "book.pdf").read_bytes() == b""


async def test_missing_book_raises_keyerror(tmp_path: Path):
    repo, _ = _repo(None)

    with pytest.raises(KeyError):
        await repo.write_file_to(999, tmp_path / "book.pdf")


async def test_book_deleted_mid_read_raises_keyerror(tmp_path: Path):
    """delete_book() racing an in-flight ingestion must not write a truncated file."""
    repo, _ = _repo(b"0123456789", vanish_after=1)

    with pytest.raises(KeyError):
        await repo.write_file_to(1, tmp_path / "book.pdf")
