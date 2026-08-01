"""Resolving a title to a downloadable edition.

Every network call is stubbed. What matters is the order sources are tried in
(clean transcriptions before scans) and that a dead source is skipped rather
than fatal.
"""

import httpx

from cognita.books import sources
from cognita.books.sources import ResolvedSource, SourceType, resolve_source


class _Response:
    def __init__(self, payload=None, content=b"", status: int = 200) -> None:
        self._payload = payload or {}
        self.content = content
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        return self._payload


class _Client:
    """Stub AsyncClient answering by URL prefix; unmatched URLs 404."""

    def __init__(self, routes: dict[str, _Response]) -> None:
        self.routes = routes
        self.requested: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url: str, params=None):
        self.requested.append(url)
        for prefix, response in self.routes.items():
            if url.startswith(prefix):
                return response
        return _Response(status=404)


def _install(monkeypatch, routes: dict[str, _Response]) -> _Client:
    client = _Client(routes)
    monkeypatch.setattr(sources.httpx, "AsyncClient", lambda **kw: client)
    return client


_GUTENBERG_HIT = _Response({
    "results": [{"formats": {"application/epub+zip": "https://gutenberg.org/x.epub"}}]
})


# ── Resolution ────────────────────────────────────────────────────────────────

async def test_gutenberg_is_tried_first(monkeypatch):
    client = _install(monkeypatch, {sources._GUTENDEX: _GUTENBERG_HIT})

    result = await resolve_source("The Republic", "Plato")

    assert result.url == "https://gutenberg.org/x.epub"
    assert result.source_type == SourceType.GUTENBERG
    assert len(client.requested) == 1, "a hit must stop the search"


async def test_gutenberg_prefers_epub_over_plain_text(monkeypatch):
    _install(monkeypatch, {
        sources._GUTENDEX: _Response({
            "results": [{
                "formats": {
                    "text/plain": "https://gutenberg.org/x.txt",
                    "application/epub+zip": "https://gutenberg.org/x.epub",
                }
            }]
        })
    })

    assert (await resolve_source("A Book")).url.endswith(".epub")


async def test_standard_ebooks_is_next(monkeypatch):
    opds = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><link type="application/epub+zip" href="https://standardebooks.org/x.epub"/></entry>
    </feed>"""
    _install(monkeypatch, {
        sources._GUTENDEX: _Response({"results": []}),
        sources._STANDARD_EBOOKS_OPDS: _Response(content=opds),
    })

    result = await resolve_source("Middlemarch", "George Eliot")

    assert result.source_type == SourceType.STANDARD_EBOOKS


async def test_archive_is_used_when_the_clean_sources_miss(monkeypatch):
    _install(monkeypatch, {
        sources._GUTENDEX: _Response({"results": []}),
        sources._STANDARD_EBOOKS_OPDS: _Response(content=b"<feed/>"),
        sources._OPEN_LIBRARY_SEARCH: _Response({"docs": []}),
        sources._ARCHIVE_SEARCH: _Response({"response": {"docs": [{"identifier": "abc"}]}}),
        "https://archive.org/metadata/": _Response({
            "files": [{"name": "book.epub", "format": "EPUB"}]
        }),
    })

    result = await resolve_source("Obscure Title")

    assert result.source_type == SourceType.ARCHIVE_ORG
    assert result.url == "https://archive.org/download/abc/book.epub"


async def test_nothing_found_returns_none(monkeypatch):
    _install(monkeypatch, {
        sources._GUTENDEX: _Response({"results": []}),
        sources._STANDARD_EBOOKS_OPDS: _Response(content=b"<feed/>"),
        sources._OPEN_LIBRARY_SEARCH: _Response({"docs": []}),
        sources._ARCHIVE_SEARCH: _Response({"response": {"docs": []}}),
        sources._WIKISOURCE_API: _Response({"query": {"search": []}}),
    })

    assert await resolve_source("A Book Published Last Tuesday") is None


async def test_a_failing_source_is_skipped(monkeypatch):
    """A 500 from one archive must not abort the whole lookup."""
    _install(monkeypatch, {
        sources._GUTENDEX: _Response(status=500),
        sources._STANDARD_EBOOKS_OPDS: _Response(status=503),
        sources._OPEN_LIBRARY_SEARCH: _Response({"docs": []}),
        sources._ARCHIVE_SEARCH: _Response({"response": {"docs": []}}),
        sources._WIKISOURCE_API: _Response({
            "query": {"search": [{"title": "The Book"}]}
        }),
    })

    result = await resolve_source("The Book")

    assert result.source_type == SourceType.WIKISOURCE
    assert "The%20Book" in result.url


async def test_malformed_json_is_survivable(monkeypatch):
    class Broken(_Response):
        def json(self):
            raise ValueError("not json")

    _install(monkeypatch, {
        sources._GUTENDEX: Broken(),
        sources._STANDARD_EBOOKS_OPDS: _Response(content=b"<feed/>"),
        sources._OPEN_LIBRARY_SEARCH: _Response({"docs": []}),
        sources._ARCHIVE_SEARCH: _Response({"response": {"docs": []}}),
        sources._WIKISOURCE_API: _Response({"query": {"search": []}}),
    })

    assert await resolve_source("Anything") is None


async def test_resolved_source_carries_the_requested_identity(monkeypatch):
    _install(monkeypatch, {sources._GUTENDEX: _GUTENBERG_HIT})

    result = await resolve_source("Meditations", "Marcus Aurelius")

    assert result.title == "Meditations"
    assert result.author == "Marcus Aurelius"


# ── Archive file preference ───────────────────────────────────────────────────

async def test_archive_prefers_epub_to_scanned_pdf(monkeypatch):
    client = _Client({
        "https://archive.org/metadata/": _Response({
            "files": [
                {"name": "scan.pdf", "format": "PDF"},
                {"name": "clean.epub", "format": "EPUB"},
            ]
        })
    })
    url, source_type = await sources._archive_best_file(client, "ident")

    assert url.endswith("clean.epub")
    assert source_type == SourceType.ARCHIVE_ORG


async def test_archive_returns_none_with_no_usable_file(monkeypatch):
    client = _Client({
        "https://archive.org/metadata/": _Response({
            "files": [{"name": "cover.jpg", "format": "JPEG"}]
        })
    })
    assert await sources._archive_best_file(client, "ident") is None


# ── Query construction ────────────────────────────────────────────────────────

def test_author_is_tried_before_title_alone():
    assert sources._queries("Republic", "Plato") == ["Republic Plato", "Republic"]


def test_missing_author_gives_a_single_query():
    assert sources._queries("Republic", "") == ["Republic"]


# ── Batch resolution ──────────────────────────────────────────────────────────

async def test_resolve_many_preserves_order_and_gaps(monkeypatch):
    async def fake(title, author=None):
        if title == "found":
            return ResolvedSource(title, author, "https://x/y.epub", SourceType.GUTENBERG)
        return None

    monkeypatch.setattr(sources, "resolve_source", fake)

    results = await sources.resolve_many([("found", None), ("missing", None), ("found", None)])

    assert [r is not None for r in results] == [True, False, True]
