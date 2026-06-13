"""Resolve suggested titles to public-domain download URLs.

Resolution order per item:
  1. Project Gutenberg via gutendex.com (EPUB preferred, then plain text)
  2. Standard Ebooks via OPDS catalog (EPUB)
  3. Open Library search → Internet Archive file (PDF preferred, then EPUB)
  4. Internet Archive advancedsearch + metadata API (PDF preferred, then EPUB)
  5. Wikisource via ws-export.wmcloud.org (EPUB, generated on demand)
  6. Fallback: source_type = USER_UPLOAD_REQUIRED, source_url = None

All network calls are best-effort — a failure leaves the item as user_upload_required.
"""

import asyncio
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

from cognita.specialties.domain import SourceType, SuggestedSource

_TIMEOUT = 10.0
_GUTENBERG_API = "https://gutendex.com/books/"
_STANDARD_EBOOKS_OPDS = "https://standardebooks.org/opds/all"
_OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
_ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
_ARCHIVE_META = "https://archive.org/metadata/{identifier}"
_WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
_WIKISOURCE_EXPORT = "https://ws-export.wmcloud.org/"

_GUTENBERG_MIME_PRIORITY = [
    "application/epub+zip",
    "text/plain; charset=utf-8",
    "text/plain",
]

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

_RESOLVERS = [
    "_try_gutenberg",
    "_try_standard_ebooks",
    "_try_open_library",
    "_try_archive",
    "_try_wikisource",
]


async def resolve_sources(sources: list[SuggestedSource]) -> list[SuggestedSource]:
    """Enrich each source with a download URL in parallel. Never raises."""
    results = await asyncio.gather(*[_resolve_one(s) for s in sources], return_exceptions=True)
    return [
        r if isinstance(r, SuggestedSource) else sources[i]
        for i, r in enumerate(results)
    ]


async def _resolve_one(source: SuggestedSource) -> SuggestedSource:
    url: str | None = None
    stype = SourceType.USER_UPLOAD_REQUIRED

    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        for fn in (
            _try_gutenberg,
            _try_standard_ebooks,
            _try_open_library,
            _try_archive,
            _try_wikisource,
        ):
            url, stype = await fn(client, source.title, source.author)
            if url is not None:
                break

    return SuggestedSource(
        title=source.title,
        author=source.author,
        tier=source.tier,
        rationale=source.rationale,
        source_url=url,
        source_type=stype,
        approved=source.approved,
    )


async def _try_gutenberg(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str | None, SourceType]:
    for query in (f"{title} {author}", title):
        try:
            resp = await client.get(_GUTENBERG_API, params={"search": query})
            resp.raise_for_status()
            for book in resp.json().get("results", [])[:3]:
                formats: dict[str, str] = book.get("formats", {})
                for mime in _GUTENBERG_MIME_PRIORITY:
                    if mime in formats:
                        return formats[mime], SourceType.GUTENBERG
        except Exception:
            pass
    return None, SourceType.USER_UPLOAD_REQUIRED


async def _try_standard_ebooks(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str | None, SourceType]:
    for query in (f"{title} {author}", title):
        try:
            resp = await client.get(_STANDARD_EBOOKS_OPDS, params={"query": query})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for entry in root.findall("atom:entry", _ATOM_NS):
                for link in entry.findall("atom:link", _ATOM_NS):
                    if link.get("type") == "application/epub+zip":
                        href = link.get("href")
                        if href:
                            return href, SourceType.STANDARD_EBOOKS
        except Exception:
            pass
    return None, SourceType.USER_UPLOAD_REQUIRED


async def _try_open_library(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str | None, SourceType]:
    try:
        resp = await client.get(
            _OPEN_LIBRARY_SEARCH,
            params={
                "title": title,
                "author": author,
                "has_fulltext": "true",
                "fields": "ia,public_scan_b",
                "limit": "3",
            },
        )
        resp.raise_for_status()
        for doc in resp.json().get("docs", []):
            if not doc.get("public_scan_b"):
                continue
            ia = doc.get("ia")
            ident = ia[0] if isinstance(ia, list) else ia
            if not ident:
                continue
            url, stype = await _archive_best_file(client, ident)
            if url:
                return url, stype
    except Exception:
        pass
    return None, SourceType.USER_UPLOAD_REQUIRED


async def _try_archive(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str | None, SourceType]:
    try:
        resp = await client.get(
            _ARCHIVE_SEARCH,
            params={
                "q": f"title:({title}) AND creator:({author})",
                "fl[]": "identifier",
                "rows": "3",
                "output": "json",
                "mediatype": "texts",
            },
        )
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
        for doc in docs:
            ident = doc.get("identifier")
            if not ident:
                continue
            url, stype = await _archive_best_file(client, ident)
            if url:
                return url, stype
    except Exception:
        pass
    return None, SourceType.USER_UPLOAD_REQUIRED


async def _archive_best_file(
    client: httpx.AsyncClient, identifier: str
) -> tuple[str | None, SourceType]:
    try:
        resp = await client.get(_ARCHIVE_META.format(identifier=identifier))
        resp.raise_for_status()
        files: list[dict] = resp.json().get("files", [])
        pdf_name = next((f["name"] for f in files if f.get("format") == "PDF"), None)
        if pdf_name:
            return f"https://archive.org/download/{identifier}/{pdf_name}", SourceType.ARCHIVE_ORG
        epub_name = next((f["name"] for f in files if f.get("format") == "EPUB"), None)
        if epub_name:
            return f"https://archive.org/download/{identifier}/{epub_name}", SourceType.ARCHIVE_ORG
    except Exception:
        pass
    return None, SourceType.USER_UPLOAD_REQUIRED


async def _try_wikisource(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str | None, SourceType]:
    for query in (f"{title} {author}", title):
        try:
            resp = await client.get(
                _WIKISOURCE_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": "0",
                    "srlimit": "3",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
            for hit in hits:
                page_title = hit.get("title", "")
                if not page_title:
                    continue
                export_url = f"{_WIKISOURCE_EXPORT}?lang=en&format=epub&title={quote(page_title)}"
                return export_url, SourceType.WIKISOURCE
        except Exception:
            pass
    return None, SourceType.USER_UPLOAD_REQUIRED
