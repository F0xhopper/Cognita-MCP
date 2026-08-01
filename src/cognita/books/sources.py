"""Find a downloadable copy of a book from its title.

Searches free, public-domain libraries in order of text quality — a Standard
Ebooks or Gutenberg edition is a clean transcription, whereas an Internet
Archive scan is OCR of variable accuracy — and returns the first hit:

  1. Project Gutenberg (via gutendex)
  2. Standard Ebooks (OPDS catalogue)
  3. Open Library → Internet Archive
  4. Internet Archive full-text search
  5. Wikisource (EPUB generated on demand)

Every call is best-effort: a source that errors or times out is skipped, and if
nothing matches the caller is told to supply a file themselves. Nothing here
raises.
"""

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import quote

import httpx

from cognita.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 10.0
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

_GUTENDEX = "https://gutendex.com/books/"
_STANDARD_EBOOKS_OPDS = "https://standardebooks.org/opds/all"
_OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
_ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
_ARCHIVE_META = "https://archive.org/metadata/{identifier}"
_WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
_WIKISOURCE_EXPORT = "https://ws-export.wmcloud.org/"

# EPUB first: it keeps chapter structure, which the chunker uses. Plain text
# is a fine second. PDF is last — it is usually a scan.
_GUTENBERG_FORMATS = ["application/epub+zip", "text/plain; charset=utf-8", "text/plain"]


class SourceType(StrEnum):
    GUTENBERG = "gutenberg"
    STANDARD_EBOOKS = "standard_ebooks"
    ARCHIVE_ORG = "archive_org"
    WIKISOURCE = "wikisource"


@dataclass
class ResolvedSource:
    title: str
    author: str | None
    url: str
    source_type: SourceType


async def resolve_source(title: str, author: str | None = None) -> ResolvedSource | None:
    """Find a downloadable edition of `title`, or None if none is available."""
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        for finder in (
            _try_gutenberg,
            _try_standard_ebooks,
            _try_open_library,
            _try_archive,
            _try_wikisource,
        ):
            try:
                found = await finder(client, title, author or "")
            except Exception as exc:  # noqa: BLE001 — one dead source must not stop the rest
                logger.debug("%s failed for %r: %s", finder.__name__, title, exc)
                continue
            if found is not None:
                url, source_type = found
                logger.info("Resolved %r to %s (%s)", title, source_type, url)
                return ResolvedSource(
                    title=title, author=author, url=url, source_type=source_type
                )
    logger.info("No public-domain source found for %r", title)
    return None


async def resolve_many(
    items: list[tuple[str, str | None]],
) -> list[ResolvedSource | None]:
    """Resolve several titles concurrently, preserving input order."""
    results = await asyncio.gather(
        *(resolve_source(title, author) for title, author in items),
        return_exceptions=True,
    )
    return [r if isinstance(r, ResolvedSource) else None for r in results]


# ── Individual sources ────────────────────────────────────────────────────────

async def _try_gutenberg(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str, SourceType] | None:
    for query in _queries(title, author):
        resp = await client.get(_GUTENDEX, params={"search": query})
        resp.raise_for_status()
        for book in resp.json().get("results", [])[:3]:
            formats: dict[str, str] = book.get("formats", {})
            for mime in _GUTENBERG_FORMATS:
                if mime in formats:
                    return formats[mime], SourceType.GUTENBERG
    return None


async def _try_standard_ebooks(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str, SourceType] | None:
    for query in _queries(title, author):
        resp = await client.get(_STANDARD_EBOOKS_OPDS, params={"query": query})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for entry in root.findall("atom:entry", _ATOM_NS):
            for link in entry.findall("atom:link", _ATOM_NS):
                if link.get("type") == "application/epub+zip" and link.get("href"):
                    return link.get("href", ""), SourceType.STANDARD_EBOOKS
    return None


async def _try_open_library(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str, SourceType] | None:
    params = {
        "title": title,
        "has_fulltext": "true",
        "fields": "ia,public_scan_b",
        "limit": "3",
    }
    if author:
        params["author"] = author

    resp = await client.get(_OPEN_LIBRARY_SEARCH, params=params)
    resp.raise_for_status()
    for doc in resp.json().get("docs", []):
        if not doc.get("public_scan_b"):
            continue
        ia = doc.get("ia")
        identifier = ia[0] if isinstance(ia, list) and ia else ia
        if not identifier:
            continue
        if found := await _archive_best_file(client, identifier):
            return found
    return None


async def _try_archive(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str, SourceType] | None:
    query = f"title:({title})" + (f" AND creator:({author})" if author else "")
    resp = await client.get(
        _ARCHIVE_SEARCH,
        params={
            "q": query,
            "fl[]": "identifier",
            "rows": "3",
            "output": "json",
            "mediatype": "texts",
        },
    )
    resp.raise_for_status()
    for doc in resp.json().get("response", {}).get("docs", []):
        identifier = doc.get("identifier")
        if not identifier:
            continue
        if found := await _archive_best_file(client, identifier):
            return found
    return None


async def _archive_best_file(
    client: httpx.AsyncClient, identifier: str
) -> tuple[str, SourceType] | None:
    resp = await client.get(_ARCHIVE_META.format(identifier=identifier))
    resp.raise_for_status()
    files: list[dict] = resp.json().get("files", [])
    for fmt in ("EPUB", "DjVuTXT", "Text", "PDF"):
        name = next((f["name"] for f in files if f.get("format") == fmt), None)
        if name:
            url = f"https://archive.org/download/{identifier}/{quote(name)}"
            return url, SourceType.ARCHIVE_ORG
    return None


async def _try_wikisource(
    client: httpx.AsyncClient, title: str, author: str
) -> tuple[str, SourceType] | None:
    for query in _queries(title, author):
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
        for hit in resp.json().get("query", {}).get("search", []):
            page_title = hit.get("title")
            if page_title:
                url = f"{_WIKISOURCE_EXPORT}?lang=en&format=epub&title={quote(page_title)}"
                return url, SourceType.WIKISOURCE
    return None


def _queries(title: str, author: str) -> list[str]:
    """Title with author first — it disambiguates — then title alone."""
    return [f"{title} {author}".strip(), title] if author else [title]
