"""Download a book from a URL, safely.

Two concerns, both handled here:

  * **SSRF** — the server may be reachable from a network the caller is not. The
    target hostname is resolved and checked against private, loopback and
    link-local ranges before connecting, and re-checked on every redirect hop so
    a public URL cannot bounce into the internal network.
  * **Format** — the content type is established from the response headers, the
    attachment filename, the URL path, and finally the file's magic bytes,
    because plenty of public archives serve books as ``application/octet-stream``.
"""

import asyncio
import ipaddress
import os
import re
import socket
from urllib.parse import unquote, urlparse

import httpx

from cognita.books.domain import BookFormat
from cognita.core.config import settings
from cognita.core.exceptions import UnsupportedFormatError, UrlFetchError
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 60.0
_CHUNK = 64 * 1024

_CONTENT_TYPES: dict[str, BookFormat] = {
    "application/pdf": BookFormat.PDF,
    "application/x-pdf": BookFormat.PDF,
    "application/epub+zip": BookFormat.EPUB,
    "application/epub": BookFormat.EPUB,
    "text/plain": BookFormat.TXT,
    "text/markdown": BookFormat.MD,
    "text/x-markdown": BookFormat.MD,
    "text/html": BookFormat.HTML,
    "application/xhtml+xml": BookFormat.HTML,
}

_EXTENSIONS: dict[str, BookFormat] = {
    ".pdf": BookFormat.PDF,
    ".epub": BookFormat.EPUB,
    ".txt": BookFormat.TXT,
    ".text": BookFormat.TXT,
    ".md": BookFormat.MD,
    ".markdown": BookFormat.MD,
    ".html": BookFormat.HTML,
    ".htm": BookFormat.HTML,
}

# Ranges the stdlib flags do not already cover.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),     # "this network"
    ipaddress.ip_network("100.64.0.0/10"),  # carrier-grade NAT
    ipaddress.ip_network("fc00::/7"),        # IPv6 unique local
]

_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.IGNORECASE)


def _check_host(hostname: str) -> None:
    """Raise if `hostname` resolves to any address we must not connect to."""
    if not hostname:
        raise UrlFetchError("URL has no host")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UrlFetchError(f"Cannot resolve hostname '{hostname}'") from exc

    for *_, sockaddr in infos:
        try:
            addr = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        blocked = (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or any(addr in net for net in _BLOCKED_NETWORKS)
        )
        if blocked:
            raise UrlFetchError(f"URL resolves to a non-routable address ({addr})")


def _detect_format(
    content_type: str | None,
    content_disposition: str | None,
    url: str,
    head_bytes: bytes = b"",
) -> BookFormat:
    if content_type:
        base = content_type.split(";")[0].strip().lower()
        if base in _CONTENT_TYPES:
            return _CONTENT_TYPES[base]

    if content_disposition and (match := _FILENAME_RE.search(content_disposition)):
        ext = os.path.splitext(unquote(match.group(1)))[1].lower()
        if ext in _EXTENSIONS:
            return _EXTENSIONS[ext]

    ext = os.path.splitext(unquote(urlparse(url).path))[1].lower()
    if ext in _EXTENSIONS:
        return _EXTENSIONS[ext]

    # Archives frequently serve books as octet-stream; the bytes do not lie.
    if head_bytes.startswith(b"%PDF"):
        return BookFormat.PDF
    if head_bytes.startswith(b"PK\x03\x04") and b"epub" in head_bytes[:200].lower():
        return BookFormat.EPUB
    if head_bytes.strip()[:1] == b"<":
        return BookFormat.HTML

    raise UnsupportedFormatError(content_type or ext or "unknown")


async def fetch_book_from_url(url: str) -> tuple[bytes, BookFormat, str]:
    """Download `url`. Returns (bytes, format, suggested filename)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlFetchError(f"Only http and https URLs are supported (got '{parsed.scheme}')")

    max_bytes = settings.MAX_FILE_MB * 1024 * 1024
    await asyncio.to_thread(_check_host, parsed.hostname or "")

    async def on_request(request: httpx.Request) -> None:
        if request.url.host:
            await asyncio.to_thread(_check_host, request.url.host)

    async with httpx.AsyncClient(
        follow_redirects=True,
        max_redirects=5,
        timeout=_TIMEOUT,
        headers={"User-Agent": "Cognita/2.0 (personal library MCP)"},
        event_hooks={"request": [on_request]},
    ) as client:
        parts: list[bytes] = []
        total = 0
        try:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()

                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise UrlFetchError(
                        f"File is {int(declared) // (1024 * 1024)} MB, "
                        f"over the {settings.MAX_FILE_MB} MB limit"
                    )

                async for chunk in resp.aiter_bytes(chunk_size=_CHUNK):
                    total += len(chunk)
                    if total > max_bytes:
                        raise UrlFetchError(
                            f"Download exceeded the {settings.MAX_FILE_MB} MB limit"
                        )
                    parts.append(chunk)

                content_type = resp.headers.get("content-type")
                disposition = resp.headers.get("content-disposition")
                final_url = str(resp.url)
        except httpx.HTTPStatusError as exc:
            raise UrlFetchError(
                f"HTTP {exc.response.status_code} downloading {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise UrlFetchError(f"Download failed: {exc}") from exc

    data = b"".join(parts)
    if not data:
        raise UrlFetchError("Downloaded file is empty")

    fmt = _detect_format(content_type, disposition, final_url, data[:512])

    filename = os.path.basename(unquote(urlparse(final_url).path)) or "book"
    if not os.path.splitext(filename)[1]:
        filename = f"{filename}.{fmt}"

    logger.info("Fetched %s (%d KB, %s)", url, len(data) // 1024, fmt)
    return data, fmt, filename
