"""SSRF-safe async URL fetcher for book downloads.

Validates the target hostname against private/loopback IP ranges before
fetching, and re-checks on every redirect hop. Enforces a 50 MB cap and
restricts content-types to PDF, EPUB, and plain-text.
"""

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urlparse

import httpx

from cognita.books.domain import BookFormat
from cognita.core.exceptions import UnsupportedFormatError, UrlFetchError

_ALLOWED_CONTENT_TYPES: dict[str, BookFormat] = {
    "application/pdf": BookFormat.PDF,
    "application/epub+zip": BookFormat.EPUB,
    "text/plain": BookFormat.TXT,
}
_ALLOWED_EXTENSIONS: dict[str, BookFormat] = {
    ".pdf": BookFormat.PDF,
    ".epub": BookFormat.EPUB,
    ".txt": BookFormat.TXT,
}
_FORMAT_EXT: dict[BookFormat, str] = {
    BookFormat.PDF: ".pdf",
    BookFormat.EPUB: ".epub",
    BookFormat.TXT: ".txt",
}
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_TIMEOUT = 30.0

# Networks to block in addition to the stdlib is_private / is_loopback checks.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),      # this network
    ipaddress.ip_network("100.64.0.0/10"),   # CGNAT / shared address space
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
]


def _check_host(hostname: str) -> None:
    """Resolve hostname and raise UrlFetchError if it lands on a private address."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UrlFetchError(f"Cannot resolve hostname '{hostname}'") from exc

    for _, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_multicast:
            raise UrlFetchError(f"URL resolves to a non-routable address ({ip_str})")
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise UrlFetchError(f"URL resolves to a non-routable address ({ip_str})")


def _detect_format(content_type: str | None, url: str) -> BookFormat:
    if content_type:
        base = content_type.split(";")[0].strip().lower()
        if base in _ALLOWED_CONTENT_TYPES:
            return _ALLOWED_CONTENT_TYPES[base]

    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in _ALLOWED_EXTENSIONS:
        return _ALLOWED_EXTENSIONS[ext]

    raise UnsupportedFormatError(content_type or "unknown")


async def fetch_book_from_url(url: str) -> tuple[bytes, BookFormat, str]:
    """Download a book from *url* with SSRF protection and size enforcement.

    Returns ``(file_bytes, format, suggested_filename)``.
    Raises ``UrlFetchError`` for network / policy failures,
    ``UnsupportedFormatError`` for unrecognised content-types.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlFetchError(f"Only http/https URLs are supported (got '{parsed.scheme}')")

    # Block private IPs before the first connection.
    await asyncio.to_thread(_check_host, parsed.hostname or "")

    # Re-check on every redirect so a public→private redirect is also blocked.
    async def _on_request(request: httpx.Request) -> None:
        host = request.url.host
        if host:
            await asyncio.to_thread(_check_host, host)

    async with httpx.AsyncClient(
        follow_redirects=True,
        max_redirects=3,
        timeout=_TIMEOUT,
        event_hooks={"request": [_on_request]},
    ) as client:
        # HEAD first — cheap validation of type and declared size.
        try:
            head = await client.head(url)
        except httpx.HTTPError as exc:
            raise UrlFetchError(f"HEAD request failed: {exc}") from exc

        content_type = head.headers.get("content-type")
        content_length_raw = head.headers.get("content-length")
        if content_length_raw:
            try:
                declared = int(content_length_raw)
                if declared > _MAX_BYTES:
                    raise UrlFetchError(
                        f"Declared file size {declared // (1024 * 1024)} MB "
                        f"exceeds the {_MAX_BYTES // (1024 * 1024)} MB limit"
                    )
            except ValueError:
                pass  # malformed header — we'll catch it during streaming

        fmt = _detect_format(content_type, url)

        # Streamed GET with a hard byte cap.
        try:
            parts: list[bytes] = []
            total = 0
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=65_536):
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        raise UrlFetchError(
                            f"Download exceeded {_MAX_BYTES // (1024 * 1024)} MB limit"
                        )
                    parts.append(chunk)
        except httpx.HTTPStatusError as exc:
            raise UrlFetchError(
                f"HTTP {exc.response.status_code} downloading URL"
            ) from exc
        except httpx.HTTPError as exc:
            raise UrlFetchError(f"Download failed: {exc}") from exc

    data = b"".join(parts)

    # Derive a sensible filename from the URL path.
    basename = os.path.basename(urlparse(url).path)
    if not basename or not os.path.splitext(basename)[1]:
        basename = (basename or "book") + _FORMAT_EXT[fmt]

    return data, fmt, basename
