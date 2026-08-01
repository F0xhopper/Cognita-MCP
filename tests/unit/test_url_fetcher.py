"""URL fetching: format detection, and the SSRF guard."""

import pytest

from cognita.books.domain import BookFormat
from cognita.books.url_fetcher import _check_host, _detect_format, fetch_book_from_url
from cognita.core.exceptions import UnsupportedFormatError, UrlFetchError

# ── Format detection ──────────────────────────────────────────────────────────

def test_content_type_wins():
    assert _detect_format("application/pdf", None, "http://x/y", b"") == BookFormat.PDF
    assert _detect_format("application/epub+zip", None, "http://x/y", b"") == BookFormat.EPUB
    assert _detect_format("text/markdown", None, "http://x/y", b"") == BookFormat.MD


def test_content_type_parameters_are_ignored():
    assert _detect_format("text/plain; charset=utf-8", None, "http://x/y", b"") == BookFormat.TXT


def test_attachment_filename_is_used_next():
    fmt = _detect_format(
        "application/octet-stream",
        'attachment; filename="the-republic.epub"',
        "http://x/download",
        b"",
    )
    assert fmt == BookFormat.EPUB


def test_url_extension_is_used_after_that():
    fmt = _detect_format("application/octet-stream", None, "http://x/book.pdf", b"")
    assert fmt == BookFormat.PDF


def test_url_extension_survives_a_query_string():
    fmt = _detect_format(None, None, "http://x/book.epub?token=abc", b"")
    assert fmt == BookFormat.EPUB


def test_magic_bytes_are_the_last_resort():
    """Archives commonly serve books as octet-stream with no useful filename."""
    assert _detect_format(None, None, "http://x/download", b"%PDF-1.7\n...") == BookFormat.PDF
    assert _detect_format(None, None, "http://x/d", b"<html><body>") == BookFormat.HTML


def test_unrecognisable_content_is_rejected():
    with pytest.raises(UnsupportedFormatError):
        _detect_format("application/zip", None, "http://x/thing", b"\x00\x01\x02")


# ── SSRF guard ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"])
def test_private_and_metadata_addresses_are_refused(host):
    with pytest.raises(UrlFetchError):
        _check_host(host)


def test_unresolvable_host_is_refused():
    with pytest.raises(UrlFetchError):
        _check_host("this-host-does-not-exist.invalid")


def test_empty_host_is_refused():
    with pytest.raises(UrlFetchError):
        _check_host("")


async def test_non_http_schemes_are_refused():
    for url in ("file:///etc/passwd", "ftp://example.com/book.pdf", "gopher://x"):
        with pytest.raises(UrlFetchError, match="http"):
            await fetch_book_from_url(url)
