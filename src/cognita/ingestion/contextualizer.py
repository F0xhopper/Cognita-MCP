"""Contextual Retrieval — situate each chunk within its source before indexing.

A chunk pulled out of a book often loses what it refers to ("he argued that this
was the central error…"). Following Anthropic's Contextual Retrieval technique, we
ask a fast model to write a 1–2 sentence blurb situating each chunk in its work,
then embed and full-text-index that blurb alongside the original passage. The blurb
is stored separately (Chunk.context) and never shown in citations — only the
original text is ever quoted.

This runs once at ingestion. To keep it cheap and bounded for large books, each
chunk is situated against its own section (not the whole book), with the section
text marked as a cacheable prefix so chunks sharing a section reuse the prompt
cache. Everything is best-effort: on any failure the chunk's context is left empty
and it falls back to plain text embedding.
"""

import asyncio

from cognita.chunks.domain import Chunk
from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.infrastructure.anthropic_client import get_anthropic_client

logger = get_logger(__name__)

_SYSTEM = (
    "You situate excerpts from a book within their surrounding text so they can be "
    "retrieved accurately by a search engine. Reply with one or two sentences of "
    "context only — no preamble, no quotation, no commentary."
)

_INSTRUCTION = (
    "Write a short, succinct context (1–2 sentences) that situates the chunk above "
    "within the book, so a search engine can retrieve it accurately. Name the work, "
    "author, and the specific topic or argument where relevant. Answer with the "
    "context only."
)


def _section_key(chunk: Chunk) -> tuple[int, int]:
    return (chunk.location.chapter_n or 0, chunk.location.section_n or 0)


def _build_section_index(chunks: list[Chunk]) -> dict[tuple[int, int], str]:
    """Reconstruct each section's text by joining the chunks that belong to it."""
    by_section: dict[tuple[int, int], list[str]] = {}
    for c in chunks:
        by_section.setdefault(_section_key(c), []).append(c.text)
    return {
        key: "\n\n".join(parts)[: settings.CONTEXT_MAX_CHARS]
        for key, parts in by_section.items()
    }


async def contextualize_chunks(
    chunks: list[Chunk],
    book_title: str,
    author: str | None,
) -> list[str]:
    """Return a situating context blurb for each chunk, aligned by index.

    Returns a list of empty strings (no-op) when contextualization is disabled,
    no Anthropic key is configured, or there are no chunks.
    """
    if not chunks:
        return []
    if not settings.context_enabled:
        return ["" for _ in chunks]

    sections = _build_section_index(chunks)
    client = get_anthropic_client()
    sem = asyncio.Semaphore(settings.CONTEXT_CONCURRENCY)
    by_line = f"{book_title}" + (f" by {author}" if author else "")

    async def one(idx: int, chunk: Chunk) -> str:
        loc = chunk.location
        surrounding = sections.get(_section_key(chunk), chunk.text)
        prefix = (
            f"<book>{by_line}</book>\n"
            f"<chapter>{loc.chapter_title or 'Unknown'}</chapter>\n"
            f"<section>{loc.section_title or 'Unknown'}</section>\n"
            f"<surrounding_text>\n{surrounding}\n</surrounding_text>"
        )
        try:
            async with sem:
                resp = await client.messages.create(
                    model=settings.CONTEXT_MODEL,
                    max_tokens=150,
                    system=_SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prefix,
                             "cache_control": {"type": "ephemeral"}},
                            {"type": "text",
                             "text": f"<chunk>\n{chunk.text}\n</chunk>\n\n{_INSTRUCTION}"},
                        ],
                    }],
                )
            return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        except Exception as exc:
            logger.warning("Contextualization failed for chunk %d: %s", idx, exc)
            return ""

    contexts = await asyncio.gather(*(one(i, c) for i, c in enumerate(chunks)))
    filled = sum(1 for c in contexts if c)
    logger.info("Contextualized %d/%d chunks", filled, len(chunks))
    return list(contexts)
