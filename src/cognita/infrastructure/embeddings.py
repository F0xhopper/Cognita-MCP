"""Embeddings via OpenAI.

EMBED_DIM is passed through to the API, so lowering it genuinely produces
shorter vectors rather than silently disagreeing with the database column.
text-embedding-3-* support this natively; older models do not, so the parameter
is only sent when the model accepts it.
"""

import asyncio

from openai import AsyncOpenAI

from cognita.core.config import settings
from cognita.core.exceptions import EmbeddingError
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_client: AsyncOpenAI | None = None
# One embedding request at a time keeps a folder import from tripping rate
# limits while a search is also running.
_semaphore = asyncio.Semaphore(2)

# Only the v3 models accept a dimensions parameter.
_SUPPORTS_DIMENSIONS = ("text-embedding-3",)


def get_embeddings_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _request_kwargs() -> dict:
    if settings.EMBED_MODEL.startswith(_SUPPORTS_DIMENSIONS):
        return {"dimensions": settings.EMBED_DIM}
    return {}


async def embed_text(text: str) -> list[float]:
    return (await embed_batch([text]))[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    client = get_embeddings_client()
    cleaned = [t.replace("\n", " ") or " " for t in texts]

    async with _semaphore:
        try:
            response = await client.embeddings.create(
                model=settings.EMBED_MODEL,
                input=cleaned,
                **_request_kwargs(),
            )
        except Exception as exc:
            logger.error("Embedding request failed: %s", exc)
            raise EmbeddingError(str(exc)) from exc

    # The API may return items out of order; index is authoritative.
    ordered = sorted(response.data, key=lambda item: item.index)
    embeddings = [item.embedding for item in ordered]

    if embeddings and len(embeddings[0]) != settings.EMBED_DIM:
        raise EmbeddingError(
            f"{settings.EMBED_MODEL} returned {len(embeddings[0])}-dimensional vectors "
            f"but EMBED_DIM is {settings.EMBED_DIM}. Set EMBED_DIM to match the model."
        )
    return embeddings
