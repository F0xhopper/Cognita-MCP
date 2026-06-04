from openai import AsyncOpenAI

from cognita.core.config import settings
from cognita.core.exceptions import EmbeddingError
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_client: AsyncOpenAI | None = None


def get_embeddings_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def embed_text(text: str) -> list[float]:
    client = get_embeddings_client()
    try:
        resp = await client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=text.replace("\n", " "),
        )
        return resp.data[0].embedding
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise EmbeddingError(str(exc)) from exc


async def embed_batch(texts: list[str]) -> list[list[float]]:
    client = get_embeddings_client()
    cleaned = [t.replace("\n", " ") for t in texts]
    try:
        resp = await client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=cleaned,
        )
        resp.data.sort(key=lambda x: x.index)
        return [item.embedding for item in resp.data]
    except Exception as exc:
        logger.error("Batch embedding failed: %s", exc)
        raise EmbeddingError(str(exc)) from exc
