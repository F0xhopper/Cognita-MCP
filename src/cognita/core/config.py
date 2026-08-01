"""Configuration — plain os.getenv, no framework.

Only two variables are strictly required: DATABASE_URL and OPENAI_API_KEY.
Everything else has a working default, and every optional integration
(Anthropic, Mistral) degrades cleanly when its key is absent.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DATABASE_SSL: bool = _flag("DATABASE_SSL", "false")

    # ── Embeddings (required) ────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "3072"))

    # ── Anthropic (optional) ─────────────────────────────────────────────────
    # Powers two search-quality features. Without a key the server still works:
    # ingestion skips the context blurb and search returns plain fused ordering.
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Contextual retrieval — situate each chunk in its source before indexing.
    CONTEXT_ENABLED: bool = _flag("CONTEXT_ENABLED", "true")
    CONTEXT_MODEL: str = os.getenv("CONTEXT_MODEL", "claude-haiku-4-5-20251001")
    CONTEXT_MAX_CHARS: int = int(os.getenv("CONTEXT_MAX_CHARS", "8000"))
    CONTEXT_CONCURRENCY: int = int(os.getenv("CONTEXT_CONCURRENCY", "4"))

    # Reranking — reorder fused candidates by relevance before returning.
    RERANK_ENABLED: bool = _flag("RERANK_ENABLED", "true")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "claude-haiku-4-5-20251001")
    RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "40"))
    RERANK_BATCH_SIZE: int = int(os.getenv("RERANK_BATCH_SIZE", "20"))

    # ── Mistral OCR (optional) ───────────────────────────────────────────────
    # Fallback for scanned PDFs that yield no extractable text.
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_OCR_MODEL: str = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE_CHARS: int = int(os.getenv("CHUNK_SIZE_CHARS", "1500"))
    CHUNK_OVERLAP_CHARS: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

    # ── Retrieval tuning ─────────────────────────────────────────────────────
    # Reciprocal Rank Fusion: score = w / (k + rank), summed across both arms.
    # k dampens the head of each list; raise it to flatten rank differences.
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    RRF_SEMANTIC_WEIGHT: float = float(os.getenv("RRF_SEMANTIC_WEIGHT", "1.0"))
    RRF_KEYWORD_WEIGHT: float = float(os.getenv("RRF_KEYWORD_WEIGHT", "1.0"))

    # HNSW search breadth. Higher = better recall, slower. pgvector filters
    # *after* the index scan, so book-scoped searches get double the breadth.
    HNSW_EF_SEARCH: int = int(os.getenv("HNSW_EF_SEARCH", "100"))

    # Merge hits that sit next to each other in the same book into one passage.
    MERGE_ADJACENT: bool = _flag("MERGE_ADJACENT", "true")
    # Cap on results from any single book, so one book cannot crowd out the
    # rest of the library. 0 disables the cap.
    MAX_PER_BOOK: int = int(os.getenv("MAX_PER_BOOK", "0"))

    # ── Ingestion ────────────────────────────────────────────────────────────
    # Books ingested concurrently. Each book already batches its own embedding
    # and contextualization calls, so 2 is a sane default.
    INGEST_CONCURRENCY: int = int(os.getenv("INGEST_CONCURRENCY", "2"))
    MAX_FILE_MB: int = int(os.getenv("MAX_FILE_MB", "100"))

    # ── MCP transport ────────────────────────────────────────────────────────
    MCP_HOST: str = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT: int = int(os.getenv("MCP_PORT", "8001"))

    # Required to serve over HTTP; callers send it as a bearer token. Unused in
    # stdio mode, where the transport is the process boundary itself.
    AUTH_TOKEN: str = os.getenv("COGNITA_AUTH_TOKEN", "")

    # Hostnames this server answers to, comma-separated. MCP's DNS-rebinding
    # protection rejects any Host header not on this list, so a deployment must
    # declare its public hostname (e.g. "cognita-mcp.fly.dev") or every request
    # comes back 421.
    ALLOWED_HOSTS: list[str] = [
        h.strip() for h in os.getenv("COGNITA_ALLOWED_HOSTS", "").split(",") if h.strip()
    ]
    # Whether the local-disk tools are usable when serving over HTTP. Off by
    # default: on a hosted server, "the disk" is not the caller's disk.
    ALLOW_LOCAL_FILES: bool = _flag("COGNITA_ALLOW_LOCAL_FILES", "false")

    @property
    def anthropic_enabled(self) -> bool:
        return bool(self.ANTHROPIC_API_KEY)

    @property
    def context_enabled(self) -> bool:
        return self.CONTEXT_ENABLED and self.anthropic_enabled

    @property
    def rerank_enabled(self) -> bool:
        return self.RERANK_ENABLED and self.anthropic_enabled

    @property
    def ocr_enabled(self) -> bool:
        return bool(self.MISTRAL_API_KEY)

    def missing_required(self) -> list[str]:
        return [
            name
            for name, value in (
                ("DATABASE_URL", self.DATABASE_URL),
                ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            )
            if not value
        ]


settings = Settings()
