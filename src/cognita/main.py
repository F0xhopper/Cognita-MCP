import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cognita.api.middleware.cors import add_cors_middleware
from cognita.api.middleware.error_handlers import add_error_handlers
from cognita.api.routes.health import router as health_router
from cognita.books.repository import BookRepository
from cognita.books.router import router as books_router
from cognita.chunks.repository import ChunkRepository
from cognita.core.config import settings
from cognita.core.logging import get_logger, setup_logging
from cognita.infrastructure.database import close_pool, get_pool, init_pool
from cognita.search.router import router as search_router
from cognita.specialties.repository import SpecialtyRepository
from cognita.specialties.router import router as specialties_router

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = settings.check_required_vars()
    if missing:
        logger.error("Missing required env vars: %s — requests will fail", ", ".join(missing))

    logger.info("Starting Cognita API…")
    await init_pool()

    pool = get_pool()
    await BookRepository(pool).ensure_schema()
    await ChunkRepository(pool).ensure_schema()
    await SpecialtyRepository(pool).ensure_schema()

    logger.info("Ready.")
    yield
    await close_pool()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cognita MCP API",
        description="Book Intelligence API — upload, ingest, and search your personal library",
        version="1.0.0",
        lifespan=lifespan,
    )

    add_cors_middleware(app)
    add_error_handlers(app)

    app.include_router(health_router)
    app.include_router(books_router)
    app.include_router(search_router)
    app.include_router(specialties_router)

    return app


app = create_app()


def run() -> None:
    import uvicorn
    uvicorn.run("cognita.main:app", host=settings.HOST, port=settings.PORT, reload=False)
