"""Integration tests for the books API.

Requires a running PostgreSQL with pgvector and a valid DATABASE_URL in .env.
Run with: pytest tests/integration -v
"""

import pytest
from httpx import ASGITransport, AsyncClient

from cognita.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def auth_headers():
    """Override with a real Supabase JWT for full integration testing."""
    return {"Authorization": "Bearer test-token"}


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_upload_requires_auth(client):
    resp = await client.post("/books/upload", data={"metadata": "{}"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_books_requires_auth(client):
    resp = await client.get("/books/")
    assert resp.status_code == 403
