"""Mistral AI integration for document parsing and OCR.

Used during ingestion to extract clean text from scanned PDFs and complex layouts.
For well-formed digital PDFs, pypdf is used directly (cheaper, faster).
"""

import base64
import json
from pathlib import Path

try:
    from mistralai import Mistral
except ImportError:  # mistralai >= 2 moved the client class
    from mistralai.client import Mistral

from cognita.core.config import settings
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_client: Mistral | None = None


def get_mistral_client() -> Mistral:
    global _client
    if _client is None:
        _client = Mistral(api_key=settings.MISTRAL_API_KEY)
    return _client


async def ocr_pdf(file_path: Path) -> str:
    """Run Mistral OCR on a PDF and return clean markdown text."""
    client = get_mistral_client()
    with file_path.open("rb") as f:
        encoded = base64.standard_b64encode(f.read()).decode()

    logger.info("Running Mistral OCR on %s", file_path.name)
    resp = client.ocr.process(
        model=settings.MISTRAL_OCR_MODEL,
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded}",
        },
        include_image_base64=False,
    )

    pages = [p.markdown for p in resp.pages if p.markdown]
    return "\n\n---PAGE---\n\n".join(pages)


async def extract_structure(raw_text: str, title: str) -> dict:
    """Ask Mistral to identify chapter/section boundaries from raw text.

    Returns a dict with keys: chapters (list of {title, start_char, end_char}).
    Only called when the document has no native ToC (e.g. plain TXT files).
    """
    client = get_mistral_client()
    prompt = (
        f'You are a book structure analyser. Given the first 8000 characters of "{title}", '
        "identify chapter headings and their approximate character offsets. "
        "Respond with JSON only: {\"chapters\": [{\"title\": str, \"start_char\": int}]}"
    )
    resp = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt + "\n\n" + raw_text[:8000]},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


async def chat_text(prompt: str, system: str | None = None) -> str:
    """Single-turn chat completion returning plain text."""
    client = get_mistral_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.complete_async(
        model=settings.MISTRAL_CHAT_MODEL,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


async def chat_json(prompt: str, system: str | None = None) -> dict:
    """Single-turn chat completion constrained to a JSON object response."""
    client = get_mistral_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.complete_async(
        model=settings.MISTRAL_CHAT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)
