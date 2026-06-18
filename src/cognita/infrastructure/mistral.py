"""Mistral AI integration for document parsing and OCR.

Used during ingestion to extract clean text from scanned PDFs and complex layouts.
For well-formed digital PDFs, pypdf is used directly (cheaper, faster).
"""

import base64
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
