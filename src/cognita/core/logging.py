import logging
import sys
from typing import TextIO


def setup_logging(log_level: str = "INFO", stream: TextIO | None = None) -> None:
    """Configure logging to a single stream.

    Defaults to stderr: in stdio mode the MCP protocol owns stdout, and anything
    else written there corrupts the message framing.
    """
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(stream or sys.stderr)],
        force=True,
    )
    # Chatty at INFO, and none of it is about the library.
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
