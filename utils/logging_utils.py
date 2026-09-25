"""Logging utilities for the YouTube RAG Chatbot."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "youtube_rag", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
