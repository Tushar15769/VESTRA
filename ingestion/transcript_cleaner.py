"""Transcript cleaning and normalization utilities."""

import html
import re
from typing import Any, Dict, List


# Regex patterns for subtitle noise and artifacts
SUBTITLE_ARTIFACT_PATTERNS = [
    re.compile(r"\[(?:music|applause|laughter|cheers|gasp|sigh|groan|screaming|silence|inaudible)\]", re.IGNORECASE),
    re.compile(r"\((?:music|applause|laughter|cheers|gasp|sigh|groan|screaming|silence|inaudible)\)", re.IGNORECASE),
    re.compile(r"\[.*?\]"),  # General brackets like [Sound]
    re.compile(r"♪+"),       # Musical note symbols
    re.compile(r"^>>\s*"),   # Speaker change markers like >>
]

WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(raw_text: str) -> str:
    """Clean and normalize a single text snippet.

    - Unescapes HTML entities (&amp;, &#39;, etc.)
    - Removes subtitle noise/artifacts ([Music], [Applause], ♪)
    - Replaces line breaks and multiple spaces with a single space
    - Strips leading/trailing whitespace
    """
    if not raw_text:
        return ""

    # Unescape HTML entities
    text = html.unescape(raw_text)

    # Remove subtitle noise
    for pattern in SUBTITLE_ARTIFACT_PATTERNS:
        text = pattern.sub(" ", text)

    # Normalize whitespace and line breaks
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def clean_transcript_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean a sequence of raw transcript segments while preserving timestamps.

    Args:
        segments: List of dicts with 'text', 'start', and optionally 'duration'.

    Returns:
        List of cleaned dicts with 'text', 'start', 'duration'.
        Empty and artifact-only segments are pruned. Consecutive duplicates are merged.
    """
    cleaned_segments: List[Dict[str, Any]] = []
    prev_text = ""

    for seg in segments:
        text = seg.get("text", "")
        cleaned = clean_text(text)

        # Skip empty segments or noise-only text
        if not cleaned:
            continue

        # Skip exact duplicate consecutive segments common in auto-generated captions
        if cleaned.lower() == prev_text.lower():
            continue

        start = float(seg.get("start", 0.0))
        duration = float(seg.get("duration", 0.0))

        cleaned_segments.append({
            "text": cleaned,
            "start": round(start, 2),
            "duration": round(duration, 2),
        })
        prev_text = cleaned

    return cleaned_segments
