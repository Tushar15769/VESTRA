"""YouTube URL parsing, validation, and timestamp utility functions."""

import re
from typing import Optional
from urllib.parse import parse_qs, urlparse


# Standard YouTube 11-character video ID regex pattern
VIDEO_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Comprehensive patterns for extracting YouTube video IDs from various URL formats
YOUTUBE_URL_PATTERNS = [
    # Standard watch URL: https://www.youtube.com/watch?v=VIDEO_ID
    re.compile(r"(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/watch\?(?:.*&)?v=([A-Za-z0-9_-]{11})"),
    # Short URL: https://youtu.be/VIDEO_ID
    re.compile(r"(?:https?:\/\/)?youtu\.be\/([A-Za-z0-9_-]{11})"),
    # Embed URL: https://www.youtube.com/embed/VIDEO_ID
    re.compile(r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([A-Za-z0-9_-]{11})"),
    # Shorts URL: https://www.youtube.com/shorts/VIDEO_ID
    re.compile(r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([A-Za-z0-9_-]{11})"),
    # Direct /v/ URL: https://www.youtube.com/v/VIDEO_ID
    re.compile(r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([A-Za-z0-9_-]{11})"),
    # Live URL: https://www.youtube.com/live/VIDEO_ID
    re.compile(r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/live\/([A-Za-z0-9_-]{11})"),
]


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract and validate the 11-character YouTube video ID from a URL or raw ID.

    Args:
        url_or_id: YouTube URL (e.g. https://www.youtube.com/watch?v=...) or direct 11-char ID.

    Returns:
        11-character string video ID, or None if invalid.
    """
    if not url_or_id or not isinstance(url_or_id, str):
        return None

    cleaned = url_or_id.strip()

    # Check if input is directly a valid 11-character video ID
    if VIDEO_ID_REGEX.match(cleaned):
        return cleaned

    # Check against regex patterns
    for pattern in YOUTUBE_URL_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            candidate = match.group(1)
            if VIDEO_ID_REGEX.match(candidate):
                return candidate

    # Fallback: parse via urllib urlparse
    try:
        parsed = urlparse(cleaned)
        if "youtube.com" in parsed.netloc:
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"] and VIDEO_ID_REGEX.match(qs["v"][0]):
                return qs["v"][0]
        elif "youtu.be" in parsed.netloc:
            path_part = parsed.path.strip("/").split("/")[0]
            if VIDEO_ID_REGEX.match(path_part):
                return path_part
    except Exception:
        pass

    return None


def is_valid_youtube_url(url_or_id: str) -> bool:
    """Validate whether the given string represents a valid YouTube URL or ID."""
    return extract_video_id(url_or_id) is not None


def format_timestamp(seconds: float) -> str:
    """Convert a timestamp in seconds to formatted MM:SS or HH:MM:SS string.

    Args:
        seconds: Float or int duration in seconds.

    Returns:
        Formatted string, e.g., '03:45' or '01:12:30'.
    """
    if seconds < 0:
        seconds = 0
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_timestamp_url(video_id: str, start_seconds: float) -> str:
    """Generate a direct YouTube link with start timestamp query parameter.

    Args:
        video_id: 11-character video ID.
        start_seconds: Start offset in seconds.

    Returns:
        Full clickable YouTube URL (e.g., https://www.youtube.com/watch?v=...&t=123s).
    """
    sec = max(0, int(start_seconds))
    return f"https://www.youtube.com/watch?v={video_id}&t={sec}s"
