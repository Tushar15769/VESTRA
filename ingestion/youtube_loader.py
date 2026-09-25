"""YouTube video metadata extraction with multiple fallback strategies."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import requests
from utils.logging_utils import get_logger
from utils.youtube_utils import extract_video_id, format_timestamp

logger = get_logger("youtube_loader")


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video."""

    video_id: str
    video_url: str
    title: str
    channel: str
    duration_seconds: int
    duration_formatted: str
    thumbnail_url: str
    description: str = ""
    view_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


class YouTubeMetadataLoader:
    """Extracts video metadata using yt-dlp with oEmbed fallback."""

    def __init__(self, request_timeout: int = 10):
        self.request_timeout = request_timeout

    def get_metadata(self, url_or_id: str) -> VideoMetadata:
        """Fetch video metadata reliably.

        Tries yt-dlp first for comprehensive info, falling back to YouTube oEmbed.
        """
        video_id = extract_video_id(url_or_id)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or ID: '{url_or_id}'")

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # 1. Try yt-dlp extraction
        try:
            return self._extract_with_ytdlp(video_id, video_url)
        except Exception as e:
            logger.warning(f"yt-dlp metadata extraction failed for {video_id}: {e}. Trying oEmbed fallback.")

        # 2. Fallback to oEmbed
        try:
            return self._extract_with_oembed(video_id, video_url)
        except Exception as e:
            logger.warning(f"oEmbed metadata extraction failed for {video_id}: {e}.")

        # 3. Final default placeholder metadata if network fails
        return VideoMetadata(
            video_id=video_id,
            video_url=video_url,
            title=f"YouTube Video ({video_id})",
            channel="Unknown Channel",
            duration_seconds=0,
            duration_formatted="N/A",
            thumbnail_url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            description="",
        )

    def _extract_with_ytdlp(self, video_id: str, video_url: str) -> VideoMetadata:
        """Extract metadata using yt-dlp without downloading video."""
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        if not info:
            raise RuntimeError(f"Could not extract info for video {video_id}")

        title = info.get("title", f"YouTube Video ({video_id})")
        channel = info.get("uploader") or info.get("channel") or "Unknown Channel"
        duration_sec = int(info.get("duration") or 0)
        formatted_duration = format_timestamp(duration_sec) if duration_sec > 0 else "N/A"

        # Find best thumbnail
        thumbnail = info.get("thumbnail") or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        description = info.get("description", "") or ""
        view_count = info.get("view_count")

        logger.info(f"Retrieved yt-dlp metadata for '{title}' by '{channel}'")
        return VideoMetadata(
            video_id=video_id,
            video_url=video_url,
            title=title,
            channel=channel,
            duration_seconds=duration_sec,
            duration_formatted=formatted_duration,
            thumbnail_url=thumbnail,
            description=description[:500],
            view_count=view_count,
        )

    def _extract_with_oembed(self, video_id: str, video_url: str) -> VideoMetadata:
        """Extract lightweight metadata from YouTube oEmbed API."""
        oembed_url = f"https://www.youtube.com/oembed?url={video_url}&format=json"
        response = requests.get(oembed_url, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()

        title = data.get("title", f"YouTube Video ({video_id})")
        channel = data.get("author_name", "Unknown Channel")
        thumbnail = data.get("thumbnail_url", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")

        logger.info(f"Retrieved oEmbed metadata for '{title}' by '{channel}'")
        return VideoMetadata(
            video_id=video_id,
            video_url=video_url,
            title=title,
            channel=channel,
            duration_seconds=0,
            duration_formatted="N/A",
            thumbnail_url=thumbnail,
            description="",
        )
