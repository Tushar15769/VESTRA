"""Robust transcript extraction using YouTube Transcript API with yt-dlp fallback."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from config.settings import get_settings
from utils.logging_utils import get_logger
from utils.youtube_utils import extract_video_id

logger = get_logger("transcript_loader")


class TranscriptError(Exception):
    """Base exception for transcript retrieval failures."""
    pass


class TranscriptNotFoundError(TranscriptError):
    """Raised when no transcript (manual or auto-generated) is available."""
    pass


class VideoUnavailableError(TranscriptError):
    """Raised when the video is private, deleted, or region/age-restricted."""
    pass


class TranscriptsDisabledError(TranscriptError):
    """Raised when subtitles/transcripts are explicitly disabled for the video."""
    pass


class YouTubeAPIError(TranscriptError):
    """Raised for network or external API errors during extraction."""
    pass


class TranscriptLoader:
    """Manages transcript fetching with multi-tier fallbacks and local disk caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.transcripts_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_transcript(self, url_or_id: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch transcript segments with caching and automatic fallbacks.

        Args:
            url_or_id: YouTube URL or 11-char video ID.
            force_refresh: If True, bypasses local disk cache.

        Returns:
            List of raw segment dicts: [{'text': str, 'start': float, 'duration': float}]

        Raises:
            TranscriptNotFoundError, VideoUnavailableError, TranscriptsDisabledError, YouTubeAPIError
        """
        video_id = extract_video_id(url_or_id)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or ID: '{url_or_id}'")

        cache_file = self.cache_dir / f"{video_id}.json"

        # Check local cache first
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, list) and len(cached_data) > 0:
                    logger.info(f"Loaded cached transcript for video {video_id} ({len(cached_data)} segments)")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to read cached transcript for {video_id}: {e}")

        # Attempt 1: youtube-transcript-api
        segments = None
        try:
            segments = self._fetch_via_transcript_api(video_id)
        except (VideoUnavailableError, TranscriptsDisabledError) as e:
            # Re-raise known user-facing errors
            raise e
        except Exception as e:
            logger.warning(f"youtube-transcript-api failed for {video_id}: {e}. Trying yt-dlp fallback.")

        # Attempt 2: yt-dlp fallback
        if not segments:
            try:
                segments = self._fetch_via_ytdlp(video_id)
            except Exception as e:
                logger.error(f"yt-dlp subtitle extraction also failed for {video_id}: {e}")

        if not segments:
            raise TranscriptNotFoundError(
                f"No transcript or subtitles could be found for video '{video_id}'. "
                "The video might not have captions enabled, or is restricted."
            )

        # Cache valid transcript
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved transcript cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not write cache file {cache_file}: {e}")

        return segments

    def _fetch_via_transcript_api(self, video_id: str) -> List[Dict[str, Any]]:
        """Fetch transcript via youtube-transcript-api library supporting both v1.2+ and legacy versions."""
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )

        def _to_dicts(fetched_data) -> List[Dict[str, Any]]:
            results = []
            for item in fetched_data:
                if isinstance(item, dict):
                    results.append({
                        "text": item.get("text", ""),
                        "start": float(item.get("start", 0.0)),
                        "duration": float(item.get("duration", 0.0)),
                    })
                else:
                    results.append({
                        "text": getattr(item, "text", ""),
                        "start": float(getattr(item, "start", 0.0)),
                        "duration": float(getattr(item, "duration", 0.0)),
                    })
            return results

        try:
            # Check if YouTubeTranscriptApi is instance-based (v1.2+) or classmethod-based (legacy)
            api_instance = YouTubeTranscriptApi() if hasattr(YouTubeTranscriptApi, "list") and callable(getattr(YouTubeTranscriptApi, "list")) else None

            transcript_list = None
            if api_instance and hasattr(api_instance, "list"):
                transcript_list = api_instance.list(video_id)
            elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            if transcript_list:
                # 1. Try finding English transcript
                for finder_method in ["find_transcript", "find_manually_created_transcript", "find_generated_transcript"]:
                    if hasattr(transcript_list, finder_method):
                        try:
                            t = getattr(transcript_list, finder_method)(["en", "en-US", "en-GB"])
                            logger.info(f"Found transcript ({finder_method}) for {video_id}")
                            return _to_dicts(t.fetch())
                        except Exception:
                            pass

                # 2. Try translating any translatable transcript to English
                for tr in transcript_list:
                    if getattr(tr, "is_translatable", False):
                        try:
                            logger.info(f"Translating transcript ({tr.language_code}) to English for {video_id}")
                            return _to_dicts(tr.translate("en").fetch())
                        except Exception:
                            pass

                # 3. Use first available transcript
                for tr in transcript_list:
                    logger.info(f"Using available transcript ({tr.language_code}) for {video_id}")
                    return _to_dicts(tr.fetch())

            # Direct fetch fallback
            if api_instance and hasattr(api_instance, "fetch"):
                return _to_dicts(api_instance.fetch(video_id))
            elif hasattr(YouTubeTranscriptApi, "get_transcript"):
                return _to_dicts(YouTubeTranscriptApi.get_transcript(video_id))

        except TranscriptsDisabled:
            raise TranscriptsDisabledError("Subtitles/transcripts are disabled for this video.")
        except VideoUnavailable:
            raise VideoUnavailableError("This video is unavailable, private, or restricted.")
        except NoTranscriptFound:
            raise TranscriptNotFoundError(f"No transcripts found for video {video_id}")
        except Exception as e:
            logger.warning(f"Error communicating with YouTube Transcript API: {e}")
            raise YouTubeAPIError(f"Error communicating with YouTube Transcript API: {e}")

        raise TranscriptNotFoundError("No transcript tracks available.")

    def _fetch_via_ytdlp(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Fallback: Extract subtitle URLs from yt-dlp metadata and parse JSON captions."""
        import yt_dlp

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        if not info:
            return None

        # Look in subtitles first, then automatic_captions
        captions_dict = info.get("subtitles") or {}
        if not captions_dict:
            captions_dict = info.get("automatic_captions") or {}

        if not captions_dict:
            return None

        # Find English or first available language track
        target_lang = None
        for lang in ["en", "en-US", "en-GB", "en-orig"]:
            if lang in captions_dict:
                target_lang = lang
                break

        if not target_lang and captions_dict:
            target_lang = next(iter(captions_dict))

        if not target_lang:
            return None

        formats = captions_dict[target_lang]
        # Prefer json3 format for easy parsing
        json3_url = None
        vtt_url = None
        for fmt in formats:
            ext = fmt.get("ext")
            if ext == "json3":
                json3_url = fmt.get("url")
            elif ext == "vtt":
                vtt_url = fmt.get("url")

        if json3_url:
            resp = requests.get(json3_url, timeout=10)
            if resp.ok:
                return self._parse_json3_subtitles(resp.json())

        if vtt_url:
            resp = requests.get(vtt_url, timeout=10)
            if resp.ok:
                return self._parse_vtt_subtitles(resp.text)

        return None

    def _parse_json3_subtitles(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse YouTube json3 caption format into standard segments."""
        events = data.get("events", [])
        segments: List[Dict[str, Any]] = []

        for event in events:
            segs = event.get("segs")
            if not segs:
                continue

            text = "".join(s.get("utf8", "") for s in segs).strip()
            if not text or text == "\n":
                continue

            start_ms = event.get("tStartMs", 0)
            duration_ms = event.get("dDurationMs", 0)

            segments.append({
                "text": text,
                "start": round(start_ms / 1000.0, 2),
                "duration": round(duration_ms / 1000.0, 2),
            })

        return segments

    def _parse_vtt_subtitles(self, vtt_text: str) -> List[Dict[str, Any]]:
        """Rudimentary WebVTT parser to segment list."""
        lines = vtt_text.splitlines()
        segments: List[Dict[str, Any]] = []
        import re

        timestamp_re = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})")

        def _vtt_to_sec(ts_str: str) -> float:
            parts = ts_str.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return 0.0

        current_start = 0.0
        current_dur = 0.0
        current_text = []

        for line in lines:
            line = line.strip()
            ts_match = timestamp_re.search(line)
            if ts_match:
                if current_text:
                    segments.append({
                        "text": " ".join(current_text),
                        "start": current_start,
                        "duration": current_dur,
                    })
                    current_text = []
                s_str, e_str = ts_match.group(1), ts_match.group(2)
                current_start = round(_vtt_to_sec(s_str), 2)
                current_dur = round(_vtt_to_sec(e_str) - current_start, 2)
            elif line and not line.startswith("WEBVTT") and not line.isdigit():
                current_text.append(line)

        if current_text:
            segments.append({
                "text": " ".join(current_text),
                "start": current_start,
                "duration": current_dur,
            })

        return segments
