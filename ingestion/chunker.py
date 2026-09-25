"""Intelligent timestamp-preserving transcript chunker."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List
from utils.youtube_utils import build_timestamp_url, format_timestamp


@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript with attached video and timestamp metadata."""

    chunk_id: int
    video_id: str
    video_url: str
    text: str
    start_timestamp: float
    end_timestamp: float
    start_time_formatted: str
    end_time_formatted: str
    timestamp_url: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to standard dictionary."""
        return asdict(self)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata dict suitable for vectorstore storage."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "video_url": self.video_url,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "start_time_formatted": self.start_time_formatted,
            "end_time_formatted": self.end_time_formatted,
            "timestamp_url": self.timestamp_url,
        }


def chunk_transcript(
    segments: List[Dict[str, Any]],
    video_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[TranscriptChunk]:
    """Chunk cleaned transcript segments into overlapping context windows with accurate timestamps.

    Args:
        segments: Cleaned transcript segments with 'text', 'start', and 'duration'.
        video_id: 11-character YouTube video ID.
        chunk_size: Target character length per chunk (default: 1000).
        chunk_overlap: Overlapping character length between consecutive chunks (default: 150).

    Returns:
        List of TranscriptChunk objects containing text and timestamp metadata.
    """
    if not segments:
        return []

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    chunks: List[TranscriptChunk] = []

    current_segments: List[Dict[str, Any]] = []
    current_length = 0
    chunk_id = 0

    idx = 0
    num_segments = len(segments)

    while idx < num_segments:
        seg = segments[idx]
        seg_text = seg["text"].strip()
        if not seg_text:
            idx += 1
            continue

        seg_len = len(seg_text) + 1  # +1 for separating space

        # If adding this segment exceeds chunk_size and we already have content
        if current_length + seg_len > chunk_size and current_segments:
            chunk_obj = _create_chunk(current_segments, chunk_id, video_id, video_url)
            chunks.append(chunk_obj)
            chunk_id += 1

            # Prepare overlap segments for the next window
            overlap_segments = []
            overlap_len = 0
            # Walk backwards from the end of current_segments to gather overlap
            for prev_seg in reversed(current_segments):
                p_len = len(prev_seg["text"]) + 1
                if overlap_len + p_len <= chunk_overlap:
                    overlap_segments.insert(0, prev_seg)
                    overlap_len += p_len
                else:
                    break

            current_segments = overlap_segments
            current_length = overlap_len

        current_segments.append(seg)
        current_length += seg_len
        idx += 1

    # Final remaining chunk
    if current_segments:
        chunk_obj = _create_chunk(current_segments, chunk_id, video_id, video_url)
        chunks.append(chunk_obj)

    return chunks


def _create_chunk(
    segments: List[Dict[str, Any]],
    chunk_id: int,
    video_id: str,
    video_url: str,
) -> TranscriptChunk:
    """Helper to assemble a TranscriptChunk from a list of segments."""
    text_content = " ".join(s["text"].strip() for s in segments)
    start_ts = segments[0]["start"]
    last_seg = segments[-1]
    end_ts = round(last_seg["start"] + last_seg.get("duration", 0.0), 2)

    start_fmt = format_timestamp(start_ts)
    end_fmt = format_timestamp(end_ts)
    ts_url = build_timestamp_url(video_id, start_ts)

    return TranscriptChunk(
        chunk_id=chunk_id,
        video_id=video_id,
        video_url=video_url,
        text=text_content,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        start_time_formatted=start_fmt,
        end_time_formatted=end_fmt,
        timestamp_url=ts_url,
    )
