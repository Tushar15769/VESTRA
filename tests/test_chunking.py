"""Tests for transcript cleaning and intelligent chunking."""

from ingestion.chunker import chunk_transcript
from ingestion.transcript_cleaner import clean_text, clean_transcript_segments


class TestTranscriptCleaning:
    """Test suite covering HTML cleaning and subtitle artifact removal."""

    def test_clean_text_artifacts(self):
        dirty = "Hello [Music] world! (applause) Welcome &amp; enjoy &#39;AI&#39; ♪"
        cleaned = clean_text(dirty)
        assert "[Music]" not in cleaned
        assert "(applause)" not in cleaned
        assert "♪" not in cleaned
        assert "&amp;" not in cleaned
        assert "&#39;" not in cleaned
        assert cleaned == "Hello world! Welcome & enjoy 'AI'"

    def test_clean_text_whitespace(self):
        dirty = "  This   has \n\n extra   spaces \t and tabs.   "
        assert clean_text(dirty) == "This has extra spaces and tabs."

    def test_clean_transcript_segments(self):
        raw_segments = [
            {"text": "[Music]", "start": 0.0, "duration": 2.5},
            {"text": "Welcome to this lecture.", "start": 2.5, "duration": 3.0},
            {"text": "Welcome to this lecture.", "start": 5.5, "duration": 1.0},  # duplicate
            {"text": "Today we discuss deep learning &amp; RAG.", "start": 6.5, "duration": 4.0},
            {"text": "", "start": 10.5, "duration": 1.0},  # empty
        ]

        cleaned = clean_transcript_segments(raw_segments)
        assert len(cleaned) == 2
        assert cleaned[0]["text"] == "Welcome to this lecture."
        assert cleaned[0]["start"] == 2.5
        assert cleaned[1]["text"] == "Today we discuss deep learning & RAG."


class TestTranscriptChunking:
    """Test suite covering timestamp-preserving chunking."""

    def test_chunk_transcript_basic(self):
        segments = [
            {"text": "This is segment one about neural networks.", "start": 0.0, "duration": 5.0},
            {"text": "Segment two continues discussing gradients and backpropagation.", "start": 5.0, "duration": 6.0},
            {"text": "Segment three dives into transformer attention mechanisms.", "start": 11.0, "duration": 7.0},
            {"text": "Segment four concludes the architecture discussion.", "start": 18.0, "duration": 4.0},
        ]

        chunks = chunk_transcript(
            segments=segments,
            video_id="test_video_1",
            chunk_size=120,
            chunk_overlap=30,
        )

        assert len(chunks) >= 2
        # Check first chunk metadata
        c1 = chunks[0]
        assert c1.chunk_id == 0
        assert c1.video_id == "test_video_1"
        assert c1.start_timestamp == 0.0
        assert c1.start_time_formatted == "00:00"
        assert "neural networks" in c1.text
        assert "https://www.youtube.com/watch?v=test_video_1&t=0s" == c1.timestamp_url

        # Check end timestamp
        assert c1.end_timestamp > c1.start_timestamp

    def test_chunk_empty_segments(self):
        assert chunk_transcript([], video_id="empty_vid") == []
