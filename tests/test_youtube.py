"""Tests for YouTube URL parsing, validation, and timestamp utilities."""

import pytest
from utils.youtube_utils import (
    build_timestamp_url,
    extract_video_id,
    format_timestamp,
    is_valid_youtube_url,
)


class TestYouTubeUtils:
    """Test suite covering URL parsing, video ID extraction, and timestamps."""

    @pytest.mark.parametrize(
        "url,expected_id",
        [
            ("https://www.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("http://www.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("https://youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("https://youtu.be/aircAruvnKk", "aircAruvnKk"),
            ("https://www.youtube.com/embed/aircAruvnKk", "aircAruvnKk"),
            ("https://www.youtube.com/v/aircAruvnKk", "aircAruvnKk"),
            ("https://www.youtube.com/shorts/aircAruvnKk", "aircAruvnKk"),
            ("https://m.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("https://www.youtube.com/watch?v=aircAruvnKk&t=120s&ab_channel=3Blue1Brown", "aircAruvnKk"),
            ("https://youtu.be/aircAruvnKk?feature=shared", "aircAruvnKk"),
            ("aircAruvnKk", "aircAruvnKk"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ],
    )
    def test_extract_video_id_valid(self, url: str, expected_id: str):
        assert extract_video_id(url) == expected_id
        assert is_valid_youtube_url(url) is True

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "",
            "   ",
            "not_a_url",
            "https://vimeo.com/12345678",
            "https://www.youtube.com/watch?v=short",
            "https://www.google.com",
            "12345",
            None,
        ],
    )
    def test_extract_video_id_invalid(self, invalid_input):
        assert extract_video_id(invalid_input) is None
        assert is_valid_youtube_url(invalid_input) is False

    @pytest.mark.parametrize(
        "seconds,expected_fmt",
        [
            (0, "00:00"),
            (45.2, "00:45"),
            (75, "01:15"),
            (599, "09:59"),
            (3600, "01:00:00"),
            (3665.8, "01:01:06"),
            (7325, "02:02:05"),
        ],
    )
    def test_format_timestamp(self, seconds: float, expected_fmt: str):
        assert format_timestamp(seconds) == expected_fmt

    def test_build_timestamp_url(self):
        url = build_timestamp_url("aircAruvnKk", 125.7)
        assert url == "https://www.youtube.com/watch?v=aircAruvnKk&t=125s"
