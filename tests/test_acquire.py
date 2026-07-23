"""Unit tests for acquire (URL detection is pure; yt-dlp invocation is mocked)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_digest.acquire import acquire_video, is_url


class TestIsUrl:
    @pytest.mark.parametrize(
        "source",
        [
            "https://www.tiktok.com/@user/video/123",
            "http://example.com/v.mp4",
            "https://youtu.be/abc123",
        ],
    )
    def test_recognizes_http_and_https(self, source):
        assert is_url(source) is True

    @pytest.mark.parametrize(
        "source",
        [
            "C:\\Users\\jaime\\Downloads\\video.mp4",
            "/home/jaime/video.mp4",
            "video.mp4",
            "./scratch/source.mp4",
        ],
    )
    def test_rejects_local_paths(self, source):
        assert is_url(source) is False


class TestAcquireVideo:
    def test_local_path_passed_through_unchanged(self, tmp_path):
        local_file = tmp_path / "clip.mp4"
        local_file.write_bytes(b"fake video bytes")
        fake_runner = MagicMock()

        result = acquire_video(str(local_file), tmp_path / "scratch", runner=fake_runner)

        assert result == local_file
        fake_runner.assert_not_called()

    def test_local_path_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            acquire_video(str(tmp_path / "does-not-exist.mp4"), tmp_path / "scratch")

    def test_url_invokes_yt_dlp_via_injected_runner(self, tmp_path):
        scratch_dir = tmp_path / "scratch"

        def fake_runner(cmd, **kwargs):
            # Simulate yt-dlp having written the file, no real network/binary call.
            scratch_dir.mkdir(parents=True, exist_ok=True)
            (scratch_dir / "source.mp4").write_bytes(b"downloaded bytes")
            return MagicMock(returncode=0, stdout="", stderr="")

        result = acquire_video("https://tiktok.com/@u/video/1", scratch_dir, runner=fake_runner)

        assert result == scratch_dir / "source.mp4"
        assert result.exists()

    def test_url_command_includes_yt_dlp_and_source(self, tmp_path):
        scratch_dir = tmp_path / "scratch"
        captured_cmd = {}

        def fake_runner(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            scratch_dir.mkdir(parents=True, exist_ok=True)
            (scratch_dir / "source.mp4").write_bytes(b"x")
            return MagicMock(returncode=0)

        acquire_video("https://example.com/v/1", scratch_dir, runner=fake_runner)

        assert captured_cmd["cmd"][0] == "yt-dlp"
        assert "https://example.com/v/1" in captured_cmd["cmd"]

    def test_no_downloaded_file_raises_runtime_error(self, tmp_path):
        scratch_dir = tmp_path / "scratch"
        fake_runner = MagicMock(return_value=MagicMock(returncode=0))
        with pytest.raises(RuntimeError):
            acquire_video("https://example.com/v/1", scratch_dir, runner=fake_runner)
