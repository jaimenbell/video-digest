"""Unit tests for audio extraction (ffmpeg invocation mocked)."""

from pathlib import Path
from unittest.mock import MagicMock

from video_digest.audio import CHANNELS, SAMPLE_RATE_HZ, extract_audio


class TestExtractAudio:
    def test_invokes_ffmpeg_with_mono_16khz_wav_args(self, tmp_path):
        fake_runner = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        video_path = Path("video.mp4")
        output_path = tmp_path / "scratch" / "audio.wav"

        result = extract_audio(video_path, output_path, runner=fake_runner)

        assert result == output_path
        fake_runner.assert_called_once()
        cmd = fake_runner.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert str(video_path) in cmd
        assert str(output_path) in cmd
        assert str(SAMPLE_RATE_HZ) in cmd
        assert str(CHANNELS) in cmd
        assert "-vn" in cmd  # no video stream

    def test_creates_parent_directory(self, tmp_path):
        fake_runner = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        output_path = tmp_path / "nested" / "dir" / "audio.wav"

        extract_audio(Path("video.mp4"), output_path, runner=fake_runner)

        assert output_path.parent.exists()

    def test_no_real_subprocess_invoked(self, tmp_path):
        fake_runner = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        extract_audio(Path("video.mp4"), tmp_path / "audio.wav", runner=fake_runner)
        fake_runner.assert_called_once()
