"""Unit tests for transcript formatting (pure) and transcribe_audio (mocked model)."""

from pathlib import Path
from unittest.mock import MagicMock

from video_digest.transcript import format_transcript, transcribe_audio


class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class TestFormatTranscript:
    def test_formats_object_segments_with_timestamps(self):
        segments = [FakeSegment(0.0, 4.2, "Hello world"), FakeSegment(4.2, 9.8, "second segment")]
        text = format_transcript(segments)
        assert "[00:00 - 00:04] Hello world" in text
        assert "[00:04 - 00:10] second segment" in text

    def test_formats_dict_segments(self):
        segments = [{"start": 0.0, "end": 2.0, "text": "  spaced  "}]
        text = format_transcript(segments)
        assert text == "[00:00 - 00:02] spaced"

    def test_formats_hour_scale_timestamps(self):
        segments = [FakeSegment(3661.0, 3665.0, "an hour in")]
        text = format_transcript(segments)
        assert "[01:01:01 - 01:01:05]" in text

    def test_empty_segments_returns_placeholder(self):
        assert format_transcript([]) == "(no speech detected)"


class TestTranscribeAudio:
    def test_uses_injected_model_factory_no_real_model_load(self, tmp_path):
        fake_model = MagicMock()
        fake_model.transcribe.return_value = (
            [FakeSegment(0.0, 1.5, "mocked transcript")],
            MagicMock(),
        )
        fake_factory = MagicMock(return_value=fake_model)

        audio_path = tmp_path / "audio.wav"
        result = transcribe_audio(audio_path, model_size="base", model_factory=fake_factory)

        fake_factory.assert_called_once_with("base", device="cpu", compute_type="int8")
        fake_model.transcribe.assert_called_once_with(str(audio_path))
        assert "mocked transcript" in result
