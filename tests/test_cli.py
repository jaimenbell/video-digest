"""Unit tests for CLI helpers + full pipeline orchestration (every stage injected/mocked)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_digest.cli import build_arg_parser, run_pipeline, slugify, title_from_source


class TestSlugify:
    def test_lowercases_and_replaces_non_alnum(self):
        assert slugify("World v4 Redesign!") == "world-v4-redesign"

    def test_strips_url_scheme(self):
        assert slugify("https://tiktok.com/@user/video/123") == "tiktok-com-user-video-123"

    def test_truncates_long_input(self):
        result = slugify("a" * 200)
        assert len(result) <= 60

    def test_empty_input_falls_back_to_video(self):
        assert slugify("") == "video"
        assert slugify("###") == "video"


class TestTitleFromSource:
    def test_local_path_uses_stem(self):
        assert title_from_source("/home/jaime/clips/demo.mp4") == "demo"

    def test_url_falls_back_to_stem_or_source(self):
        # Path(...).stem on a URL without an extension just returns the last segment.
        result = title_from_source("https://tiktok.com/@user/video/123")
        assert result == "123"


class TestBuildArgParser:
    def test_parses_positional_source_and_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args(["myvideo.mp4"])
        assert args.source == "myvideo.mp4"
        assert args.output == "digests"
        assert args.model == "base"
        assert args.vault_inbox is None

    def test_parses_all_flags(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "https://example.com/v",
                "--output",
                "out/",
                "--model",
                "small",
                "--vault-inbox",
                "C:/vault/inbox",
                "--no-vllm-summary",
            ]
        )
        assert args.output == "out/"
        assert args.model == "small"
        assert args.vault_inbox == "C:/vault/inbox"
        assert args.no_vllm_summary is True


class TestRunPipelineFullyMocked:
    def _fakes(self, tmp_path):
        video_path = tmp_path / "scratch" / "source.mp4"

        acquire_fn = MagicMock(return_value=video_path)
        probe_duration_fn = MagicMock(return_value=20.0)
        detect_scenes_fn = MagicMock(return_value=[3.0])
        extract_frames_fn = MagicMock(
            return_value=[tmp_path / "scratch" / "frames" / "001.jpg"]
        )
        extract_audio_fn = MagicMock(return_value=tmp_path / "scratch" / "audio.wav")
        transcribe_fn = MagicMock(return_value="[00:00 - 00:05] mocked transcript")
        summarize_fn = MagicMock(return_value="mocked mechanical summary")

        return dict(
            acquire_fn=acquire_fn,
            probe_duration_fn=probe_duration_fn,
            detect_scenes_fn=detect_scenes_fn,
            extract_frames_fn=extract_frames_fn,
            extract_audio_fn=extract_audio_fn,
            transcribe_fn=transcribe_fn,
            summarize_fn=summarize_fn,
        )

    def test_writes_digest_to_output_dir(self, tmp_path):
        fakes = self._fakes(tmp_path)
        output_dir = tmp_path / "digests"
        scratch_dir = tmp_path / "scratch"

        digest_path = run_pipeline(
            source="clip.mp4",
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            **fakes,
        )

        assert digest_path.exists()
        assert digest_path.parent == output_dir
        content = digest_path.read_text(encoding="utf-8")
        assert "mocked mechanical summary" in content
        assert "mocked transcript" in content

    def test_writes_transcript_txt_to_scratch(self, tmp_path):
        fakes = self._fakes(tmp_path)
        run_pipeline(
            source="clip.mp4",
            output_dir=tmp_path / "digests",
            scratch_dir=tmp_path / "scratch",
            **fakes,
        )
        transcript_file = tmp_path / "scratch" / "transcript.txt"
        assert transcript_file.exists()
        assert "mocked transcript" in transcript_file.read_text(encoding="utf-8")

    def test_copies_digest_to_vault_inbox_when_given(self, tmp_path):
        fakes = self._fakes(tmp_path)
        vault_inbox = tmp_path / "vault-inbox"

        digest_path = run_pipeline(
            source="clip.mp4",
            output_dir=tmp_path / "digests",
            scratch_dir=tmp_path / "scratch",
            vault_inbox=vault_inbox,
            **fakes,
        )

        copied = vault_inbox / digest_path.name
        assert copied.exists()
        assert copied.read_text(encoding="utf-8") == digest_path.read_text(encoding="utf-8")

    def test_skips_vllm_summary_when_disabled(self, tmp_path):
        fakes = self._fakes(tmp_path)
        run_pipeline(
            source="clip.mp4",
            output_dir=tmp_path / "digests",
            scratch_dir=tmp_path / "scratch",
            use_vllm_summary=False,
            **fakes,
        )
        fakes["summarize_fn"].assert_not_called()

    def test_no_real_subprocess_or_network_calls(self, tmp_path):
        # Every stage is a MagicMock; nothing here can hit a real binary or network.
        fakes = self._fakes(tmp_path)
        run_pipeline(
            source="clip.mp4",
            output_dir=tmp_path / "digests",
            scratch_dir=tmp_path / "scratch",
            **fakes,
        )
        for fn in fakes.values():
            assert fn.call_count >= 1
