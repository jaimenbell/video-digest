"""Unit tests for keyframe timestamp math + ffmpeg-invoking functions (mocked subprocess)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_digest.keyframes import (
    detect_scene_changes,
    extract_frames,
    probe_duration,
    select_keyframe_timestamps,
)


class TestSelectKeyframeTimestamps:
    def test_includes_zero_and_floor_grid_with_no_scene_changes(self):
        timestamps = select_keyframe_timestamps(duration=20.0, scene_timestamps=[], floor_interval=5.0)
        assert timestamps == [0.0, 5.0, 10.0, 15.0]

    def test_merges_scene_changes_into_floor_grid(self):
        timestamps = select_keyframe_timestamps(
            duration=20.0, scene_timestamps=[2.5, 7.1], floor_interval=5.0
        )
        assert timestamps == [0.0, 2.5, 5.0, 7.1, 10.0, 15.0]

    def test_dedupes_scene_change_matching_floor_timestamp(self):
        timestamps = select_keyframe_timestamps(
            duration=10.0, scene_timestamps=[5.0], floor_interval=5.0
        )
        assert timestamps == [0.0, 5.0]

    def test_ignores_scene_changes_outside_duration(self):
        timestamps = select_keyframe_timestamps(
            duration=10.0, scene_timestamps=[-1.0, 999.0], floor_interval=5.0
        )
        assert timestamps == [0.0, 5.0]

    def test_caps_at_max_frames(self):
        # 100s / 1s floor -> 100 raw timestamps, must be capped.
        timestamps = select_keyframe_timestamps(
            duration=100.0, scene_timestamps=[], floor_interval=1.0, max_frames=20
        )
        assert len(timestamps) <= 20
        assert len(timestamps) > 0

    def test_capped_result_keeps_first_and_last(self):
        timestamps = select_keyframe_timestamps(
            duration=100.0, scene_timestamps=[], floor_interval=1.0, max_frames=10
        )
        assert timestamps[0] == 0.0
        assert timestamps[-1] == pytest.approx(99.0)

    def test_capped_result_is_sorted_and_unique(self):
        timestamps = select_keyframe_timestamps(
            duration=200.0, scene_timestamps=[10.3, 45.6, 120.9], floor_interval=2.0, max_frames=25
        )
        assert timestamps == sorted(set(timestamps))
        assert len(timestamps) <= 25

    def test_short_video_under_floor_interval_still_gets_one_frame(self):
        timestamps = select_keyframe_timestamps(duration=3.0, scene_timestamps=[], floor_interval=5.0)
        assert timestamps == [0.0]

    def test_zero_duration_returns_single_zero_timestamp(self):
        timestamps = select_keyframe_timestamps(duration=0.0, scene_timestamps=[], floor_interval=5.0)
        assert timestamps == [0.0]

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError):
            select_keyframe_timestamps(duration=-1.0, scene_timestamps=[])

    def test_rejects_non_positive_floor_interval(self):
        with pytest.raises(ValueError):
            select_keyframe_timestamps(duration=10.0, scene_timestamps=[], floor_interval=0)

    def test_rejects_max_frames_below_one(self):
        with pytest.raises(ValueError):
            select_keyframe_timestamps(duration=10.0, scene_timestamps=[], max_frames=0)

    def test_max_frames_one_returns_single_earliest_timestamp(self):
        timestamps = select_keyframe_timestamps(
            duration=50.0, scene_timestamps=[], floor_interval=5.0, max_frames=1
        )
        assert timestamps == [0.0]

    def test_never_exceeds_upper_bound_of_spec_range(self):
        # Spec caps ~20-40 total frames regardless of video length.
        timestamps = select_keyframe_timestamps(
            duration=3600.0, scene_timestamps=list(range(0, 3600, 3)), floor_interval=5.0, max_frames=40
        )
        assert len(timestamps) <= 40


class TestProbeDuration:
    def test_parses_ffprobe_stdout(self):
        fake_runner = MagicMock(return_value=MagicMock(stdout="123.456\n", stderr=""))
        duration = probe_duration(Path("video.mp4"), runner=fake_runner)
        assert duration == pytest.approx(123.456)
        fake_runner.assert_called_once()
        args = fake_runner.call_args[0][0]
        assert args[0] == "ffprobe"
        assert "video.mp4" in args


class TestDetectSceneChanges:
    def test_parses_pts_time_from_stderr(self):
        fake_stderr = (
            "frame=1 pts_time:0.5 something\n"
            "frame=2 pts_time:3.25 something\n"
        )
        fake_runner = MagicMock(return_value=MagicMock(stdout="", stderr=fake_stderr))
        timestamps = detect_scene_changes(Path("video.mp4"), runner=fake_runner)
        assert timestamps == [0.5, 3.25]
        args = fake_runner.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert any("scene" in a for a in args)

    def test_returns_empty_list_when_no_scenes_detected(self):
        fake_runner = MagicMock(return_value=MagicMock(stdout="", stderr="no matches here"))
        timestamps = detect_scene_changes(Path("video.mp4"), runner=fake_runner)
        assert timestamps == []


class TestExtractFrames:
    def test_calls_ffmpeg_once_per_timestamp_and_returns_paths(self, tmp_path):
        fake_runner = MagicMock(return_value=MagicMock(stdout="", stderr="", returncode=0))
        output_dir = tmp_path / "frames"
        paths = extract_frames(Path("video.mp4"), [0.0, 5.0, 10.0], output_dir, runner=fake_runner)

        assert fake_runner.call_count == 3
        assert paths == [output_dir / "001.jpg", output_dir / "002.jpg", output_dir / "003.jpg"]
        assert output_dir.exists()

    def test_no_real_subprocess_invoked(self, tmp_path):
        fake_runner = MagicMock(return_value=MagicMock(stdout="", stderr="", returncode=0))
        extract_frames(Path("video.mp4"), [1.0], tmp_path / "frames", runner=fake_runner)
        # The injected runner stands in for subprocess.run entirely.
        fake_runner.assert_called_once()
