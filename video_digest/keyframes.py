"""Stage 2: keyframe extraction.

Combines ffmpeg scene-change detection (select='gt(scene,0.3)') with a floor of
1 frame per N seconds so static videos still get coverage, then caps the total
frame count so a long video doesn't blow up the digest.

The timestamp-selection math is pure and unit-testable; only the actual ffmpeg
invocations (scene detection, frame extraction, duration probe) touch subprocess,
and those accept an injectable `runner` for mocking in tests.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

DEFAULT_SCENE_THRESHOLD = 0.3
DEFAULT_FLOOR_INTERVAL_SECONDS = 5.0
DEFAULT_MAX_FRAMES = 30

_PTS_TIME_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


def select_keyframe_timestamps(
    duration: float,
    scene_timestamps: list[float] | None = None,
    floor_interval: float = DEFAULT_FLOOR_INTERVAL_SECONDS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[float]:
    """Pure function: merge scene-change timestamps with a floor sampling grid,
    dedupe, sort, and cap at `max_frames` (evenly downsampled if over the cap).

    Always includes timestamp 0.0 (so the very first frame is covered) as long
    as duration >= 0 and max_frames >= 1.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if floor_interval <= 0:
        raise ValueError("floor_interval must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    scene_timestamps = scene_timestamps or []

    floor_timestamps: list[float] = []
    t = 0.0
    while t < duration or (t == 0.0 and duration == 0.0):
        floor_timestamps.append(round(t, 3))
        t += floor_interval
        if duration == 0.0:
            break

    merged = {round(ts, 3) for ts in floor_timestamps}
    for ts in scene_timestamps:
        if 0 <= ts <= duration:
            merged.add(round(ts, 3))

    timestamps = sorted(merged)

    if len(timestamps) <= max_frames:
        return timestamps

    # Evenly downsample to at most max_frames, always keeping first and last.
    if max_frames == 1:
        return [timestamps[0]]

    last = len(timestamps) - 1
    seen: set[int] = set()
    selected_indices: list[int] = []
    for i in range(max_frames):
        idx = round(i * last / (max_frames - 1))
        if idx not in seen:
            seen.add(idx)
            selected_indices.append(idx)

    return [timestamps[i] for i in selected_indices]


def probe_duration(video_path: Path, runner=subprocess.run) -> float:
    """Return video duration in seconds via ffprobe. `runner` is injectable for tests."""
    result = runner(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def detect_scene_changes(
    video_path: Path,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    runner=subprocess.run,
) -> list[float]:
    """Run ffmpeg's scene-change filter and parse pts_time values from its stderr
    `showinfo` output. `runner` is injectable for tests (mock its stderr)."""
    result = runner(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    stderr = result.stderr or ""
    return [float(m) for m in _PTS_TIME_RE.findall(stderr)]


def extract_frames(
    video_path: Path,
    timestamps: list[float],
    output_dir: Path,
    runner=subprocess.run,
) -> list[Path]:
    """Extract one JPEG frame per timestamp via ffmpeg -ss <t> -vframes 1.

    Writes frames/NNN.jpg (1-indexed, zero-padded to 3 digits) into `output_dir`.
    `runner` is injectable for tests — no real ffmpeg invocation in the test suite.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for i, ts in enumerate(timestamps, start=1):
        frame_path = output_dir / f"{i:03d}.jpg"
        runner(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(ts),
                "-i",
                str(video_path),
                "-vframes",
                "1",
                str(frame_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        frame_paths.append(frame_path)
    return frame_paths
