"""Stage 3: audio extraction — ffmpeg to mono 16kHz wav (the format faster-whisper wants)."""

from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE_HZ = 16000
CHANNELS = 1


def extract_audio(video_path: Path, output_path: Path, runner=subprocess.run) -> Path:
    """Extract mono 16kHz wav audio from `video_path` to `output_path` via ffmpeg.

    `runner` is injectable (defaults to subprocess.run) so tests can mock it —
    no real ffmpeg invocation happens in the test suite.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runner(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE_HZ),
            "-f",
            "wav",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return output_path
