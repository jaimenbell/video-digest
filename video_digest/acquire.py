"""Stage 1: acquire a video — download via yt-dlp if given a URL, or pass a local path through."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlparse


def is_url(source: str) -> bool:
    """Pure check: does `source` look like a URL (vs a local file path)?"""
    try:
        parsed = urlparse(source)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https")


def acquire_video(source: str, scratch_dir: Path, runner=subprocess.run) -> Path:
    """Return a local path to the video, downloading via yt-dlp first if `source` is a URL.

    - If `source` is a local path, it is returned unchanged (as a Path), untouched.
    - If `source` is a URL, yt-dlp downloads it into `scratch_dir` and the resulting
      file path is returned.

    `runner` is injectable (defaults to subprocess.run) so tests can mock it without
    invoking the real yt-dlp binary or touching the network.
    """
    if not is_url(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"local video path does not exist: {path}")
        return path

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(scratch_dir / "source.%(ext)s")

    result = runner(
        [
            "yt-dlp",
            "-f",
            "mp4/best",
            "-o",
            output_template,
            "--no-playlist",
            source,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    _ = result  # available to callers for logging; not required for the return value

    downloaded = sorted(scratch_dir.glob("source.*"))
    if not downloaded:
        raise RuntimeError(
            f"yt-dlp reported success but no source.* file was found in {scratch_dir}"
        )
    return downloaded[0]
