"""CLI entry point + pipeline orchestration.

    python -m video_digest <url-or-path> [--output digests/] [--model base]
                           [--vault-inbox <path>] [--no-vllm-summary] [--no-visual]

Every stage function is injected as a keyword argument to `run_pipeline` so the
whole pipeline is testable without a real yt-dlp/ffmpeg/faster-whisper/vLLM call.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import date as date_cls
from pathlib import Path

from .acquire import acquire_video
from .audio import extract_audio
from .digest import assemble_digest
from .keyframes import (
    DEFAULT_FLOOR_INTERVAL_SECONDS,
    DEFAULT_MAX_FRAMES,
    detect_scene_changes,
    extract_frames,
    probe_duration,
    select_keyframe_timestamps,
)
from .transcript import transcribe_audio
from .visual import analyze_frames, summarize_visuals
from .vllm_summary import summarize_with_vllm

DEFAULT_SCRATCH_DIR = Path("scratch")
DEFAULT_OUTPUT_DIR = Path("digests")


def slugify(text: str) -> str:
    """Pure function: turn a title/source string into a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:60] or "video"


def title_from_source(source: str) -> str:
    """Pure function: derive a human-ish title from a URL or local path."""
    return Path(source).stem or source


def run_pipeline(
    source: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scratch_dir: Path = DEFAULT_SCRATCH_DIR,
    model_size: str = "base",
    vault_inbox: Path | None = None,
    use_vllm_summary: bool = True,
    use_visual_analysis: bool = True,
    acquire_fn=acquire_video,
    probe_duration_fn=probe_duration,
    detect_scenes_fn=detect_scene_changes,
    extract_frames_fn=extract_frames,
    extract_audio_fn=extract_audio,
    transcribe_fn=transcribe_audio,
    summarize_fn=summarize_with_vllm,
    analyze_frames_fn=analyze_frames,
    summarize_visuals_fn=summarize_visuals,
) -> Path:
    """Run the full acquire -> keyframes -> audio -> transcript -> digest pipeline.

    Returns the path to the written digest markdown file.
    """
    scratch_dir = Path(scratch_dir)
    output_dir = Path(output_dir)
    frames_dir = scratch_dir / "frames"
    audio_path = scratch_dir / "audio.wav"
    transcript_path = scratch_dir / "transcript.txt"

    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = acquire_fn(source, scratch_dir)

    duration = probe_duration_fn(video_path)
    scene_timestamps = detect_scenes_fn(video_path)
    timestamps = select_keyframe_timestamps(
        duration,
        scene_timestamps,
        floor_interval=DEFAULT_FLOOR_INTERVAL_SECONDS,
        max_frames=DEFAULT_MAX_FRAMES,
    )
    frame_paths = extract_frames_fn(video_path, timestamps, frames_dir)

    extract_audio_fn(video_path, audio_path)
    transcript_text = transcribe_fn(audio_path, model_size=model_size)
    transcript_path.write_text(transcript_text, encoding="utf-8")

    summary_text = None
    if use_vllm_summary:
        summary_text = summarize_fn(transcript_text)

    frame_visuals = None
    visual_rollup = None
    if use_visual_analysis:
        frame_visuals = analyze_frames_fn(frame_paths)
        visual_rollup = summarize_visuals_fn(frame_visuals)

    title = title_from_source(source)
    digest_md = assemble_digest(
        title=title,
        source=source,
        frame_paths=frame_paths,
        transcript_text=transcript_text,
        frame_timestamps=timestamps,
        summary_text=summary_text,
        frame_visuals=frame_visuals,
        visual_rollup=visual_rollup,
    )

    today = date_cls.today().isoformat()
    digest_filename = f"video-digest-{today}-{slugify(title)}.md"
    digest_path = output_dir / digest_filename
    digest_path.write_text(digest_md, encoding="utf-8")

    if vault_inbox is not None:
        vault_inbox = Path(vault_inbox)
        vault_inbox.mkdir(parents=True, exist_ok=True)
        shutil.copy2(digest_path, vault_inbox / digest_filename)

    return digest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-digest",
        description="Turn a video (URL or local mp4 path) into keyframes + transcript + markdown digest.",
    )
    parser.add_argument("source", help="Video URL (yt-dlp) or local mp4 path")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for the digest markdown (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--scratch",
        default=str(DEFAULT_SCRATCH_DIR),
        help=f"Scratch directory for intermediate files (default: {DEFAULT_SCRATCH_DIR})",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="faster-whisper model size (tiny/base/small/medium/large-v3, default: base)",
    )
    parser.add_argument(
        "--vault-inbox",
        default=None,
        help="Optional: also copy the finished digest into this directory (e.g. an Obsidian vault inbox)",
    )
    parser.add_argument(
        "--no-vllm-summary",
        action="store_true",
        help="Skip the local vLLM first-pass text summary; leave a TODO placeholder instead",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help=(
            "Skip the deterministic keyframe colour analysis (palette/luminance/"
            "saturation/bright-mark density); leave TODO placeholders instead"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    digest_path = run_pipeline(
        source=args.source,
        output_dir=Path(args.output),
        scratch_dir=Path(args.scratch),
        model_size=args.model,
        vault_inbox=Path(args.vault_inbox) if args.vault_inbox else None,
        use_vllm_summary=not args.no_vllm_summary,
        use_visual_analysis=not args.no_visual,
    )
    print(f"Digest written: {digest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
