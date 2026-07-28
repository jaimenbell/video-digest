"""Stage 5: digest assembly — pure template assembly, no visual synthesis.

This module never looks at pixels itself. It takes frame paths (+ optional
timestamps), transcript text, an optional mechanical summary (e.g. from local
vLLM, text-only), and — optionally — the measured `FrameVisual`/`VideoVisual`
records produced by `video_digest.visual`, and assembles the final markdown:

    Summary / Visual notes (per keyframe) / Visual roll-up / Ideas to steal /
    Action items / Full transcript (folded in a <details> block)

The Summary is left as a TODO placeholder unless a mechanical summary is
supplied. The Visual-notes section renders MEASURED colour data when visual
analysis was run (hex swatches + coverage, luminance, saturation, bright-mark
density) and falls back to the TODO placeholder only where there is genuinely
nothing measured to say — no keyframe visuals passed in, or a frame that could
not be read. This module still never fabricates a description of a frame.
"""

from __future__ import annotations

from datetime import date as date_cls
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module import-light
    from .visual import FrameVisual, VideoVisual

VISUAL_NOTE_TODO = "TODO — describe what's on screen in this range"
SUMMARY_TODO = "<!-- TODO: fill in after reviewing the keyframes + transcript below -->"
VISUAL_ROLLUP_NOT_RUN = "(visual analysis not run — no measured palette data)"
UNREADABLE_FRAME_PREFIX = "FRAME UNREADABLE"


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _relative_frame_ref(frame_path: Path | str) -> str:
    path = Path(frame_path)
    # Keep just "frames/NNN.jpg" if that's the tail, otherwise the given path.
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "frames":
        return f"frames/{parts[-1]}"
    return str(path)


def _format_swatches(swatches) -> str:
    return " · ".join(f"`{s.hex_color}` {s.coverage * 100:.1f}%" for s in swatches)


def _measured_frame_lines(visual: "FrameVisual") -> list[str]:
    """Render one analysed frame's measured facts as indented sub-bullets."""
    lines = []
    if visual.substrate is not None:
        lines.append(
            f"  - substrate `{visual.substrate.hex_color}` — luma "
            f"{visual.substrate_luma:.0f}/255 ({visual.substrate_verdict}); "
            f"{visual.dark_fraction * 100:.1f}% of pixels below luma 32"
        )
    if visual.palette:
        lines.append(f"  - palette: {_format_swatches(visual.palette)}")
    lines.append(
        f"  - saturation: mean {visual.mean_saturation:.2f} · "
        f"{visual.vivid_fraction * 100:.1f}% of pixels vivid (S>=0.50)"
    )
    lines.append(
        f"  - bright marks: {visual.bright_mark_density * 100:.2f}% of pixels at/above "
        f"luma 200 ({visual.glow_fraction * 100:.2f}% above 160)"
    )
    if visual.description:
        lines.append(f"  - description: {visual.description}")
    if visual.describe_error:
        lines.append(f"  - description unavailable: {visual.describe_error}")
    return lines


def build_visual_notes_section(
    frame_paths: Sequence[Path | str],
    frame_timestamps: Sequence[float] | None = None,
    frame_visuals: Sequence["FrameVisual"] | None = None,
) -> str:
    """Pure function: render the per-keyframe notes.

    With `frame_visuals` supplied, each frame gets its MEASURED colour data. The
    TODO placeholder survives only where nothing was measured: no visuals passed
    in at all, or a frame that could not be read (which is reported explicitly,
    never silently skipped).
    """
    if not frame_paths:
        return "(no keyframes extracted)"

    lines = []
    for i, frame_path in enumerate(frame_paths):
        ref = _relative_frame_ref(frame_path)
        if frame_timestamps and i < len(frame_timestamps):
            ts_label = f"[{_format_timestamp(frame_timestamps[i])}] "
        else:
            ts_label = ""

        visual = None
        if frame_visuals is not None and i < len(frame_visuals):
            visual = frame_visuals[i]

        if visual is None:
            lines.append(f"- {ts_label}`{ref}` — {VISUAL_NOTE_TODO}")
        elif not visual.ok:
            lines.append(
                f"- {ts_label}`{ref}` — {UNREADABLE_FRAME_PREFIX} "
                f"({visual.error}) — {VISUAL_NOTE_TODO}"
            )
        else:
            lines.append(f"- {ts_label}`{ref}`")
            lines.extend(_measured_frame_lines(visual))
    return "\n".join(lines)


def build_visual_rollup_section(rollup: "VideoVisual | None" = None) -> str:
    """Pure function: render the whole-video measured roll-up."""
    if rollup is None:
        return VISUAL_ROLLUP_NOT_RUN
    if rollup.analyzed_count == 0:
        return (
            f"(no keyframes could be analysed — {rollup.failed_count} unreadable)"
            if rollup.failed_count
            else VISUAL_ROLLUP_NOT_RUN
        )

    lines = [
        f"- Frames analysed: **{rollup.analyzed_count}** "
        f"({rollup.failed_count} unreadable)",
    ]
    if rollup.substrate is not None:
        lines.append(
            f"- Substrate: `{rollup.substrate.hex_color}` — **{rollup.substrate_verdict}** "
            f"(mean {rollup.mean_dark_fraction * 100:.1f}% of pixels below luma 32)"
        )
    lines.append(f"- Mean luma: {rollup.mean_luma:.1f}/255")
    lines.append(
        f"- Saturation: mean {rollup.mean_saturation:.2f} · "
        f"{rollup.mean_vivid_fraction * 100:.1f}% vivid"
    )
    lines.append(
        f"- Bright-mark density: {rollup.mean_bright_mark_density * 100:.2f}% "
        f"of pixels at/above luma 200"
    )

    if rollup.palette:
        lines.append("")
        lines.append("| Role | Hex | Coverage |")
        lines.append("|---|---|---|")
        for rank, swatch in enumerate(rollup.palette):
            role = "substrate" if rank == 0 else f"accent {rank}"
            lines.append(
                f"| {role} | `{swatch.hex_color}` | {swatch.coverage * 100:.2f}% |"
            )

    lines.append("")
    if rollup.outliers:
        lines.append("**Outlier frames** (distinct scenes cluster apart):")
        for outlier in rollup.outliers:
            ref = _relative_frame_ref(outlier.path)
            lines.append(f"- `{ref}` — {outlier.reason}")
    else:
        lines.append("**Outlier frames:** none — the keyframes are visually uniform.")

    return "\n".join(lines)


def assemble_digest(
    title: str,
    source: str,
    frame_paths: Sequence[Path | str],
    transcript_text: str,
    frame_timestamps: Sequence[float] | None = None,
    summary_text: str | None = None,
    digest_date: str | None = None,
    frame_visuals: Sequence["FrameVisual"] | None = None,
    visual_rollup: "VideoVisual | None" = None,
) -> str:
    """Assemble the final markdown digest. Pure function — no I/O, no subprocess,
    no fabricated visual synthesis. `summary_text`, if provided (e.g. from a local
    vLLM text-only first pass), fills the Summary section. `frame_visuals` /
    `visual_rollup`, if provided by `video_digest.visual`, fill the Visual-notes
    and Visual-roll-up sections with measured colour data instead of placeholders.
    """
    digest_date = digest_date or date_cls.today().isoformat()
    visual_notes = build_visual_notes_section(frame_paths, frame_timestamps, frame_visuals)
    visual_rollup_md = build_visual_rollup_section(visual_rollup)
    summary = summary_text.strip() if summary_text else SUMMARY_TODO

    transcript_text = transcript_text if transcript_text and transcript_text.strip() else "(no transcript available)"

    return f"""---
title: "video-digest — {title}"
date: {digest_date}
source: {source}
tags: [video-digest]
---

# {title}

> [!info] Source
> {source}

## Summary
{summary}

## Visual notes
{visual_notes}

## Visual roll-up
{visual_rollup_md}

## Ideas to steal
- TODO

## Action items
- [ ] TODO

## Full transcript

<details>
<summary>Full transcript (click to expand)</summary>

{transcript_text}

</details>
"""
