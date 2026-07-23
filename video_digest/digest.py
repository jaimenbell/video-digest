"""Stage 5: digest assembly — pure template assembly, no visual synthesis.

This module never looks at pixels. It takes frame paths (+ optional timestamps),
transcript text, and an optional mechanical summary (e.g. from local vLLM,
text-only), and assembles the final markdown structure:

    Summary / Visual notes (per keyframe) / Ideas to steal / Action items /
    Full transcript (folded in a <details> block)

The Summary and Visual-notes sections are left as TODO placeholders (or filled
with the optional mechanical summary) for a human or Claude-in-the-loop to
complete after actually looking at the keyframe images — this function cannot
do that part, it just assembles the template.
"""

from __future__ import annotations

from datetime import date as date_cls
from pathlib import Path
from typing import Sequence

VISUAL_NOTE_TODO = "TODO — describe what's on screen in this range"
SUMMARY_TODO = "<!-- TODO: fill in after reviewing the keyframes + transcript below -->"


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


def build_visual_notes_section(
    frame_paths: Sequence[Path | str],
    frame_timestamps: Sequence[float] | None = None,
) -> str:
    """Pure function: render the per-keyframe placeholder list."""
    if not frame_paths:
        return "(no keyframes extracted)"

    lines = []
    for i, frame_path in enumerate(frame_paths):
        ref = _relative_frame_ref(frame_path)
        if frame_timestamps and i < len(frame_timestamps):
            ts_label = f"[{_format_timestamp(frame_timestamps[i])}] "
        else:
            ts_label = ""
        lines.append(f"- {ts_label}`{ref}` — {VISUAL_NOTE_TODO}")
    return "\n".join(lines)


def assemble_digest(
    title: str,
    source: str,
    frame_paths: Sequence[Path | str],
    transcript_text: str,
    frame_timestamps: Sequence[float] | None = None,
    summary_text: str | None = None,
    digest_date: str | None = None,
) -> str:
    """Assemble the final markdown digest. Pure function — no I/O, no subprocess,
    no visual synthesis. `summary_text`, if provided (e.g. from a local vLLM
    text-only first pass), fills the Summary section; otherwise a TODO placeholder
    is left for a human/Claude to fill in after reviewing the keyframes.
    """
    digest_date = digest_date or date_cls.today().isoformat()
    visual_notes = build_visual_notes_section(frame_paths, frame_timestamps)
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
