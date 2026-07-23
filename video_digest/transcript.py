"""Stage 4: transcript — faster-whisper (local, $0), formatted with timestamps.

Model loading is injected via `model_factory` so tests never pull a real
CTranslate2 Whisper model or touch a GPU/CPU inference call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

DEFAULT_MODEL_SIZE = "base"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


def _field(segment: Any, name: str):
    """Read `name` off a segment whether it's an attribute-bearing object
    (like faster-whisper's Segment) or a plain dict."""
    if isinstance(segment, dict):
        return segment[name]
    return getattr(segment, name)


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(segments: Iterable[Any]) -> str:
    """Pure function: render whisper segments into a timestamped text transcript.

    Each segment needs start/end/text (attribute or dict access). Produces one
    line per segment: `[MM:SS - MM:SS] text`.
    """
    lines = []
    for segment in segments:
        start = _format_timestamp(_field(segment, "start"))
        end = _format_timestamp(_field(segment, "end"))
        text = _field(segment, "text").strip()
        lines.append(f"[{start} - {end}] {text}")
    if not lines:
        return "(no speech detected)"
    return "\n".join(lines)


def transcribe_audio(
    audio_path: Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    model_factory=None,
) -> str:
    """Transcribe `audio_path` with faster-whisper and return a formatted transcript.

    `model_factory(model_size, device=..., compute_type=...)` must return an object
    with a `.transcribe(path)` method yielding `(segments, info)` — this is
    faster-whisper's `WhisperModel` interface. Defaults to the real WhisperModel;
    tests should inject a fake factory.
    """
    if model_factory is None:
        from faster_whisper import WhisperModel as model_factory  # noqa: N813

    model = model_factory(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(audio_path))
    return format_transcript(segments)
