"""Optional stage: a $0 first-pass mechanical summary of the transcript TEXT ONLY,
via the operator's local vLLM server. vLLM never sees the keyframes/images —
Claude (or a human) does the actual visual synthesis after reviewing the frames.

This is a convenience default, not a hard dependency: if the local vLLM server is
unreachable, `summarize_with_vllm` returns None and the caller falls back to a
plain template the human/Claude fills in.
"""

from __future__ import annotations

DEFAULT_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3-14b"

_PROMPT_TEMPLATE = (
    "You are given the timestamped transcript of a short video. Write a concise "
    "3-5 sentence plain-text summary of what is said, no preamble, no markdown "
    "headers.\n\nTranscript:\n{transcript}"
)


def summarize_with_vllm(
    transcript_text: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    http_post=None,
    timeout: float = 30.0,
) -> str | None:
    """Ask the local vLLM server for a first-pass text-only summary.

    `http_post` is injectable (defaults to `requests.post`) so tests never make a
    real HTTP call. Returns None (rather than raising) on any failure — the
    caller is expected to fall back to a template the human/Claude fills in.
    """
    if not transcript_text or not transcript_text.strip():
        return None

    if http_post is None:
        import requests

        http_post = requests.post

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": _PROMPT_TEMPLATE.format(transcript=transcript_text)}
        ],
        "temperature": 0.2,
    }

    try:
        response = http_post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
