"""Optional stage: a $0 first-pass mechanical summary of the transcript TEXT ONLY,
via the operator's local vLLM server. vLLM never sees the keyframes/images —
Claude (or a human) does the actual visual synthesis after reviewing the frames.

This is a convenience default, not a hard dependency: if the local vLLM server is
unreachable, or the transcript overflows the model's context window,
`summarize_with_vllm` returns a `SummaryResult` with `text=None` and a distinct,
typed `error` message instead of a bare None — the caller (the CLI) can then
report *why* the fallback TODO placeholder is being used instead of guessing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import requests

DEFAULT_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3-14b"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 30.0

_PROMPT_TEMPLATE = (
    "You are given the timestamped transcript of a short video. Write a concise "
    "3-5 sentence plain-text summary of what is said, no preamble, no markdown "
    "headers.\n\nTranscript:\n{transcript}"
)

_OVERFLOW_PHRASES = ("context length", "maximum context", "context window", "too long")


@dataclass
class SummaryResult:
    """Result of one `summarize_with_vllm` call.

    `error` is None on success, and also None for the legitimate empty-transcript
    no-op (nothing to summarize is not a failure). On failure it is a
    `"<kind>: <detail>"` string with one of a small set of distinct kinds, so a
    caller can tell these apart instead of getting an opaque None:

        "context-overflow: ..." - the request didn't fit the model's context window
        "connection: ..."       - could not reach the vLLM server at all
        "timeout: ..."          - the request exceeded its timeout
        "http-error: ..."       - a non-2xx status that isn't a context overflow
        "parse-error: ..."      - the response body couldn't be parsed into text
    """

    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.text is not None


def _log_failure(message: str) -> None:
    print(f"[vllm_summary] {message}", file=sys.stderr)


def _is_context_overflow(status_code: int | None, body: str) -> bool:
    if status_code != 400:
        return False
    body_lower = body.lower()
    return any(phrase in body_lower for phrase in _OVERFLOW_PHRASES)


def _post_chat_completion(
    prompt: str,
    endpoint: str,
    model: str,
    http_post,
    timeout: float,
    max_tokens: int,
) -> SummaryResult:
    """Low-level: POST one chat/completions request and classify any failure."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    try:
        response = http_post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        message = f"timeout: request to {endpoint} exceeded {timeout}s ({exc})"
        _log_failure(message)
        return SummaryResult(text=None, error=message)
    except (requests.exceptions.ConnectionError, ConnectionError) as exc:
        message = f"connection: could not reach vLLM server at {endpoint} ({exc})"
        _log_failure(message)
        return SummaryResult(text=None, error=message)
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        body = ""
        if exc.response is not None:
            try:
                body = (exc.response.text or "")[:500]
            except Exception:
                body = ""
        if _is_context_overflow(status_code, body):
            message = (
                f"context-overflow: vLLM rejected the request as too long for its "
                f"context window (HTTP {status_code}): {body}"
            )
        else:
            message = f"http-error: vLLM returned HTTP {status_code}: {body}"
        _log_failure(message)
        return SummaryResult(text=None, error=message)
    except requests.exceptions.RequestException as exc:
        message = f"connection: request to vLLM failed ({exc})"
        _log_failure(message)
        return SummaryResult(text=None, error=message)

    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        message = f"parse-error: could not parse vLLM response ({exc})"
        _log_failure(message)
        return SummaryResult(text=None, error=message)

    return SummaryResult(text=text, error=None)


def summarize_with_vllm(
    transcript_text: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    http_post=None,
    timeout: float = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SummaryResult:
    """Ask the local vLLM server for a first-pass text-only summary.

    `http_post` is injectable (defaults to `requests.post`) so tests never make a
    real HTTP call. Returns a `SummaryResult` (never raises) — the caller falls
    back to a template placeholder when `result.text` is None, and can log/print
    `result.error` to say why.
    """
    if not transcript_text or not transcript_text.strip():
        return SummaryResult(text=None, error=None)

    if http_post is None:
        http_post = requests.post

    prompt = _PROMPT_TEMPLATE.format(transcript=transcript_text)
    return _post_chat_completion(prompt, endpoint, model, http_post, timeout, max_tokens)
