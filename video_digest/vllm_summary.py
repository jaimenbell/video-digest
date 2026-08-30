"""Optional stage: a $0 first-pass mechanical summary of the transcript TEXT ONLY,
via the operator's local vLLM server. vLLM never sees the keyframes/images —
Claude (or a human) does the actual visual synthesis after reviewing the frames.

This is a convenience default, not a hard dependency. `summarize_with_vllm`
never raises: on any failure it returns a `SummaryResult` with `text=None` and
a distinct, typed `error` message (context-overflow / connection / timeout /
http-error / parse-error) instead of a bare None, so the caller (the CLI) can
report *why* the fallback TODO placeholder is being used instead of guessing.

The served model (qwen3-14b) has a finite context window (this deployment:
`--max-model-len 8192`). A dense 20-30 min transcript can overflow a single
`chat/completions` request. Before sending anything, this module asks the
server's own `/tokenize` endpoint (NOTE: this is a plain vLLM extra endpoint,
NOT under `/v1` -- `/v1/tokenize` 404s) how many tokens the transcript is and
what the server's actual `max_model_len` is, and budgets off THAT number
rather than a hardcoded constant. If the transcript fits, it is summarized in
one request exactly as before. If it doesn't, the transcript is split on
paragraph/line boundaries into chunks that each fit the budget, each chunk is
summarized independently (map), and the per-chunk summaries are combined into
one final summary (reduce). If even the combined per-chunk summaries don't fit
the context window, they are concatenated mechanically instead of being
reduced further, and the result says so (`reduce_fallback=True`).

If `/tokenize` itself is unreachable, token counts are estimated locally via a
chars/4 heuristic and the server's `max_model_len` is assumed to be
`FALLBACK_MAX_MODEL_LEN` -- `SummaryResult.estimator` always says which
estimator actually ran ("tokenize-endpoint" or "chars/4-fallback") so this
never has to be guessed after the fact.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import requests

DEFAULT_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_TOKENIZE_ENDPOINT = "http://localhost:8000/tokenize"  # NOT /v1/tokenize (404s there)
DEFAULT_MODEL = "qwen3-14b"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 90.0  # p90 under batch load measured ~26.6s single-shot; multi-chunk
# runs need headroom per request, so this is per-request, not a total budget.

# Only used when the server's own /tokenize endpoint can't be reached, so we
# can't ask it directly what its context window is. Matches this deployment's
# known `--max-model-len 8192` -- NOT used at all when /tokenize answers.
FALLBACK_MAX_MODEL_LEN = 8192
CHARS_PER_TOKEN_ESTIMATE = 4
SAFETY_MARGIN_TOKENS = 32  # chat-template special tokens /tokenize doesn't count
MIN_CHUNK_BUDGET_TOKENS = 256  # floor so a tiny/misreported max_model_len can't produce 0-token chunks

_PROMPT_TEMPLATE = (
    "You are given the timestamped transcript of a short video. Write a concise "
    "3-5 sentence plain-text summary of what is said, no preamble, no markdown "
    "headers.\n\nTranscript:\n{transcript}"
)

_CHUNK_PROMPT_TEMPLATE = (
    "You are given part {index} of {total} of the timestamped transcript of a "
    "video (a contiguous excerpt, transcribed in order). Write a concise 2-4 "
    "sentence plain-text summary of what is said in THIS PART ONLY, no preamble, "
    "no markdown headers.\n\nTranscript excerpt:\n{transcript}"
)

_REDUCE_PROMPT_TEMPLATE = (
    "You are given {total} partial summaries of consecutive parts of one video's "
    "transcript, in order. Combine them into one concise 4-6 sentence plain-text "
    "summary of the whole video, no preamble, no markdown headers.\n\n{parts}"
)

_REDUCE_FALLBACK_NOTE = (
    "(vLLM chunked summary: each section below was summarized independently, but "
    "the combined section summaries still exceed the model's context window, so "
    "they are concatenated here mechanically instead of being reduced into one "
    "paragraph.)"
)

_OVERFLOW_PHRASES = ("context length", "maximum context", "context window", "too long")


@dataclass
class SummaryResult:
    """Result of one `summarize_with_vllm` call.

    `error` is None on success, and also None for the legitimate empty-transcript
    no-op (nothing to summarize is not a failure). On failure it is a
    `"<kind>: <detail>"` string with one of a small set of distinct kinds:

        "context-overflow: ..." - a request didn't fit the model's context window
        "connection: ..."       - could not reach the vLLM server at all
        "timeout: ..."          - a request exceeded its timeout
        "http-error: ..."       - a non-2xx status that isn't a context overflow
        "parse-error: ..."      - a response body couldn't be parsed into text

    `estimator` says how the transcript's token count was measured
    ("tokenize-endpoint" or "chars/4-fallback"). `chunked` is True when the
    transcript didn't fit in one request and map-reduce ran. `reduce_fallback`
    is True when even the combined chunk summaries didn't fit the context
    window, so they were concatenated mechanically instead of reduced.
    """

    text: str | None
    error: str | None = None
    estimator: str | None = None
    chunked: bool = False
    reduce_fallback: bool = False

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


def _estimate_tokens(text: str) -> int:
    """Cheap, local, $0 heuristic -- used for chunk packing always, and for the
    overall token count only when the /tokenize endpoint is unreachable."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _measure_tokens(
    text: str,
    tokenize_endpoint: str,
    model: str,
    http_post,
    timeout: float,
) -> tuple[int, int, str]:
    """Returns (token_count, max_model_len, estimator_name).

    Tries the server's own /tokenize endpoint first (authoritative: real token
    count AND the server's actual max_model_len). Falls back to a local chars/4
    estimate + FALLBACK_MAX_MODEL_LEN if /tokenize is unreachable or returns an
    unexpected shape -- this is a graceful degrade, never raises.
    """
    try:
        response = http_post(tokenize_endpoint, json={"model": model, "prompt": text}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return int(data["count"]), int(data["max_model_len"]), "tokenize-endpoint"
    except Exception:
        return _estimate_tokens(text), FALLBACK_MAX_MODEL_LEN, "chars/4-fallback"


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


def _split_oversized_paragraph(paragraph: str, budget_tokens: int) -> list[str]:
    if _estimate_tokens(paragraph) <= budget_tokens:
        return [paragraph]

    lines = paragraph.split("\n")
    if len(lines) <= 1:
        # No line boundary to split on -- hard-slice by an estimated char budget.
        char_budget = max(1, budget_tokens * CHARS_PER_TOKEN_ESTIMATE)
        return [paragraph[i : i + char_budget] for i in range(0, len(paragraph), char_budget)] or [""]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = _estimate_tokens(line)
        if current and current_tokens + line_tokens > budget_tokens:
            pieces.append("\n".join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens
    if current:
        pieces.append("\n".join(current))
    return pieces


def _chunk_transcript(text: str, budget_tokens: int) -> list[str]:
    """Split on paragraph (blank-line) boundaries, falling back to line
    boundaries for any paragraph that alone exceeds the budget, then greedily
    pack pieces into chunks that each fit `budget_tokens` (chars/4 estimate)."""
    budget_tokens = max(1, budget_tokens)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]

    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_oversized_paragraph(paragraph, budget_tokens))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = _estimate_tokens(piece)
        if current and current_tokens + piece_tokens > budget_tokens:
            chunks.append("\n\n".join(current))
            current = [piece]
            current_tokens = piece_tokens
        else:
            current.append(piece)
            current_tokens += piece_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _map_reduce_summarize(
    transcript_text: str,
    endpoint: str,
    model: str,
    http_post,
    timeout: float,
    max_tokens: int,
    max_model_len: int,
    estimator: str,
) -> SummaryResult:
    overhead_map = _estimate_tokens(
        _CHUNK_PROMPT_TEMPLATE.format(index=1, total=99, transcript="")
    )
    chunk_budget = max(
        MIN_CHUNK_BUDGET_TOKENS,
        max_model_len - max_tokens - SAFETY_MARGIN_TOKENS - overhead_map,
    )
    chunks = _chunk_transcript(transcript_text, chunk_budget)

    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_prompt = _CHUNK_PROMPT_TEMPLATE.format(index=i + 1, total=len(chunks), transcript=chunk)
        result = _post_chat_completion(chunk_prompt, endpoint, model, http_post, timeout, max_tokens)
        if result.error:
            message = f"{result.error} (map chunk {i + 1}/{len(chunks)})"
            return SummaryResult(text=None, error=message, estimator=estimator, chunked=True)
        chunk_summaries.append(result.text or "")

    parts_text = "\n\n".join(
        f"[Part {i + 1}/{len(chunk_summaries)}] {s}" for i, s in enumerate(chunk_summaries)
    )
    reduce_prompt = _REDUCE_PROMPT_TEMPLATE.format(total=len(chunk_summaries), parts=parts_text)

    overhead_reduce = _estimate_tokens(_REDUCE_PROMPT_TEMPLATE.format(total=len(chunk_summaries), parts=""))
    reduce_budget = max_model_len - max_tokens - SAFETY_MARGIN_TOKENS - overhead_reduce
    reduce_prompt_tokens = _estimate_tokens(parts_text)

    if reduce_prompt_tokens > reduce_budget:
        fallback_text = _REDUCE_FALLBACK_NOTE + "\n\n" + parts_text
        return SummaryResult(
            text=fallback_text, error=None, estimator=estimator, chunked=True, reduce_fallback=True
        )

    reduce_result = _post_chat_completion(reduce_prompt, endpoint, model, http_post, timeout, max_tokens)
    if reduce_result.error:
        message = f"{reduce_result.error} (reduce step over {len(chunk_summaries)} chunk summaries)"
        return SummaryResult(text=None, error=message, estimator=estimator, chunked=True)

    return SummaryResult(text=reduce_result.text, error=None, estimator=estimator, chunked=True)


def summarize_with_vllm(
    transcript_text: str,
    endpoint: str = DEFAULT_ENDPOINT,
    tokenize_endpoint: str = DEFAULT_TOKENIZE_ENDPOINT,
    model: str = DEFAULT_MODEL,
    http_post=None,
    timeout: float = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SummaryResult:
    """Ask the local vLLM server for a first-pass text-only summary.

    `http_post` is injectable (defaults to `requests.post`) so tests never make a
    real HTTP call -- it is used for both the /tokenize precheck and the
    chat/completions call(s), distinguished by the endpoint URL passed in, same
    as the real deployment. Never raises: returns a `SummaryResult`, falling
    back to `text=None` + a distinct `error` on any failure.
    """
    if not transcript_text or not transcript_text.strip():
        return SummaryResult(text=None, error=None)

    if http_post is None:
        http_post = requests.post

    token_count, max_model_len, estimator = _measure_tokens(
        transcript_text, tokenize_endpoint, model, http_post, timeout
    )

    overhead_single = _estimate_tokens(_PROMPT_TEMPLATE.format(transcript=""))
    single_budget = max_model_len - max_tokens - SAFETY_MARGIN_TOKENS - overhead_single

    if token_count <= single_budget:
        prompt = _PROMPT_TEMPLATE.format(transcript=transcript_text)
        result = _post_chat_completion(prompt, endpoint, model, http_post, timeout, max_tokens)
        return SummaryResult(text=result.text, error=result.error, estimator=estimator, chunked=False)

    return _map_reduce_summarize(
        transcript_text, endpoint, model, http_post, timeout, max_tokens, max_model_len, estimator
    )
