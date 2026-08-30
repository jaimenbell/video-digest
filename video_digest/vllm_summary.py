"""Optional stage: a $0 first-pass mechanical summary of the transcript TEXT ONLY,
via the operator's local vLLM server. vLLM never sees the keyframes/images —
Claude (or a human) does the actual visual synthesis after reviewing the frames.

This is a convenience default, not a hard dependency. `summarize_with_vllm`
never raises: on any failure it returns a `SummaryResult` with `text=None` and
a distinct, typed `error` message (context-overflow / connection / timeout /
http-error / parse-error / unexpected-error) instead of a bare None, so the
caller (the CLI) can report *why* the fallback TODO placeholder is being used
instead of guessing.

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

Chunk packing needs to know how many characters correspond to one token for
THIS transcript. A fixed chars-per-token constant is wrong in practice: a
whisper-format transcript (`[MM:SS - MM:SS] text` timestamps everywhere)
tokenizes far more densely than prose -- measured against the real server,
~3 chars/token, not the textbook-English ~4. So whenever `/tokenize` answers,
the real ratio (`len(transcript) / token_count`) is derived from that exact
response and used to size chunks, instead of assuming a constant. Only when
`/tokenize` itself is unreachable does packing fall back to a conservative
fixed ratio (`FALLBACK_CHARS_PER_TOKEN`) -- deliberately pessimistic, since
under-counting tokens is exactly what causes a chunk to still overflow.
`SummaryResult.estimator` always says which mode actually ran
("tokenize-endpoint" or "chars/N-fallback") so this never has to be guessed
after the fact.
"""

from __future__ import annotations

import math
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
# can't ask it directly what its context window (or its real chars/token
# ratio) is. Matches this deployment's known `--max-model-len 8192` -- NOT
# used at all when /tokenize answers.
FALLBACK_MAX_MODEL_LEN = 8192
# Deliberately conservative (real English prose is closer to ~4, but a dense
# whisper-format transcript with `[MM:SS - MM:SS]` timestamps measured ~3 on
# the real server) -- under-estimating chars/token means OVER-counting tokens,
# which is the safe direction to be wrong in when we have no live calibration.
FALLBACK_CHARS_PER_TOKEN = 2.5
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

        "context-overflow: ..."  - a request didn't fit the model's context window
        "connection: ..."        - could not reach the vLLM server at all
        "timeout: ..."           - a request exceeded its timeout
        "http-error: ..."        - a non-2xx status that isn't a context overflow
        "parse-error: ..."       - a response body couldn't be parsed into text
        "unexpected-error: ..."  - anything else (e.g. a buggy injected http_post)

    `estimator` says how the transcript's token count/chars-per-token ratio was
    measured ("tokenize-endpoint" or "chars/N-fallback"). `chunked` is True when
    the transcript didn't fit in one request and map-reduce ran.
    `reduce_fallback` is True when even the combined chunk summaries didn't fit
    the context window, so they were concatenated mechanically instead of
    reduced.
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


def _tokens_for_chars(char_count: int, chars_per_token: float) -> int:
    """Round UP -- under-counting tokens is exactly the bug that causes a
    request to still overflow the real context window."""
    if char_count <= 0:
        return 0
    return max(1, math.ceil(char_count / chars_per_token))


def _measure_tokens(
    text: str,
    tokenize_endpoint: str,
    model: str,
    http_post,
    timeout: float,
) -> tuple[int, int, str, float]:
    """Returns (token_count, max_model_len, estimator_name, chars_per_token).

    Tries the server's own /tokenize endpoint first (authoritative: real token
    count AND the server's actual max_model_len). `chars_per_token` is then
    DERIVED from that real response (len(text) / count) rather than assumed --
    this is what makes chunk packing accurate for this specific transcript
    instead of guessing a textbook-English ratio that a dense, timestamped
    transcript doesn't actually follow. Falls back to a conservative fixed
    ratio + FALLBACK_MAX_MODEL_LEN if /tokenize is unreachable or returns an
    unexpected shape -- this is a graceful degrade, never raises.
    """
    try:
        response = http_post(tokenize_endpoint, json={"model": model, "prompt": text}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        count = int(data["count"])
        max_model_len = int(data["max_model_len"])
        chars_per_token = (len(text) / count) if count > 0 else FALLBACK_CHARS_PER_TOKEN
        return count, max_model_len, "tokenize-endpoint", chars_per_token
    except Exception:
        estimated_count = _tokens_for_chars(len(text), FALLBACK_CHARS_PER_TOKEN)
        return estimated_count, FALLBACK_MAX_MODEL_LEN, "chars/N-fallback", FALLBACK_CHARS_PER_TOKEN


def _prompt_token_count(
    prompt: str,
    tokenize_endpoint: str,
    model: str,
    http_post,
    timeout: float,
    chars_per_token: float,
) -> int:
    """Real token count for ONE prompt via /tokenize, falling back to the
    rounded-up chars estimate. Never raises."""
    try:
        response = http_post(
            tokenize_endpoint, json={"model": model, "prompt": prompt}, timeout=timeout
        )
        response.raise_for_status()
        return int(response.json()["count"])
    except Exception:
        return _tokens_for_chars(len(prompt), chars_per_token)


def _verified_chunks(
    chunks: list[str],
    tokenize_endpoint: str,
    model: str,
    http_post,
    timeout: float,
    max_tokens: int,
    max_model_len: int,
    chars_per_token: float,
) -> list[str]:
    """Re-verify every packed chunk against the REAL tokenizer before it is
    ever sent. The packer sizes chunks in characters via a global
    chars-per-token ratio; a region denser than the transcript average (heavy
    timestamps, code, numbers) can overflow the window anyway -- the 2026-08-29
    production failure was exactly this: a packed chunk measured 7169 real
    input tokens against a 7168 budget and vLLM 400'd. A chunk that measures
    over budget is re-split using its OWN measured density (true local
    chars-per-token), which converges in one or two passes; a piece that still
    cannot fit is hard-sliced. Never raises; never returns an empty list for
    non-empty input."""
    allowed_prompt = max_model_len - max_tokens - SAFETY_MARGIN_TOKENS
    probe_overhead = _CHUNK_PROMPT_TEMPLATE.format(index=99, total=99, transcript="")
    verified: list[str] = []
    queue = list(chunks)
    guard = 0
    while queue:
        guard += 1
        if guard > 256:
            # Structural runaway stop -- accept the remainder rather than loop
            # forever; the send path still classifies any residual overflow
            # honestly as context-overflow.
            verified.extend(queue)
            break
        chunk = queue.pop(0)
        probe = probe_overhead + chunk
        measured = _prompt_token_count(
            probe, tokenize_endpoint, model, http_post, timeout, chars_per_token
        )
        if measured <= allowed_prompt:
            verified.append(chunk)
            continue
        # True local density of THIS text, from the real measurement.
        local_cpt = (len(probe) / measured) if measured > 0 else chars_per_token
        retry_budget = max(MIN_CHUNK_BUDGET_TOKENS, int(allowed_prompt * 0.9))
        pieces = _chunk_transcript(chunk, retry_budget, local_cpt)
        if len(pieces) <= 1 and pieces and pieces[0] == chunk:
            pieces = _hard_slice(chunk, max(1, len(chunk) // 2))
        # Preserve order: re-verify the split pieces before later chunks.
        queue[0:0] = pieces
    return verified


def _post_chat_completion(
    prompt: str,
    endpoint: str,
    model: str,
    http_post,
    timeout: float,
    max_tokens: int,
) -> SummaryResult:
    """Low-level: POST one chat/completions request and classify any failure.
    Never raises -- any exception, including one from a non-`requests` source
    (e.g. a buggy injected `http_post`), is caught and classified."""
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
    except Exception as exc:
        # Anything that isn't even a requests exception (e.g. a bug in a
        # caller-injected http_post) must still not propagate -- the documented
        # contract is that this function never raises.
        message = f"unexpected-error: request to vLLM raised an unexpected error ({exc})"
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


def _hard_slice(text: str, budget_chars: int) -> list[str]:
    budget_chars = max(1, budget_chars)
    pieces = [text[i : i + budget_chars] for i in range(0, len(text), budget_chars)]
    return pieces or [""]


def _split_oversized_piece(paragraph: str, budget_chars: int) -> list[str]:
    """Split one paragraph into pieces that each fit `budget_chars`, splitting
    on line boundaries first. A single LINE that alone exceeds the budget is
    hard-sliced on its own (regardless of how many other lines surround it) --
    otherwise an oversized line inside an otherwise-normal paragraph would
    still be emitted whole and overflow its chunk."""
    if len(paragraph) <= budget_chars:
        return [paragraph]

    lines = paragraph.split("\n")
    if len(lines) <= 1:
        return _hard_slice(paragraph, budget_chars)

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if len(line) > budget_chars:
            if current:
                pieces.append("\n".join(current))
                current = []
                current_len = 0
            pieces.extend(_hard_slice(line, budget_chars))
            continue
        sep_len = 1 if current else 0  # the "\n" this line would be joined with
        if current and current_len + sep_len + len(line) > budget_chars:
            pieces.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current_len += sep_len + len(line)
            current.append(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _chunk_transcript(text: str, budget_tokens: int, chars_per_token: float) -> list[str]:
    """Split on paragraph (blank-line) boundaries, falling back to line (then
    hard-char) boundaries for any paragraph/line that alone exceeds the
    budget, then greedily pack pieces into chunks that each fit
    `budget_tokens` -- measured in CHARACTERS via `chars_per_token` (the real,
    calibrated ratio when available) and tracking join-separator length
    explicitly, so the packer's own bookkeeping can't silently undercount."""
    budget_tokens = max(1, budget_tokens)
    budget_chars = max(1, int(budget_tokens * chars_per_token))

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]

    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_oversized_piece(paragraph, budget_chars))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        sep_len = 2 if current else 0  # the "\n\n" this piece would be joined with
        if current and current_len + sep_len + len(piece) > budget_chars:
            chunks.append("\n\n".join(current))
            current = [piece]
            current_len = len(piece)
        else:
            current_len += sep_len + len(piece)
            current.append(piece)
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
    chars_per_token: float,
    tokenize_endpoint: str = DEFAULT_TOKENIZE_ENDPOINT,
) -> SummaryResult:
    overhead_map = _tokens_for_chars(
        len(_CHUNK_PROMPT_TEMPLATE.format(index=1, total=99, transcript="")), chars_per_token
    )
    available_for_chunks = max_model_len - max_tokens - SAFETY_MARGIN_TOKENS - overhead_map

    if available_for_chunks < MIN_CHUNK_BUDGET_TOKENS:
        # The configuration itself can't support chunking at all: even a
        # minimal chunk plus the requested completion length wouldn't fit.
        # Flooring the budget here anyway would just send doomed requests.
        message = (
            f"context-overflow: max_model_len={max_model_len} leaves no room for "
            f"even a minimal {MIN_CHUNK_BUDGET_TOKENS}-token chunk once "
            f"max_tokens={max_tokens} and overhead are reserved -- lower "
            f"max_tokens or use a model with a larger context window"
        )
        _log_failure(message)
        return SummaryResult(text=None, error=message, estimator=estimator, chunked=True)

    chunk_budget = max(MIN_CHUNK_BUDGET_TOKENS, available_for_chunks)
    chunks = _chunk_transcript(transcript_text, chunk_budget, chars_per_token)
    chunks = _verified_chunks(
        chunks, tokenize_endpoint, model, http_post, timeout,
        max_tokens, max_model_len, chars_per_token,
    )

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

    # Real measurement of the FULL reduce prompt, not a chars estimate --
    # same density trap as the map chunks (the char ratio under-counts dense
    # text and the reduce would 400). Compared against the full-prompt budget.
    measured_reduce_prompt = _prompt_token_count(
        reduce_prompt, tokenize_endpoint, model, http_post, timeout, chars_per_token
    )

    if measured_reduce_prompt > (max_model_len - max_tokens - SAFETY_MARGIN_TOKENS):
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

    token_count, max_model_len, estimator, chars_per_token = _measure_tokens(
        transcript_text, tokenize_endpoint, model, http_post, timeout
    )

    overhead_single = _tokens_for_chars(len(_PROMPT_TEMPLATE.format(transcript="")), chars_per_token)
    single_budget = max_model_len - max_tokens - SAFETY_MARGIN_TOKENS - overhead_single

    if token_count <= single_budget:
        prompt = _PROMPT_TEMPLATE.format(transcript=transcript_text)
        result = _post_chat_completion(prompt, endpoint, model, http_post, timeout, max_tokens)
        return SummaryResult(text=result.text, error=result.error, estimator=estimator, chunked=False)

    return _map_reduce_summarize(
        transcript_text,
        endpoint,
        model,
        http_post,
        timeout,
        max_tokens,
        max_model_len,
        estimator,
        chars_per_token,
        tokenize_endpoint=tokenize_endpoint,
    )
