"""Unit tests for the optional local-vLLM text-only first-pass summary (HTTP mocked).

Every summarize_with_vllm call now does a /tokenize precheck BEFORE the
chat/completions call, so `http_post` fakes below distinguish the two by URL
(mirroring the real deployment, where both go through the same requests.post).
"""

from unittest.mock import MagicMock

import requests

from video_digest.vllm_summary import (
    DEFAULT_TOKENIZE_ENDPOINT,
    SummaryResult,
    summarize_with_vllm,
)


def _http_error_response(status_code: int, body: str) -> MagicMock:
    """Build a fake requests.Response whose raise_for_status() raises HTTPError."""
    response = MagicMock()
    response.status_code = status_code
    response.text = body
    error = requests.exceptions.HTTPError(response=response)
    response.raise_for_status.side_effect = error
    return response


def _tokenize_response(count: int, max_model_len: int) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"count": count, "max_model_len": max_model_len}
    return response


def _completion_response(content: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def _routed_post(tokenize_response, non_tokenize_response_or_fn):
    """A fake http_post that returns tokenize_response for the /tokenize URL and
    delegates everything else to non_tokenize_response_or_fn (a fixed response,
    or a callable(url, json, timeout) -> response for per-call control)."""

    def _post(url, json=None, timeout=None):
        if url == DEFAULT_TOKENIZE_ENDPOINT:
            if isinstance(tokenize_response, Exception):
                raise tokenize_response
            return tokenize_response
        if callable(non_tokenize_response_or_fn) and not isinstance(
            non_tokenize_response_or_fn, MagicMock
        ):
            return non_tokenize_response_or_fn(url, json, timeout)
        return non_tokenize_response_or_fn

    return MagicMock(side_effect=_post)


class TestSummarizeWithVllmSingleShot:
    def test_returns_content_from_response_on_success(self):
        fake_post = _routed_post(
            _tokenize_response(50, 8192), _completion_response("  A short summary.  ")
        )

        result = summarize_with_vllm("some transcript text", http_post=fake_post)

        assert isinstance(result, SummaryResult)
        assert result.text == "A short summary."
        assert result.error is None
        assert result.ok is True
        assert result.chunked is False
        assert result.estimator == "tokenize-endpoint"
        assert fake_post.call_count == 2

    def test_sends_transcript_text_only_never_images(self):
        fake_post = _routed_post(_tokenize_response(50, 8192), _completion_response("ok"))

        summarize_with_vllm("transcript body here", http_post=fake_post)

        completion_call = fake_post.call_args_list[-1]
        payload = completion_call.kwargs["json"]
        message_content = payload["messages"][0]["content"]
        assert "transcript body here" in message_content
        assert "image" not in str(payload).lower()

    def test_sends_explicit_max_tokens(self):
        fake_post = _routed_post(_tokenize_response(50, 8192), _completion_response("ok"))

        summarize_with_vllm("transcript text", http_post=fake_post, max_tokens=777)

        completion_call = fake_post.call_args_list[-1]
        assert completion_call.kwargs["json"]["max_tokens"] == 777

    def test_uses_default_max_tokens_when_not_overridden(self):
        fake_post = _routed_post(_tokenize_response(50, 8192), _completion_response("ok"))

        summarize_with_vllm("transcript text", http_post=fake_post)

        completion_call = fake_post.call_args_list[-1]
        assert completion_call.kwargs["json"]["max_tokens"] == 1024

    def test_returns_no_op_on_empty_transcript(self):
        fake_post = MagicMock()
        result = summarize_with_vllm("   ", http_post=fake_post)
        assert result.text is None
        assert result.error is None
        fake_post.assert_not_called()

    def test_connection_failure_gets_distinct_message(self):
        # The old code collapsed this into a bare None indistinguishable from
        # overflow/timeout/parse failures.
        fake_post = _routed_post(
            _tokenize_response(50, 8192),
            lambda url, json, timeout: (_ for _ in ()).throw(ConnectionError("no server")),
        )
        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("connection:")

    def test_timeout_gets_distinct_message(self):
        fake_post = _routed_post(
            _tokenize_response(50, 8192),
            lambda url, json, timeout: (_ for _ in ()).throw(requests.exceptions.Timeout("slow")),
        )
        result = summarize_with_vllm("transcript text", http_post=fake_post, timeout=30.0)
        assert result.text is None
        assert result.error.startswith("timeout:")

    def test_context_overflow_gets_distinct_message(self):
        overflow_response = _http_error_response(
            400, "This model's maximum context length is 8192 tokens"
        )
        fake_post = _routed_post(_tokenize_response(50, 8192), overflow_response)
        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("context-overflow:")

    def test_other_http_error_is_not_mistaken_for_overflow(self):
        error_response = _http_error_response(500, "internal server error")
        fake_post = _routed_post(_tokenize_response(50, 8192), error_response)
        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("http-error:")

    def test_malformed_response_gets_distinct_message(self):
        malformed_response = MagicMock()
        malformed_response.raise_for_status.return_value = None
        malformed_response.json.return_value = {"unexpected": "shape"}
        fake_post = _routed_post(_tokenize_response(50, 8192), malformed_response)

        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("parse-error:")

    def test_each_failure_mode_produces_a_distinct_message(self):
        # Regression guard: overflow / connection / timeout / parse must never
        # collapse back into the same opaque bare-except None again.
        connection_result = summarize_with_vllm(
            "t",
            http_post=_routed_post(
                _tokenize_response(5, 8192),
                lambda url, json, timeout: (_ for _ in ()).throw(ConnectionError("x")),
            ),
        )
        timeout_result = summarize_with_vllm(
            "t",
            http_post=_routed_post(
                _tokenize_response(5, 8192),
                lambda url, json, timeout: (_ for _ in ()).throw(requests.exceptions.Timeout("x")),
            ),
        )
        overflow_result = summarize_with_vllm(
            "t",
            http_post=_routed_post(
                _tokenize_response(5, 8192), _http_error_response(400, "maximum context length")
            ),
        )
        parse_response = MagicMock()
        parse_response.raise_for_status.return_value = None
        parse_response.json.return_value = {}
        parse_result = summarize_with_vllm(
            "t", http_post=_routed_post(_tokenize_response(5, 8192), parse_response)
        )

        kinds = {
            connection_result.error.split(":")[0],
            timeout_result.error.split(":")[0],
            overflow_result.error.split(":")[0],
            parse_result.error.split(":")[0],
        }
        assert kinds == {"connection", "timeout", "context-overflow", "parse-error"}


class TestTokenizePrecheck:
    def test_tokenize_request_sends_model_and_transcript(self):
        calls = []

        def fake_post_fn(url, json=None, timeout=None):
            calls.append((url, json))
            if url == DEFAULT_TOKENIZE_ENDPOINT:
                return _tokenize_response(5, 8192)
            return _completion_response("ok")

        fake_post = MagicMock(side_effect=fake_post_fn)
        summarize_with_vllm("transcript body", http_post=fake_post, model="qwen3-14b")

        tokenize_call = next(c for c in calls if c[0] == DEFAULT_TOKENIZE_ENDPOINT)
        assert tokenize_call[1]["prompt"] == "transcript body"
        assert tokenize_call[1]["model"] == "qwen3-14b"

    def test_falls_back_to_chars4_estimator_when_tokenize_unreachable(self):
        def fake_post_fn(url, json=None, timeout=None):
            if url == DEFAULT_TOKENIZE_ENDPOINT:
                raise ConnectionError("tokenize down")
            return _completion_response("ok")

        fake_post = MagicMock(side_effect=fake_post_fn)
        result = summarize_with_vllm("short transcript text", http_post=fake_post)

        assert result.estimator == "chars/4-fallback"
        assert result.error is None
        assert result.text == "ok"

    def test_uses_tokenize_endpoint_when_reachable(self):
        fake_post = _routed_post(_tokenize_response(10, 8192), _completion_response("ok"))
        result = summarize_with_vllm("short transcript text", http_post=fake_post)

        assert result.estimator == "tokenize-endpoint"
        assert result.chunked is False
        assert result.text == "ok"


class TestChunkedMapReduce:
    def _long_transcript(self, paragraphs: int = 20, words_per_paragraph: int = 100) -> str:
        return "\n\n".join(
            f"Paragraph {i}. " + ("word " * words_per_paragraph) for i in range(paragraphs)
        )

    def test_long_transcript_gets_chunked_via_map_reduce(self):
        transcript = self._long_transcript()

        def fake_post_fn(url, json=None, timeout=None):
            if url == DEFAULT_TOKENIZE_ENDPOINT:
                return _tokenize_response(50000, 500)
            return _completion_response("A short partial summary.")

        fake_post = MagicMock(side_effect=fake_post_fn)

        result = summarize_with_vllm(transcript, http_post=fake_post, max_tokens=100)

        assert result.chunked is True
        assert result.error is None
        assert result.text is not None
        # tokenize + at least 2 map calls + (reduce or fallback with no extra call)
        assert fake_post.call_count >= 3

    def test_reduce_falls_back_to_mechanical_concat_when_summaries_still_overflow(self):
        transcript = self._long_transcript(paragraphs=20, words_per_paragraph=100)
        long_chunk_summary = "summary word " * 40  # long enough that N of these overflow reduce budget

        def fake_post_fn(url, json=None, timeout=None):
            if url == DEFAULT_TOKENIZE_ENDPOINT:
                return _tokenize_response(100000, 300)
            return _completion_response(long_chunk_summary)

        fake_post = MagicMock(side_effect=fake_post_fn)

        result = summarize_with_vllm(transcript, http_post=fake_post, max_tokens=50)

        assert result.chunked is True
        assert result.reduce_fallback is True
        assert result.error is None
        assert result.text is not None
        assert "concatenated here mechanically" in result.text

    def test_map_chunk_failure_is_reported_with_chunk_position(self):
        transcript = self._long_transcript()
        call_count = {"n": 0}

        def fake_post_fn(url, json=None, timeout=None):
            if url == DEFAULT_TOKENIZE_ENDPOINT:
                return _tokenize_response(50000, 500)
            call_count["n"] += 1
            raise requests.exceptions.Timeout("slow chunk")

        fake_post = MagicMock(side_effect=fake_post_fn)
        result = summarize_with_vllm(transcript, http_post=fake_post, max_tokens=100)

        assert result.text is None
        assert result.error.startswith("timeout:")
        assert "chunk" in result.error
        assert result.chunked is True

    def test_long_transcript_never_silently_returns_none_like_the_old_bare_except(self):
        """The old code's bare `except Exception: return None` meant an
        oversized transcript that overflowed vLLM's context window produced
        text=None with NO error message -- indistinguishable from an empty
        transcript or a down server. Chunking means this must never happen
        again: a huge transcript either produces a real chunked summary, or an
        explicit, distinct error -- never a silent (text=None, error=None)."""
        transcript = self._long_transcript(paragraphs=30, words_per_paragraph=150)

        def fake_post_fn(url, json=None, timeout=None):
            if url == DEFAULT_TOKENIZE_ENDPOINT:
                return _tokenize_response(200000, 8192)
            return _completion_response("A concise partial summary.")

        fake_post = MagicMock(side_effect=fake_post_fn)
        result = summarize_with_vllm(transcript, http_post=fake_post)

        assert not (result.text is None and result.error is None)
        assert result.chunked is True
        assert result.text is not None
        assert result.error is None
