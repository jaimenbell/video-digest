"""Unit tests for the optional local-vLLM text-only first-pass summary (HTTP mocked)."""

from unittest.mock import MagicMock

import requests

from video_digest.vllm_summary import SummaryResult, summarize_with_vllm


def _http_error_response(status_code: int, body: str) -> MagicMock:
    """Build a fake requests.Response whose raise_for_status() raises HTTPError."""
    response = MagicMock()
    response.status_code = status_code
    response.text = body
    error = requests.exceptions.HTTPError(response=response)
    response.raise_for_status.side_effect = error
    return response


class TestSummarizeWithVllm:
    def test_returns_content_from_response_on_success(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "choices": [{"message": {"content": "  A short summary.  "}}]
        }
        fake_response.raise_for_status.return_value = None
        fake_post = MagicMock(return_value=fake_response)

        result = summarize_with_vllm("some transcript text", http_post=fake_post)

        assert isinstance(result, SummaryResult)
        assert result.text == "A short summary."
        assert result.error is None
        assert result.ok is True
        fake_post.assert_called_once()

    def test_sends_transcript_text_only_never_images(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_post = MagicMock(return_value=fake_response)

        summarize_with_vllm("transcript body here", http_post=fake_post)

        _, kwargs = fake_post.call_args
        payload = kwargs["json"]
        message_content = payload["messages"][0]["content"]
        assert "transcript body here" in message_content
        assert "image" not in str(payload).lower()

    def test_sends_explicit_max_tokens(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_response.raise_for_status.return_value = None
        fake_post = MagicMock(return_value=fake_response)

        summarize_with_vllm("transcript text", http_post=fake_post, max_tokens=777)

        _, kwargs = fake_post.call_args
        assert kwargs["json"]["max_tokens"] == 777

    def test_uses_default_max_tokens_when_not_overridden(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_response.raise_for_status.return_value = None
        fake_post = MagicMock(return_value=fake_response)

        summarize_with_vllm("transcript text", http_post=fake_post)

        _, kwargs = fake_post.call_args
        assert kwargs["json"]["max_tokens"] == 1024

    def test_returns_no_op_on_empty_transcript(self):
        fake_post = MagicMock()
        result = summarize_with_vllm("   ", http_post=fake_post)
        assert result.text is None
        assert result.error is None
        fake_post.assert_not_called()

    def test_connection_failure_gets_distinct_message(self):
        # The old code collapsed this into a bare None indistinguishable from
        # overflow/timeout/parse failures.
        fake_post = MagicMock(side_effect=ConnectionError("no server"))
        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("connection:")

    def test_timeout_gets_distinct_message(self):
        fake_post = MagicMock(side_effect=requests.exceptions.Timeout("slow"))
        result = summarize_with_vllm("transcript text", http_post=fake_post, timeout=30.0)
        assert result.text is None
        assert result.error.startswith("timeout:")

    def test_context_overflow_gets_distinct_message(self):
        fake_response = _http_error_response(
            400, "This model's maximum context length is 8192 tokens"
        )
        fake_post = MagicMock(return_value=fake_response)
        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("context-overflow:")

    def test_other_http_error_is_not_mistaken_for_overflow(self):
        fake_response = _http_error_response(500, "internal server error")
        fake_post = MagicMock(return_value=fake_response)
        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("http-error:")

    def test_malformed_response_gets_distinct_message(self):
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"unexpected": "shape"}
        fake_post = MagicMock(return_value=fake_response)

        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result.text is None
        assert result.error.startswith("parse-error:")

    def test_each_failure_mode_produces_a_distinct_message(self):
        # Regression guard: overflow / connection / timeout / parse must never
        # collapse back into the same opaque bare-except None again.
        connection_result = summarize_with_vllm(
            "t", http_post=MagicMock(side_effect=ConnectionError("x"))
        )
        timeout_result = summarize_with_vllm(
            "t", http_post=MagicMock(side_effect=requests.exceptions.Timeout("x"))
        )
        overflow_result = summarize_with_vllm(
            "t",
            http_post=MagicMock(
                return_value=_http_error_response(400, "maximum context length")
            ),
        )
        parse_response = MagicMock()
        parse_response.raise_for_status.return_value = None
        parse_response.json.return_value = {}
        parse_result = summarize_with_vllm(
            "t", http_post=MagicMock(return_value=parse_response)
        )

        kinds = {
            connection_result.error.split(":")[0],
            timeout_result.error.split(":")[0],
            overflow_result.error.split(":")[0],
            parse_result.error.split(":")[0],
        }
        assert kinds == {"connection", "timeout", "context-overflow", "parse-error"}
