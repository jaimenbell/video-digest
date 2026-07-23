"""Unit tests for the optional local-vLLM text-only first-pass summary (HTTP mocked)."""

from unittest.mock import MagicMock

from video_digest.vllm_summary import summarize_with_vllm


class TestSummarizeWithVllm:
    def test_returns_content_from_response_on_success(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "choices": [{"message": {"content": "  A short summary.  "}}]
        }
        fake_response.raise_for_status.return_value = None
        fake_post = MagicMock(return_value=fake_response)

        result = summarize_with_vllm("some transcript text", http_post=fake_post)

        assert result == "A short summary."
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

    def test_returns_none_on_empty_transcript(self):
        fake_post = MagicMock()
        result = summarize_with_vllm("   ", http_post=fake_post)
        assert result is None
        fake_post.assert_not_called()

    def test_returns_none_when_vllm_unreachable(self):
        fake_post = MagicMock(side_effect=ConnectionError("no server"))
        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result is None

    def test_returns_none_on_malformed_response(self):
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"unexpected": "shape"}
        fake_post = MagicMock(return_value=fake_response)

        result = summarize_with_vllm("transcript text", http_post=fake_post)
        assert result is None
