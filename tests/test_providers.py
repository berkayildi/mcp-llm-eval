"""Tests for provider runners with mocked SDK clients."""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mcp_llm_eval.providers import anthropic as anthropic_runner
from mcp_llm_eval.providers import openai as openai_runner
from mcp_llm_eval.providers import google as google_runner


# ---------------------------------------------------------------------------
# Anthropic runner
# ---------------------------------------------------------------------------


class TestAnthropicRunner:
    def _make_mock_client(self, response_text="Hello world", input_tok=50, output_tok=20):
        client = MagicMock()
        stream_ctx = MagicMock()
        stream_obj = MagicMock()

        # text_stream yields chunks
        stream_obj.text_stream = iter([response_text[:5], response_text[5:]])

        # get_final_message
        final = MagicMock()
        final.usage.input_tokens = input_tok
        final.usage.output_tokens = output_tok
        final.stop_reason = "end_turn"
        stream_obj.get_final_message.return_value = final

        stream_ctx.__enter__ = MagicMock(return_value=stream_obj)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        client.messages.stream.return_value = stream_ctx
        return client

    def test_basic_run(self):
        client = self._make_mock_client("Hello world")
        result = anthropic_runner.run(client, "claude-3", "system", "question")
        assert result["response"] == "Hello world"
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 20
        assert result["stop_reason"] == "end_turn"

    def test_ttft_captured(self):
        client = self._make_mock_client("Hello")
        result = anthropic_runner.run(client, "m", "s", "q")
        assert result["time_to_first_token_ms"] is not None
        assert isinstance(result["time_to_first_token_ms"], int)

    def test_total_latency_captured(self):
        client = self._make_mock_client("Hi")
        result = anthropic_runner.run(client, "m", "s", "q")
        assert result["total_latency_ms"] >= 0

    def test_max_tokens_passed(self):
        client = self._make_mock_client()
        anthropic_runner.run(client, "m", "s", "q", max_tokens=1000)
        call_kwargs = client.messages.stream.call_args[1]
        assert call_kwargs["max_tokens"] == 1000

    def test_system_prompt_passed(self):
        client = self._make_mock_client()
        anthropic_runner.run(client, "m", "my system prompt", "q")
        call_kwargs = client.messages.stream.call_args[1]
        assert call_kwargs["system"] == "my system prompt"

    def test_question_passed(self):
        client = self._make_mock_client()
        anthropic_runner.run(client, "m", "s", "what is 2+2?")
        call_kwargs = client.messages.stream.call_args[1]
        assert call_kwargs["messages"][0]["content"] == "what is 2+2?"

    def test_model_passed(self):
        client = self._make_mock_client()
        anthropic_runner.run(client, "claude-sonnet-4-6", "s", "q")
        call_kwargs = client.messages.stream.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_empty_response(self):
        client = MagicMock()
        stream_ctx = MagicMock()
        stream_obj = MagicMock()
        stream_obj.text_stream = iter([])
        final = MagicMock()
        final.usage.input_tokens = 10
        final.usage.output_tokens = 0
        final.stop_reason = "end_turn"
        stream_obj.get_final_message.return_value = final
        stream_ctx.__enter__ = MagicMock(return_value=stream_obj)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        client.messages.stream.return_value = stream_ctx

        result = anthropic_runner.run(client, "m", "s", "q")
        assert result["response"] == ""
        assert result["time_to_first_token_ms"] is None

    def test_get_final_message_error_handled(self):
        client = MagicMock()
        stream_ctx = MagicMock()
        stream_obj = MagicMock()
        stream_obj.text_stream = iter(["hi"])
        stream_obj.get_final_message.side_effect = RuntimeError("oops")
        stream_ctx.__enter__ = MagicMock(return_value=stream_obj)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        client.messages.stream.return_value = stream_ctx

        result = anthropic_runner.run(client, "m", "s", "q")
        assert result["response"] == "hi"
        assert result["input_tokens"] == 0


class TestAnthropicGetClient:
    def test_missing_sdk_raises(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install anthropic"):
                anthropic_runner._get_client()


# ---------------------------------------------------------------------------
# OpenAI runner
# ---------------------------------------------------------------------------


class TestOpenaiRunner:
    def _make_chunks(self, text="Hello world", input_tok=30, output_tok=15):
        chunks = []
        # Content chunks
        for char in text:
            chunk = MagicMock()
            chunk.usage = None
            choice = MagicMock()
            choice.delta = MagicMock(content=char)
            choice.finish_reason = None
            chunk.choices = [choice]
            chunks.append(chunk)

        # Final chunk with finish reason
        final_content = MagicMock()
        final_content.usage = None
        final_choice = MagicMock()
        final_choice.delta = MagicMock(content=None)
        final_choice.finish_reason = "stop"
        final_content.choices = [final_choice]
        chunks.append(final_content)

        # Usage chunk
        usage_chunk = MagicMock()
        usage_chunk.usage = MagicMock(prompt_tokens=input_tok, completion_tokens=output_tok)
        usage_chunk.choices = []
        chunks.append(usage_chunk)

        return chunks

    def _make_mock_client(self, text="Hello world"):
        client = MagicMock()
        client.chat.completions.create.return_value = iter(self._make_chunks(text))
        return client

    def test_basic_run(self):
        client = self._make_mock_client("Hello")
        result = openai_runner.run(client, "gpt-4o", "system", "question")
        assert result["response"] == "Hello"
        assert result["input_tokens"] == 30
        assert result["output_tokens"] == 15
        assert result["stop_reason"] == "stop"

    def test_ttft_captured(self):
        client = self._make_mock_client("Hi")
        result = openai_runner.run(client, "m", "s", "q")
        assert result["time_to_first_token_ms"] is not None

    def test_total_latency_captured(self):
        client = self._make_mock_client("Hi")
        result = openai_runner.run(client, "m", "s", "q")
        assert result["total_latency_ms"] >= 0

    def test_max_tokens_passed(self):
        client = self._make_mock_client()
        openai_runner.run(client, "m", "s", "q", max_tokens=800)
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 800

    def test_system_and_user_messages(self):
        client = self._make_mock_client()
        openai_runner.run(client, "m", "sys prompt", "user question")
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "sys prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "user question"

    def test_stream_options_include_usage(self):
        client = self._make_mock_client()
        openai_runner.run(client, "m", "s", "q")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"]["include_usage"] is True

    def test_empty_response(self):
        client = MagicMock()
        chunk = MagicMock()
        chunk.usage = MagicMock(prompt_tokens=10, completion_tokens=0)
        chunk.choices = []
        client.chat.completions.create.return_value = iter([chunk])

        result = openai_runner.run(client, "m", "s", "q")
        assert result["response"] == ""
        assert result["time_to_first_token_ms"] is None


class TestOpenaiGetClient:
    def test_missing_sdk_raises(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="pip install openai"):
                openai_runner._get_client()


# ---------------------------------------------------------------------------
# Google runner
# ---------------------------------------------------------------------------


class TestGoogleRunner:
    def _setup_genai_mock(self):
        """Inject a mock genai module into the google provider."""
        mock_genai = MagicMock()
        mock_genai.types = MagicMock()
        google_runner.genai = mock_genai
        return mock_genai

    def _make_mock_client(self, text="Hello world", input_tok=40, output_tok=25):
        client = MagicMock()

        chunks = []
        for i, char in enumerate(text):
            chunk = MagicMock()
            chunk.text = char
            chunk.usage_metadata = None
            chunk.candidates = None
            chunks.append(chunk)

        # Last chunk has usage and candidates
        if chunks:
            last = chunks[-1]
            last.usage_metadata = MagicMock(
                prompt_token_count=input_tok,
                candidates_token_count=output_tok,
            )
            candidate = MagicMock()
            candidate.finish_reason = "STOP"
            last.candidates = [candidate]

        client.models.generate_content_stream.return_value = iter(chunks)
        return client

    def test_basic_run(self):
        self._setup_genai_mock()
        client = self._make_mock_client("Hello")
        result = google_runner.run(client, "gemini-2.5-flash", "system", "question")
        assert result["response"] == "Hello"
        assert result["input_tokens"] == 40
        assert result["output_tokens"] == 25

    def test_ttft_captured(self):
        self._setup_genai_mock()
        client = self._make_mock_client("Hi")
        result = google_runner.run(client, "m", "s", "q")
        assert result["time_to_first_token_ms"] is not None

    def test_total_latency_captured(self):
        self._setup_genai_mock()
        client = self._make_mock_client("Hi")
        result = google_runner.run(client, "m", "s", "q")
        assert result["total_latency_ms"] >= 0

    def test_empty_response(self):
        self._setup_genai_mock()
        client = MagicMock()
        client.models.generate_content_stream.return_value = iter([])

        result = google_runner.run(client, "m", "s", "q")
        assert result["response"] == ""
        assert result["time_to_first_token_ms"] is None
        assert result["input_tokens"] == 0

    def test_stop_reason_captured(self):
        self._setup_genai_mock()
        client = self._make_mock_client("ok")
        result = google_runner.run(client, "m", "s", "q")
        assert result["stop_reason"] == "STOP"


class TestGoogleGetClient:
    def test_missing_sdk_raises(self):
        # Reset the module-level genai to None so _ensure_genai tries to import
        original = google_runner.genai
        google_runner.genai = None
        try:
            with patch.dict("sys.modules", {"google": None, "google.genai": None}):
                with pytest.raises(ImportError, match="pip install google-genai"):
                    google_runner._get_client()
        finally:
            google_runner.genai = original
