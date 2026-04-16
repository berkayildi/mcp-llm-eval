"""Tests for judge scoring with mocked OpenAI client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_eval.judge import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    _get_openai_client,
    evaluate_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str) -> MagicMock:
    """Create a mock OpenAI client that returns the given text."""
    client = MagicMock()
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(prompt_tokens=200, completion_tokens=50)
    client.chat.completions.create.return_value = completion
    return client


def _valid_scores_json(**overrides) -> str:
    scores = {
        "faithfulness_score": 0.9,
        "faithfulness_reason": "Well grounded in context",
        "relevance_score": 0.85,
        "relevance_reason": "Directly addresses the question",
    }
    scores.update(overrides)
    return json.dumps(scores)


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    def test_basic_scoring(self):
        client = _make_mock_client(_valid_scores_json())
        result = evaluate_response(
            client=client,
            context="Notes on a database migration.",
            question="Why DynamoDB?",
            expected_response="Low latency at scale.",
            response="DynamoDB offers single-digit ms latency.",
        )
        assert result["faithfulness_score"] == 0.9
        assert result["relevance_score"] == 0.85
        assert result["faithfulness_reason"] == "Well grounded in context"
        assert result["relevance_reason"] == "Directly addresses the question"
        assert result["judge_model"] == DEFAULT_JUDGE_MODEL

    def test_custom_judge_model(self):
        client = _make_mock_client(_valid_scores_json())
        result = evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
            judge_model="gpt-4o",
        )
        assert result["judge_model"] == "gpt-4o"
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_custom_temperature(self):
        client = _make_mock_client(_valid_scores_json())
        evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
            temperature=0.5,
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_default_temperature_is_zero(self):
        client = _make_mock_client(_valid_scores_json())
        evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0

    def test_strips_markdown_code_fences(self):
        fenced = f"```json\n{_valid_scores_json()}\n```"
        client = _make_mock_client(fenced)
        result = evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        assert result["faithfulness_score"] == 0.9

    def test_strips_bare_code_fences(self):
        fenced = f"```\n{_valid_scores_json()}\n```"
        client = _make_mock_client(fenced)
        result = evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        assert result["relevance_score"] == 0.85

    def test_perfect_scores(self):
        client = _make_mock_client(_valid_scores_json(
            faithfulness_score=1.0, relevance_score=1.0
        ))
        result = evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        assert result["faithfulness_score"] == 1.0
        assert result["relevance_score"] == 1.0

    def test_zero_scores(self):
        client = _make_mock_client(_valid_scores_json(
            faithfulness_score=0.0, relevance_score=0.0,
            faithfulness_reason="Hallucinated", relevance_reason="Off-topic",
        ))
        result = evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        assert result["faithfulness_score"] == 0.0
        assert result["relevance_score"] == 0.0

    def test_malformed_json_raises(self):
        client = _make_mock_client("not valid json at all")
        with pytest.raises(json.JSONDecodeError):
            evaluate_response(
                client=client,
                context="ctx", question="q", expected_response="exp", response="resp",
            )

    def test_missing_keys_raises(self):
        client = _make_mock_client('{"faithfulness_score": 0.5}')
        with pytest.raises(KeyError):
            evaluate_response(
                client=client,
                context="ctx", question="q", expected_response="exp", response="resp",
            )

    def test_scores_cast_to_float(self):
        client = _make_mock_client(_valid_scores_json(faithfulness_score=1, relevance_score=0))
        result = evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        assert isinstance(result["faithfulness_score"], float)
        assert isinstance(result["relevance_score"], float)

    def test_system_prompt_passed_correctly(self):
        client = _make_mock_client(_valid_scores_json())
        evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == JUDGE_SYSTEM_PROMPT

    def test_user_template_includes_all_fields(self):
        client = _make_mock_client(_valid_scores_json())
        evaluate_response(
            client=client,
            context="my context",
            question="my question",
            expected_response="my expected",
            response="my response",
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert "my context" in user_msg
        assert "my question" in user_msg
        assert "my expected" in user_msg
        assert "my response" in user_msg

    def test_max_tokens_is_300(self):
        client = _make_mock_client(_valid_scores_json())
        evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="resp",
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 300

    def test_api_error_propagates(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError, match="API down"):
            evaluate_response(
                client=client,
                context="ctx", question="q", expected_response="exp", response="resp",
            )

    def test_empty_response_text(self):
        client = _make_mock_client(_valid_scores_json())
        result = evaluate_response(
            client=client,
            context="ctx", question="q", expected_response="exp", response="",
        )
        assert result["faithfulness_score"] == 0.9


# ---------------------------------------------------------------------------
# _get_openai_client
# ---------------------------------------------------------------------------


class TestGetOpenaiClient:
    def test_missing_sdk_raises_import_error(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="pip install openai"):
                _get_openai_client()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_judge_model(self):
        assert DEFAULT_JUDGE_MODEL == "gpt-4o-mini"

    def test_judge_system_prompt_mentions_faithfulness(self):
        assert "Faithfulness" in JUDGE_SYSTEM_PROMPT

    def test_judge_system_prompt_mentions_relevance(self):
        assert "Relevance" in JUDGE_SYSTEM_PROMPT

    def test_judge_user_template_has_placeholders(self):
        assert "{context}" in JUDGE_USER_TEMPLATE
        assert "{question}" in JUDGE_USER_TEMPLATE
        assert "{expected_response}" in JUDGE_USER_TEMPLATE
        assert "{response}" in JUDGE_USER_TEMPLATE
