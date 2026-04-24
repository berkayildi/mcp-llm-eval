"""Tests for judge scoring with mocked OpenAI client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_eval.judge import (
    CITATION_FAITHFULNESS_SYSTEM_PROMPT,
    CONTEXT_RELEVANCE_SYSTEM_PROMPT,
    DEFAULT_JUDGE_MODEL,
    JUDGE_MODEL_ENV_VAR,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    _get_openai_client,
    _parse_judge_json,
    _resolve_judge_model,
    _scale_1_to_5,
    evaluate_response,
    judge_citation_faithfulness,
    judge_context_relevance,
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


# ---------------------------------------------------------------------------
# v0.5.0 helpers and judges
# ---------------------------------------------------------------------------


def _make_judge_score_json(score: int, reason: str = "ok") -> str:
    return json.dumps({"score": score, "reason": reason})


class TestParseJudgeJson:
    def test_unfenced(self):
        raw = '{"score": 3, "reason": "fine"}'
        assert _parse_judge_json(raw) == {"score": 3, "reason": "fine"}

    def test_fenced_with_language(self):
        raw = '```json\n{"score": 4, "reason": "ok"}\n```'
        assert _parse_judge_json(raw) == {"score": 4, "reason": "ok"}

    def test_fenced_bare(self):
        raw = '```\n{"score": 5, "reason": "ok"}\n```'
        assert _parse_judge_json(raw) == {"score": 5, "reason": "ok"}

    def test_malformed_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_judge_json("not json")

    def test_empty_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_judge_json("")


class TestScale1To5:
    def test_one_maps_to_zero(self):
        assert _scale_1_to_5(1) == 0.0

    def test_two_maps_to_025(self):
        assert _scale_1_to_5(2) == 0.25

    def test_three_maps_to_half(self):
        assert _scale_1_to_5(3) == 0.5

    def test_four_maps_to_075(self):
        assert _scale_1_to_5(4) == 0.75

    def test_five_maps_to_one(self):
        assert _scale_1_to_5(5) == 1.0

    def test_float_25_maps_to_0375(self):
        assert _scale_1_to_5(2.5) == 0.375

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _scale_1_to_5(0)

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _scale_1_to_5(6)


class TestResolveJudgeModel:
    def test_explicit_arg_wins(self, monkeypatch):
        monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, "env-model")
        assert _resolve_judge_model("explicit") == "explicit"

    def test_env_var_used_when_no_arg(self, monkeypatch):
        monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, "env-model")
        assert _resolve_judge_model(None) == "env-model"

    def test_default_when_no_arg_no_env(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
        assert _resolve_judge_model(None) == DEFAULT_JUDGE_MODEL

    def test_empty_string_env_falls_through(self, monkeypatch):
        monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, "")
        assert _resolve_judge_model(None) == DEFAULT_JUDGE_MODEL


class TestJudgeContextRelevance:
    def test_basic_call(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
        client = _make_mock_client(_make_judge_score_json(4, "clearly on-topic"))
        result = judge_context_relevance(client, "what is X?", "X is a thing.")
        assert result["score"] == 0.75
        assert result["raw_score"] == 4
        assert result["reason"] == "clearly on-topic"
        assert result["judge_model"] == DEFAULT_JUDGE_MODEL

    def test_score_in_0_1_range(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
        for raw_score in (1, 2, 3, 4, 5):
            client = _make_mock_client(_make_judge_score_json(raw_score))
            result = judge_context_relevance(client, "q", "c")
            assert 0.0 <= result["score"] <= 1.0

    def test_model_override_via_arg(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
        client = _make_mock_client(_make_judge_score_json(3))
        result = judge_context_relevance(client, "q", "c", judge_model="gpt-4o")
        assert result["judge_model"] == "gpt-4o"
        assert client.chat.completions.create.call_args[1]["model"] == "gpt-4o"

    def test_model_override_via_env_var(self, monkeypatch):
        monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, "env-judge")
        client = _make_mock_client(_make_judge_score_json(3))
        result = judge_context_relevance(client, "q", "c")
        assert result["judge_model"] == "env-judge"

    def test_temperature_zero(self):
        client = _make_mock_client(_make_judge_score_json(3))
        judge_context_relevance(client, "q", "c")
        assert client.chat.completions.create.call_args[1]["temperature"] == 0

    def test_max_tokens_150(self):
        client = _make_mock_client(_make_judge_score_json(3))
        judge_context_relevance(client, "q", "c")
        assert client.chat.completions.create.call_args[1]["max_tokens"] == 150

    def test_uses_context_relevance_system_prompt(self):
        client = _make_mock_client(_make_judge_score_json(3))
        judge_context_relevance(client, "q", "c")
        msgs = client.chat.completions.create.call_args[1]["messages"]
        assert msgs[0]["content"] == CONTEXT_RELEVANCE_SYSTEM_PROMPT

    def test_malformed_json_propagates(self):
        client = _make_mock_client("not json")
        with pytest.raises(json.JSONDecodeError):
            judge_context_relevance(client, "q", "c")

    def test_out_of_range_score_raises(self):
        client = _make_mock_client(_make_judge_score_json(6))
        with pytest.raises(ValueError, match="out of range"):
            judge_context_relevance(client, "q", "c")


class TestJudgeCitationFaithfulness:
    def test_basic_call(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
        client = _make_mock_client(_make_judge_score_json(5, "fully supported"))
        result = judge_citation_faithfulness(
            client, "answer text", ["chunk A", "chunk B"],
        )
        assert result["score"] == 1.0
        assert result["raw_score"] == 5
        assert result["judge_model"] == DEFAULT_JUDGE_MODEL

    def test_empty_chunks_short_circuits(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
        client = MagicMock()
        result = judge_citation_faithfulness(client, "answer", [])
        assert result == {
            "score": 0.0,
            "reason": "No chunks cited",
            "raw_score": 1,
            "judge_model": DEFAULT_JUDGE_MODEL,
        }
        client.chat.completions.create.assert_not_called()

    def test_empty_chunks_uses_env_var_model(self, monkeypatch):
        monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, "env-m")
        client = MagicMock()
        result = judge_citation_faithfulness(client, "answer", [])
        assert result["judge_model"] == "env-m"
        client.chat.completions.create.assert_not_called()

    def test_chunks_rendered_with_separator(self):
        client = _make_mock_client(_make_judge_score_json(3))
        judge_citation_faithfulness(client, "ans", ["first", "second", "third"])
        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "[chunk 1]\nfirst" in user_msg
        assert "[chunk 2]\nsecond" in user_msg
        assert "[chunk 3]\nthird" in user_msg
        assert "\n\n---\n\n" in user_msg

    def test_max_tokens_200(self):
        client = _make_mock_client(_make_judge_score_json(3))
        judge_citation_faithfulness(client, "a", ["c"])
        assert client.chat.completions.create.call_args[1]["max_tokens"] == 200

    def test_temperature_zero(self):
        client = _make_mock_client(_make_judge_score_json(3))
        judge_citation_faithfulness(client, "a", ["c"])
        assert client.chat.completions.create.call_args[1]["temperature"] == 0

    def test_uses_citation_faithfulness_system_prompt(self):
        client = _make_mock_client(_make_judge_score_json(3))
        judge_citation_faithfulness(client, "a", ["c"])
        msgs = client.chat.completions.create.call_args[1]["messages"]
        assert msgs[0]["content"] == CITATION_FAITHFULNESS_SYSTEM_PROMPT

    def test_model_override_via_arg(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
        client = _make_mock_client(_make_judge_score_json(3))
        result = judge_citation_faithfulness(
            client, "a", ["c"], judge_model="gpt-4o",
        )
        assert result["judge_model"] == "gpt-4o"

    def test_malformed_json_propagates(self):
        client = _make_mock_client("not json")
        with pytest.raises(json.JSONDecodeError):
            judge_citation_faithfulness(client, "a", ["c"])
