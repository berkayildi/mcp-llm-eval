"""Tests for the eval engine: dataset loading, orchestration, aggregation, thresholds."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_eval.engine import (
    _aggregate,
    _get_client,
    _get_runner,
    check_thresholds,
    load_dataset,
    run_evaluation,
)
from mcp_llm_eval.types import (
    EvalEntry,
    EvalResult,
    MetricCheck,
    ModelConfig,
    RunSummary,
    ThresholdConfig,
    ThresholdResult,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DATASET = FIXTURES_DIR / "sample_dataset.json"


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_load_sample_dataset(self):
        entries = load_dataset(str(SAMPLE_DATASET))
        assert len(entries) == 3
        assert all(isinstance(e, EvalEntry) for e in entries)

    def test_first_entry_fields(self):
        entries = load_dataset(str(SAMPLE_DATASET))
        e = entries[0]
        assert e.id == "adr-001"
        assert e.category == "factual"
        assert len(e.context) > 0
        assert len(e.question) > 0
        assert len(e.expected_response) > 0
        assert isinstance(e.tags, list)

    def test_all_categorys(self):
        entries = load_dataset(str(SAMPLE_DATASET))
        types = {e.category for e in entries}
        assert types == {"factual", "reasoning", "summarization"}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/dataset.json")

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            try:
                with pytest.raises(Exception):
                    load_dataset(f.name)
            finally:
                os.unlink(f.name)

    def test_not_array(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            try:
                with pytest.raises(ValueError, match="JSON array"):
                    load_dataset(f.name)
            finally:
                os.unlink(f.name)

    def test_empty_array(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            f.flush()
            try:
                with pytest.raises(ValueError, match="empty"):
                    load_dataset(f.name)
            finally:
                os.unlink(f.name)

    def test_missing_required_keys(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"id": "x", "category": "adr"}], f)
            f.flush()
            try:
                with pytest.raises(ValueError, match="missing required"):
                    load_dataset(f.name)
            finally:
                os.unlink(f.name)

    def test_tags_optional(self):
        data = [{
            "id": "t1", "category": "adr",
            "context": "c", "question": "q", "expected_response": "r",
        }]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            try:
                entries = load_dataset(f.name)
                assert entries[0].tags == []
            finally:
                os.unlink(f.name)

    def test_multiple_entries_with_tags(self):
        data = [
            {
                "id": f"t{i}", "category": "adr",
                "context": "c", "question": "q", "expected_response": "r",
                "tags": [f"tag{i}"],
            }
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            try:
                entries = load_dataset(f.name)
                assert len(entries) == 5
                assert entries[2].tags == ["tag2"]
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_empty_list(self):
        assert _aggregate([], "p", "m") is None

    def test_single_result(self):
        r = EvalResult(
            eval_id="e1", category="adr", model="m", provider="p",
            input_tokens=100, output_tokens=50,
            time_to_first_token_ms=150, total_latency_ms=2000,
            cost_per_query=0.001, faithfulness_score=0.9, relevance_score=0.8,
        )
        agg = _aggregate([r], "p", "m")
        assert agg.runs == 1
        assert agg.avg_ttft_ms == 150.0
        assert agg.avg_latency_ms == 2000.0
        assert agg.avg_faithfulness == 0.9
        assert agg.avg_relevance == 0.8

    def test_multiple_results(self):
        results = [
            EvalResult(
                eval_id=f"e{i}", category="adr", model="m", provider="p",
                input_tokens=100, output_tokens=50,
                time_to_first_token_ms=100 + i * 50, total_latency_ms=1000 + i * 500,
                cost_per_query=0.001,
                faithfulness_score=0.8 + i * 0.05, relevance_score=0.7 + i * 0.1,
            )
            for i in range(3)
        ]
        agg = _aggregate(results, "p", "m")
        assert agg.runs == 3
        assert agg.avg_ttft_ms == 150.0  # (100+150+200)/3

    def test_no_scores(self):
        r = EvalResult(
            eval_id="e1", category="adr", model="m", provider="p",
            input_tokens=100, output_tokens=50,
            total_latency_ms=1000, cost_per_query=0.001,
        )
        agg = _aggregate([r], "p", "m")
        assert agg.avg_faithfulness is None
        assert agg.avg_relevance is None

    def test_mixed_scored_unscored(self):
        r1 = EvalResult(
            eval_id="e1", category="adr", model="m", provider="p",
            input_tokens=100, output_tokens=50, total_latency_ms=1000,
            cost_per_query=0.001, faithfulness_score=0.9, relevance_score=0.8,
        )
        r2 = EvalResult(
            eval_id="e2", category="adr", model="m", provider="p",
            input_tokens=100, output_tokens=50, total_latency_ms=1000,
            cost_per_query=0.001,
        )
        agg = _aggregate([r1, r2], "p", "m")
        assert agg.runs == 2
        assert agg.avg_faithfulness == 0.9  # only r1 scored

    def test_null_ttft_treated_as_zero(self):
        r = EvalResult(
            eval_id="e1", category="adr", model="m", provider="p",
            input_tokens=100, output_tokens=50, total_latency_ms=1000,
            cost_per_query=0.001,
        )
        agg = _aggregate([r], "p", "m")
        assert agg.avg_ttft_ms == 0.0


# ---------------------------------------------------------------------------
# _get_runner
# ---------------------------------------------------------------------------


class TestGetRunner:
    def test_anthropic(self):
        mod = _get_runner("anthropic")
        assert hasattr(mod, "run")

    def test_openai(self):
        mod = _get_runner("openai")
        assert hasattr(mod, "run")

    def test_google(self):
        mod = _get_runner("google")
        assert hasattr(mod, "run")

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _get_runner("bedrock")


# ---------------------------------------------------------------------------
# check_thresholds
# ---------------------------------------------------------------------------


class TestCheckThresholds:
    def _make_summary(self, overall=None, results=None):
        return RunSummary(
            timestamp="20250101_000000",
            total_questions=3,
            total_model_runs=6,
            total_errors=0,
            total_elapsed_sec=30.0,
            total_estimated_cost=0.01,
            judge_model="gpt-4o-mini",
            overall=overall or {},
            results=results or [],
        )

    def test_all_pass(self):
        summary = self._make_summary(
            overall={
                "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85},
            },
            results=[
                {"eval_id": "e1", "category": "adr", "model": "m", "provider": "p",
                 "time_to_first_token_ms": 100, "cost_per_query": 0.001},
                {"eval_id": "e2", "category": "adr", "model": "m", "provider": "p",
                 "time_to_first_token_ms": 200, "cost_per_query": 0.002},
            ],
        )
        thresholds = ThresholdConfig(
            avg_faithfulness=0.8, avg_relevance=0.7,
            p95_ttft_ms=500, max_cost_per_query=0.01,
        )
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is True
        assert len(result.per_metric) == 4

    def test_faithfulness_fail(self):
        summary = self._make_summary(
            overall={"model-a": {"avg_faithfulness": 0.5, "avg_relevance": 0.9}},
        )
        thresholds = ThresholdConfig(avg_faithfulness=0.8)
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is False
        assert result.per_metric[0].metric == "avg_faithfulness"
        assert result.per_metric[0].passed is False

    def test_relevance_fail(self):
        summary = self._make_summary(
            overall={"model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.3}},
        )
        thresholds = ThresholdConfig(avg_relevance=0.7)
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is False

    def test_p95_ttft_fail(self):
        summary = self._make_summary(
            results=[
                {"eval_id": f"e{i}", "category": "adr", "model": "m", "provider": "p",
                 "time_to_first_token_ms": 100 + i * 100, "cost_per_query": 0.001}
                for i in range(20)
            ],
        )
        thresholds = ThresholdConfig(p95_ttft_ms=500)
        result = check_thresholds(summary, thresholds)
        p95_check = [c for c in result.per_metric if c.metric == "p95_ttft_ms"][0]
        # p95 of [100,200,...,2000] = values[19*0.95] = values[18] = 1900
        assert p95_check.actual > 500
        assert p95_check.passed is False

    def test_cost_fail(self):
        summary = self._make_summary(
            results=[
                {"eval_id": "e1", "category": "adr", "model": "m", "provider": "p",
                 "cost_per_query": 0.05},
            ],
        )
        thresholds = ThresholdConfig(max_cost_per_query=0.01)
        result = check_thresholds(summary, thresholds)
        cost_check = [c for c in result.per_metric if c.metric == "max_cost_per_query"][0]
        assert cost_check.passed is False

    def test_no_thresholds_passes(self):
        summary = self._make_summary()
        thresholds = ThresholdConfig()
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is True
        assert len(result.per_metric) == 0

    def test_partial_thresholds(self):
        summary = self._make_summary(
            overall={"m": {"avg_faithfulness": 0.9, "avg_relevance": 0.8}},
        )
        thresholds = ThresholdConfig(avg_faithfulness=0.85)
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is True
        assert len(result.per_metric) == 1

    def test_multiple_models_averaged(self):
        summary = self._make_summary(
            overall={
                "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.8},
                "model-b": {"avg_faithfulness": 0.7, "avg_relevance": 0.6},
            },
        )
        thresholds = ThresholdConfig(avg_faithfulness=0.8)
        result = check_thresholds(summary, thresholds)
        faith_check = result.per_metric[0]
        assert faith_check.actual == 0.8  # (0.9+0.7)/2
        assert faith_check.passed is True  # 0.8 >= 0.8

    def test_error_results_excluded(self):
        summary = self._make_summary(
            results=[
                {"eval_id": "e1", "category": "adr", "model": "m", "provider": "p",
                 "cost_per_query": 0.001},
                {"eval_id": "e2", "category": "adr", "model": "m", "provider": "p",
                 "error": "timeout", "cost_per_query": 0.0},
            ],
        )
        thresholds = ThresholdConfig(max_cost_per_query=0.01)
        result = check_thresholds(summary, thresholds)
        cost_check = [c for c in result.per_metric if c.metric == "max_cost_per_query"][0]
        assert cost_check.actual == 0.001  # error result excluded

    def test_no_ttft_values(self):
        summary = self._make_summary(results=[])
        thresholds = ThresholdConfig(p95_ttft_ms=500)
        result = check_thresholds(summary, thresholds)
        p95_check = [c for c in result.per_metric if c.metric == "p95_ttft_ms"][0]
        assert p95_check.actual == 0
        assert p95_check.passed is True

    def test_boundary_threshold_exact(self):
        summary = self._make_summary(
            overall={"m": {"avg_faithfulness": 0.8}},
        )
        thresholds = ThresholdConfig(avg_faithfulness=0.8)
        result = check_thresholds(summary, thresholds)
        assert result.per_metric[0].passed is True

    def test_result_serialization(self):
        summary = self._make_summary(
            overall={"m": {"avg_faithfulness": 0.9}},
        )
        thresholds = ThresholdConfig(avg_faithfulness=0.8)
        result = check_thresholds(summary, thresholds)
        d = result.to_dict()
        assert "overall_pass" in d
        assert "per_metric" in d
        assert d["per_metric"][0]["metric"] == "avg_faithfulness"


# ---------------------------------------------------------------------------
# run_evaluation (mocked)
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def _make_dataset(self):
        return [
            EvalEntry(
                id="e1", category="adr", context="ctx",
                question="q", expected_response="r",
            ),
        ]

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    def test_basic_run(self, mock_get_client, mock_get_runner, mock_judge_mod):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        runner_mod = MagicMock()
        runner_mod.run.return_value = {
            "response": "answer", "input_tokens": 100, "output_tokens": 50,
            "stop_reason": "end_turn", "time_to_first_token_ms": 150,
            "total_latency_ms": 2000,
        }
        mock_get_runner.return_value = runner_mod

        mock_judge_mod.DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
        mock_judge_mod.evaluate_response.return_value = {
            "faithfulness_score": 0.9, "faithfulness_reason": "good",
            "relevance_score": 0.85, "relevance_reason": "relevant",
            "judge_model": "gpt-4o-mini",
        }

        dataset = self._make_dataset()
        models = [ModelConfig(provider="openai", model="gpt-4o")]

        summary = run_evaluation(dataset, models)
        assert isinstance(summary, RunSummary)
        assert summary.total_questions == 1
        assert summary.total_model_runs == 1
        assert len(summary.results) == 1

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    def test_saves_output(self, mock_get_client, mock_get_runner, mock_judge_mod):
        mock_get_client.return_value = MagicMock()
        runner_mod = MagicMock()
        runner_mod.run.return_value = {
            "response": "ans", "input_tokens": 10, "output_tokens": 5,
            "stop_reason": "stop", "time_to_first_token_ms": 50,
            "total_latency_ms": 500,
        }
        mock_get_runner.return_value = runner_mod
        mock_judge_mod.DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
        mock_judge_mod.evaluate_response.return_value = {
            "faithfulness_score": 0.9, "faithfulness_reason": "ok",
            "relevance_score": 0.8, "relevance_reason": "ok",
            "judge_model": "gpt-4o-mini",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._make_dataset()
            models = [ModelConfig(provider="openai", model="gpt-4o")]
            summary = run_evaluation(dataset, models, output_dir=tmpdir)

            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 2  # summary + benchmark
            names = {f.name for f in files}
            assert any("summary" in n for n in names)
            assert any("benchmark" in n for n in names)

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    def test_provider_error_captured(self, mock_get_client, mock_get_runner, mock_judge_mod):
        mock_get_client.return_value = MagicMock()
        runner_mod = MagicMock()
        runner_mod.run.side_effect = RuntimeError("API error")
        mock_get_runner.return_value = runner_mod
        mock_judge_mod.DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
        mock_judge_mod.evaluate_response.return_value = {
            "faithfulness_score": 0.5, "faithfulness_reason": "ok",
            "relevance_score": 0.5, "relevance_reason": "ok",
            "judge_model": "gpt-4o-mini",
        }

        dataset = self._make_dataset()
        models = [ModelConfig(provider="openai", model="gpt-4o")]
        summary = run_evaluation(dataset, models)
        assert summary.total_errors == 1
        assert summary.results[0]["error"] == "API error"

    @patch("mcp_llm_eval.engine._get_client")
    def test_missing_provider_sdk(self, mock_get_client):
        mock_get_client.side_effect = ImportError("pip install openai")
        dataset = self._make_dataset()
        models = [ModelConfig(provider="openai", model="gpt-4o")]
        with pytest.raises(ImportError, match="pip install openai"):
            run_evaluation(dataset, models)

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    def test_cost_calculation(self, mock_get_client, mock_get_runner, mock_judge_mod):
        mock_get_client.return_value = MagicMock()
        runner_mod = MagicMock()
        runner_mod.run.return_value = {
            "response": "ans", "input_tokens": 1000, "output_tokens": 500,
            "stop_reason": "stop", "time_to_first_token_ms": 100,
            "total_latency_ms": 1000,
        }
        mock_get_runner.return_value = runner_mod
        mock_judge_mod.DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
        mock_judge_mod.evaluate_response.return_value = {
            "faithfulness_score": 0.9, "faithfulness_reason": "ok",
            "relevance_score": 0.8, "relevance_reason": "ok",
            "judge_model": "gpt-4o-mini",
        }

        dataset = self._make_dataset()
        models = [ModelConfig(
            provider="openai", model="gpt-4o",
            input_cost_per_mtok=2.5, output_cost_per_mtok=10.0,
        )]
        summary = run_evaluation(dataset, models)
        # cost = (1000 * 2.5 + 500 * 10.0) / 1_000_000 = 0.0075
        result = summary.results[0]
        assert abs(result["cost_per_query"] - 0.0075) < 0.0001
