"""Tests for the eval engine: dataset loading, orchestration, aggregation, thresholds."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_eval.engine import (
    _aggregate,
    _build_retrieval_adapter,
    _format_rag_system_prompt,
    _get_client,
    _get_runner,
    _percentiles,
    check_retrieval_drift,
    check_thresholds,
    load_dataset,
    load_jsonl_dataset,
    run_evaluation,
    run_rag_evaluation,
    run_retrieval_evaluation,
)
from mcp_llm_eval.types import (
    EvalEntry,
    EvalResult,
    MetricCheck,
    ModelConfig,
    RetrievedChunk,
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
        assert e.id == "entry-001"
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
            json.dump([{"id": "x", "category": "factual"}], f)
            f.flush()
            try:
                with pytest.raises(ValueError, match="missing required"):
                    load_dataset(f.name)
            finally:
                os.unlink(f.name)

    def test_tags_optional(self):
        data = [{
            "id": "t1", "category": "factual",
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
                "id": f"t{i}", "category": "factual",
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
            eval_id="e1", category="factual", model="m", provider="p",
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
                eval_id=f"e{i}", category="factual", model="m", provider="p",
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
            eval_id="e1", category="factual", model="m", provider="p",
            input_tokens=100, output_tokens=50,
            total_latency_ms=1000, cost_per_query=0.001,
        )
        agg = _aggregate([r], "p", "m")
        assert agg.avg_faithfulness is None
        assert agg.avg_relevance is None

    def test_mixed_scored_unscored(self):
        r1 = EvalResult(
            eval_id="e1", category="factual", model="m", provider="p",
            input_tokens=100, output_tokens=50, total_latency_ms=1000,
            cost_per_query=0.001, faithfulness_score=0.9, relevance_score=0.8,
        )
        r2 = EvalResult(
            eval_id="e2", category="factual", model="m", provider="p",
            input_tokens=100, output_tokens=50, total_latency_ms=1000,
            cost_per_query=0.001,
        )
        agg = _aggregate([r1, r2], "p", "m")
        assert agg.runs == 2
        assert agg.avg_faithfulness == 0.9  # only r1 scored

    def test_null_ttft_treated_as_zero(self):
        r = EvalResult(
            eval_id="e1", category="factual", model="m", provider="p",
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
                {"eval_id": "e1", "category": "factual", "model": "m", "provider": "p",
                 "time_to_first_token_ms": 100, "cost_per_query": 0.001},
                {"eval_id": "e2", "category": "factual", "model": "m", "provider": "p",
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
                {"eval_id": f"e{i}", "category": "factual", "model": "m", "provider": "p",
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
                {"eval_id": "e1", "category": "factual", "model": "m", "provider": "p",
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
                {"eval_id": "e1", "category": "factual", "model": "m", "provider": "p",
                 "cost_per_query": 0.001},
                {"eval_id": "e2", "category": "factual", "model": "m", "provider": "p",
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
                id="e1", category="factual", context="ctx",
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


# ---------------------------------------------------------------------------
# v0.5.0 — JSONL dataset loader
# ---------------------------------------------------------------------------


class TestLoadJsonlDataset:
    def _write_jsonl(self, lines: list[dict]) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for line in lines:
            f.write(json.dumps(line) + "\n")
        f.flush()
        f.close()
        return f.name

    def test_loads_entries(self):
        path = self._write_jsonl([
            {"id": "r1", "category": "factual", "context": "c", "question": "q?",
             "expected_response": "r", "relevant_chunk_ids": ["a"]},
            {"id": "r2", "category": "factual", "context": "c", "question": "q?",
             "expected_response": "r"},
        ])
        try:
            entries = load_jsonl_dataset(path)
            assert len(entries) == 2
            assert entries[0].relevant_chunk_ids == ["a"]
            assert entries[1].relevant_chunk_ids is None
        finally:
            os.unlink(path)

    def test_blank_lines_skipped(self):
        path = self._write_jsonl([
            {"id": "r1", "category": "factual", "context": "c", "question": "q?",
             "expected_response": "r"},
        ])
        # Append a blank line
        with open(path, "a") as f:
            f.write("\n   \n")
        try:
            entries = load_jsonl_dataset(path)
            assert len(entries) == 1
        finally:
            os.unlink(path)

    def test_missing_required_keys(self):
        path = self._write_jsonl([{"id": "r1", "category": "factual"}])
        try:
            with pytest.raises(ValueError, match="Line 1: missing required"):
                load_jsonl_dataset(path)
        finally:
            os.unlink(path)

    def test_malformed_line(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"id": "r1", "category": "factual", "context": "c", '
                    '"question": "q?", "expected_response": "r"}\n')
            f.write("not json\n")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Line 2:"):
                    load_jsonl_dataset(f.name)
            finally:
                os.unlink(f.name)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.flush()
            try:
                with pytest.raises(ValueError, match="empty"):
                    load_jsonl_dataset(f.name)
            finally:
                os.unlink(f.name)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl_dataset("/nonexistent/dataset.jsonl")


# ---------------------------------------------------------------------------
# v0.5.0 — _build_retrieval_adapter, _percentiles, _format_rag_system_prompt
# ---------------------------------------------------------------------------


class TestBuildRetrievalAdapter:
    def test_unknown_adapter_raises(self):
        with pytest.raises(ValueError, match="Unsupported retrieval adapter"):
            _build_retrieval_adapter("faiss", "/some/path")


class TestPercentiles:
    def test_empty(self):
        assert _percentiles([]) == (0.0, 0.0)

    def test_single_value(self):
        assert _percentiles([42.0]) == (42.0, 42.0)

    def test_known_distribution(self):
        # statistics.quantiles inclusive on [10, 20, 30, 40, 50]
        # produces evenly-spaced cut points; just sanity-check ordering.
        p50, p95 = _percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
        assert 25.0 <= p50 <= 35.0
        assert p95 >= 45.0


class TestFormatRagSystemPrompt:
    def test_renders_chunks(self):
        chunks = [
            RetrievedChunk(chunk_id="c1", content="alpha", score=1.0),
            RetrievedChunk(chunk_id="c2", content="beta", score=0.5),
        ]
        prompt = _format_rag_system_prompt("Original context.", chunks)
        assert "Original context." in prompt
        assert "[chunk 1]\nalpha" in prompt
        assert "[chunk 2]\nbeta" in prompt
        assert "Answer the question using only the retrieved context above." in prompt


# ---------------------------------------------------------------------------
# v0.5.0 — run_retrieval_evaluation
# ---------------------------------------------------------------------------


def _make_rag_dataset_file(entries: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for e in entries:
        f.write(json.dumps(e) + "\n")
    f.flush()
    f.close()
    return f.name


class _FakeAdapter:
    """Returns canned chunks per query keyed by question text."""

    def __init__(self, canned: dict[str, list[RetrievedChunk]]):
        self.canned = canned
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int):
        self.calls.append((query, k))
        return list(self.canned.get(query, []))[:k]


class TestRunRetrievalEvaluation:
    def _entries(self):
        return [
            {"id": "q1", "category": "factual", "context": "ctx",
             "question": "q1?", "expected_response": "r",
             "relevant_chunk_ids": ["c1"]},
            {"id": "q2", "category": "factual", "context": "ctx",
             "question": "q2?", "expected_response": "r",
             "relevant_chunk_ids": ["c2", "c3"]},
            {"id": "skip", "category": "factual", "context": "ctx",
             "question": "skipped?", "expected_response": "r"},
        ]

    @patch("mcp_llm_eval.engine._build_retrieval_adapter")
    def test_happy_path_with_skipping(self, mock_build, capsys):
        canned = {
            "q1?": [RetrievedChunk(chunk_id="c1", content="x", score=1.0)],
            "q2?": [
                RetrievedChunk(chunk_id="c2", content="x", score=1.0),
                RetrievedChunk(chunk_id="c3", content="x", score=0.5),
            ],
        }
        mock_build.return_value = _FakeAdapter(canned)

        path = _make_rag_dataset_file(self._entries())
        try:
            summary = run_retrieval_evaluation(
                dataset_path=path, corpus_path="ignored", k=2,
            )
        finally:
            os.unlink(path)

        assert summary["total_queries"] == 2
        assert summary["skipped"] == ["skip"]
        assert summary["aggregate"]["avg_recall_at_k"] == 1.0
        assert summary["aggregate"]["avg_precision_at_k"] == pytest.approx(0.75)
        assert summary["overall"]["_retrieval"]["avg_recall_at_k"] == 1.0
        # warning printed to stderr
        err = capsys.readouterr().err
        assert "skip" in err

    @patch("mcp_llm_eval.engine._build_retrieval_adapter")
    def test_no_relevant_entries_returns_error(self, mock_build):
        path = _make_rag_dataset_file([
            {"id": "x", "category": "factual", "context": "c",
             "question": "?", "expected_response": "r"},
        ])
        try:
            summary = run_retrieval_evaluation(
                dataset_path=path, corpus_path="ignored",
            )
        finally:
            os.unlink(path)
        assert "error" in summary
        mock_build.assert_not_called()

    def test_dataset_not_found(self):
        summary = run_retrieval_evaluation(
            dataset_path="/nonexistent.jsonl", corpus_path="ignored",
        )
        assert "error" in summary

    @patch("mcp_llm_eval.engine._build_retrieval_adapter")
    def test_writes_output_files(self, mock_build):
        canned = {
            "q1?": [RetrievedChunk(chunk_id="c1", content="x", score=1.0)],
        }
        mock_build.return_value = _FakeAdapter(canned)
        path = _make_rag_dataset_file([
            {"id": "q1", "category": "factual", "context": "c",
             "question": "q1?", "expected_response": "r",
             "relevant_chunk_ids": ["c1"]},
        ])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                run_retrieval_evaluation(
                    dataset_path=path, corpus_path="ignored", output_dir=tmpdir,
                )
                files = {f.name for f in Path(tmpdir).glob("*.json")}
                assert any("retrieval_summary" in n for n in files)
                assert any("retrieval_detail" in n for n in files)
                assert "latest_retrieval_summary.json" in files
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# v0.5.0 — run_rag_evaluation
# ---------------------------------------------------------------------------


class TestRunRagEvaluation:
    def _entries(self):
        return [
            {"id": "q1", "category": "factual", "context": "Ctx 1.",
             "question": "Where?", "expected_response": "Here.",
             "relevant_chunk_ids": ["c1"]},
        ]

    def _gen_response(self):
        return {
            "response": "Here.",
            "input_tokens": 100,
            "output_tokens": 20,
            "stop_reason": "stop",
            "time_to_first_token_ms": 80,
            "total_latency_ms": 400,
        }

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    @patch("mcp_llm_eval.engine._build_retrieval_adapter")
    def test_happy_path(
        self, mock_build, mock_get_client, mock_get_runner, mock_judge_mod,
    ):
        mock_build.return_value = _FakeAdapter({
            "Where?": [RetrievedChunk(chunk_id="c1", content="The answer is here.", score=1.0)],
        })
        mock_get_client.return_value = MagicMock()
        runner_mod = MagicMock()
        runner_mod.run.return_value = self._gen_response()
        mock_get_runner.return_value = runner_mod

        mock_judge_mod._resolve_judge_model.return_value = "gpt-4o-mini"
        mock_judge_mod.judge_context_relevance.return_value = {
            "score": 1.0, "reason": "good", "raw_score": 5, "judge_model": "gpt-4o-mini",
        }
        mock_judge_mod.judge_citation_faithfulness.return_value = {
            "score": 0.75, "reason": "mostly", "raw_score": 4, "judge_model": "gpt-4o-mini",
        }

        path = _make_rag_dataset_file(self._entries())
        try:
            summary = run_rag_evaluation(
                dataset_path=path, corpus_path="ignored",
                models=[ModelConfig(provider="openai", model="gpt-4o-mini")],
                k=1,
            )
        finally:
            os.unlink(path)

        assert summary["total_queries"] == 1
        assert summary["total_model_runs"] == 1
        assert summary["total_errors"] == 0
        rag = summary["per_query"][0]
        assert rag["answer"] == "Here."
        assert rag["context_relevance_score"] == 1.0
        assert rag["citation_faithfulness_score"] == 0.75
        agg = summary["overall"]["gpt-4o-mini"]
        assert agg["avg_recall_at_k"] == 1.0
        assert agg["avg_context_relevance"] == 1.0
        assert agg["avg_citation_faithfulness"] == 0.75

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    @patch("mcp_llm_eval.engine._build_retrieval_adapter")
    def test_provider_error_captured(
        self, mock_build, mock_get_client, mock_get_runner, mock_judge_mod,
    ):
        mock_build.return_value = _FakeAdapter({
            "Where?": [RetrievedChunk(chunk_id="c1", content="x", score=1.0)],
        })
        mock_get_client.return_value = MagicMock()
        runner_mod = MagicMock()
        runner_mod.run.side_effect = RuntimeError("API error")
        mock_get_runner.return_value = runner_mod
        mock_judge_mod._resolve_judge_model.return_value = "gpt-4o-mini"
        mock_judge_mod.judge_context_relevance.return_value = {
            "score": 0.5, "reason": "ok", "raw_score": 3, "judge_model": "gpt-4o-mini",
        }

        path = _make_rag_dataset_file(self._entries())
        try:
            summary = run_rag_evaluation(
                dataset_path=path, corpus_path="ignored",
                models=[ModelConfig(provider="openai", model="gpt-4o-mini")],
                k=1,
            )
        finally:
            os.unlink(path)

        assert summary["total_errors"] == 1
        assert "generation failed" in summary["per_query"][0]["error"]
        # errored result excluded from aggregate
        assert summary["overall"]["gpt-4o-mini"]["runs"] == 0

    @patch("mcp_llm_eval.engine._build_retrieval_adapter")
    def test_dataset_load_error_returns_error_dict(self, mock_build):
        summary = run_rag_evaluation(
            dataset_path="/nonexistent.jsonl", corpus_path="ignored",
            models=[ModelConfig(provider="openai", model="gpt-4o-mini")],
        )
        assert "error" in summary
        mock_build.assert_not_called()

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    @patch("mcp_llm_eval.engine._build_retrieval_adapter")
    def test_writes_rag_output_files(
        self, mock_build, mock_get_client, mock_get_runner, mock_judge_mod,
    ):
        mock_build.return_value = _FakeAdapter({
            "Where?": [RetrievedChunk(chunk_id="c1", content="x", score=1.0)],
        })
        mock_get_client.return_value = MagicMock()
        runner_mod = MagicMock()
        runner_mod.run.return_value = self._gen_response()
        mock_get_runner.return_value = runner_mod
        mock_judge_mod._resolve_judge_model.return_value = "gpt-4o-mini"
        mock_judge_mod.judge_context_relevance.return_value = {
            "score": 1.0, "reason": "ok", "raw_score": 5, "judge_model": "gpt-4o-mini",
        }
        mock_judge_mod.judge_citation_faithfulness.return_value = {
            "score": 1.0, "reason": "ok", "raw_score": 5, "judge_model": "gpt-4o-mini",
        }

        path = _make_rag_dataset_file(self._entries())
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                run_rag_evaluation(
                    dataset_path=path, corpus_path="ignored",
                    models=[ModelConfig(provider="openai", model="gpt-4o-mini")],
                    k=1, output_dir=tmpdir,
                )
                files = {f.name for f in Path(tmpdir).glob("*.json")}
                assert any("rag_summary" in n for n in files)
                assert any("rag_benchmark" in n for n in files)
                assert "latest_rag_summary.json" in files
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# v0.5.0 — check_retrieval_drift
# ---------------------------------------------------------------------------


class TestCheckRetrievalDrift:
    def _baseline(self):
        return {
            "timestamp": "20260101_000000",
            "aggregate": {
                "avg_recall_at_k": 0.80,
                "avg_precision_at_k": 0.60,
                "avg_mrr": 0.75,
                "avg_ndcg_at_k": 0.78,
                "p95_retrieval_latency_ms": 5.0,
            },
        }

    def test_no_regressions(self):
        baseline = self._baseline()
        current = {
            "timestamp": "20260102_000000",
            "aggregate": dict(baseline["aggregate"]),
        }
        result = check_retrieval_drift(baseline, current)
        assert result["has_regressions"] is False
        assert all(not m["regression"] for m in result["metrics"].values())

    def test_quality_regression_flagged(self):
        baseline = self._baseline()
        current = {
            "timestamp": "20260102_000000",
            "aggregate": {**baseline["aggregate"], "avg_recall_at_k": 0.65},
        }
        result = check_retrieval_drift(baseline, current)
        assert result["has_regressions"] is True
        assert result["metrics"]["avg_recall_at_k"]["regression"] is True

    def test_latency_regression_flagged(self):
        baseline = self._baseline()
        current = {
            "timestamp": "20260102_000000",
            "aggregate": {**baseline["aggregate"], "p95_retrieval_latency_ms": 200.0},
        }
        result = check_retrieval_drift(baseline, current)
        assert result["has_regressions"] is True
        assert result["metrics"]["p95_retrieval_latency_ms"]["regression"] is True

    def test_custom_tolerance(self):
        baseline = self._baseline()
        # 0.04 drop with default 0.05 tolerance: not a regression.
        current = {
            "timestamp": "20260102_000000",
            "aggregate": {**baseline["aggregate"], "avg_recall_at_k": 0.76},
        }
        loose = check_retrieval_drift(baseline, current)
        assert loose["has_regressions"] is False
        # Tighter tolerance flips it:
        tight = check_retrieval_drift(baseline, current, {"recall_at_k": 0.01})
        assert tight["has_regressions"] is True

    def test_works_with_overall_block(self):
        baseline = {
            "timestamp": "b",
            "overall": {"m1": {"avg_recall_at_k": 0.80, "p95_retrieval_latency_ms": 5.0}},
        }
        current = {
            "timestamp": "c",
            "overall": {"m1": {"avg_recall_at_k": 0.60, "p95_retrieval_latency_ms": 5.0}},
        }
        result = check_retrieval_drift(baseline, current)
        assert result["metrics"]["avg_recall_at_k"]["regression"] is True


# ---------------------------------------------------------------------------
# v0.5.0 — check_thresholds: retrieval / RAG metric extension
# ---------------------------------------------------------------------------


class TestCheckThresholdsRetrieval:
    def _summary(self, overall):
        return RunSummary(
            timestamp="t", total_questions=1, total_model_runs=1,
            total_errors=0, total_elapsed_sec=0.0, total_estimated_cost=0.0,
            overall=overall,
        )

    def test_recall_threshold_pass(self):
        summary = self._summary({"_retrieval": {"avg_recall_at_k": 0.90}})
        thresholds = ThresholdConfig(avg_recall_at_k=0.80)
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is True
        assert result.per_metric[0].metric == "avg_recall_at_k"

    def test_recall_threshold_fail(self):
        summary = self._summary({"_retrieval": {"avg_recall_at_k": 0.60}})
        thresholds = ThresholdConfig(avg_recall_at_k=0.80)
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is False

    def test_p95_retrieval_latency_fail(self):
        summary = self._summary({"_retrieval": {"p95_retrieval_latency_ms": 200.0}})
        thresholds = ThresholdConfig(p95_retrieval_latency_ms=50)
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is False
        assert result.per_metric[0].actual == 200.0

    def test_rag_thresholds_averaged_across_models(self):
        summary = self._summary({
            "model-a": {"avg_context_relevance": 0.90, "avg_citation_faithfulness": 0.80},
            "model-b": {"avg_context_relevance": 0.70, "avg_citation_faithfulness": 0.60},
        })
        thresholds = ThresholdConfig(avg_context_relevance=0.75)
        result = check_thresholds(summary, thresholds)
        # avg = (0.90 + 0.70) / 2 = 0.80 >= 0.75
        assert result.overall_pass is True

    def test_missing_metric_treated_as_zero(self):
        summary = self._summary({"_retrieval": {}})
        thresholds = ThresholdConfig(avg_mrr=0.5)
        result = check_thresholds(summary, thresholds)
        assert result.per_metric[0].actual == 0.0
        assert result.overall_pass is False

    def test_v04_thresholds_still_work(self):
        summary = self._summary({"m": {"avg_faithfulness": 0.9, "avg_relevance": 0.85}})
        thresholds = ThresholdConfig(avg_faithfulness=0.8)
        result = check_thresholds(summary, thresholds)
        assert result.overall_pass is True
        assert len(result.per_metric) == 1
