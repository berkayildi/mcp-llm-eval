"""Tests for type construction, validation, and serialization."""

import pytest

from mcp_llm_eval.types import (
    AggregateMetrics,
    EvalEntry,
    EvalResult,
    MetricCheck,
    ModelConfig,
    RunSummary,
    ThresholdConfig,
    ThresholdResult,
)


# ---------------------------------------------------------------------------
# EvalEntry
# ---------------------------------------------------------------------------


class TestEvalEntry:
    def test_from_dict_minimal(self):
        data = {
            "id": "test-001",
            "category": "adr",
            "context": "Some context",
            "question": "What happened?",
            "expected_response": "Something happened.",
        }
        entry = EvalEntry.from_dict(data)
        assert entry.id == "test-001"
        assert entry.category == "adr"
        assert entry.context == "Some context"
        assert entry.question == "What happened?"
        assert entry.expected_response == "Something happened."
        assert entry.tags == []

    def test_from_dict_with_tags(self):
        data = {
            "id": "test-002",
            "category": "sprint",
            "context": "Context",
            "question": "Question",
            "expected_response": "Response",
            "tags": ["tag1", "tag2"],
        }
        entry = EvalEntry.from_dict(data)
        assert entry.tags == ["tag1", "tag2"]

    def test_to_dict_roundtrip(self):
        data = {
            "id": "test-003",
            "category": "discovery",
            "context": "Context",
            "question": "Question",
            "expected_response": "Response",
            "tags": ["a"],
        }
        entry = EvalEntry.from_dict(data)
        result = entry.to_dict()
        assert result["id"] == "test-003"
        assert result["tags"] == ["a"]

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(KeyError):
            EvalEntry.from_dict({"id": "x"})

    def test_default_tags(self):
        entry = EvalEntry(
            id="x", category="adr", context="c", question="q", expected_response="r"
        )
        assert entry.tags == []

    def test_tags_are_independent_instances(self):
        e1 = EvalEntry(id="a", category="t", context="c", question="q", expected_response="r")
        e2 = EvalEntry(id="b", category="t", context="c", question="q", expected_response="r")
        e1.tags.append("x")
        assert e2.tags == []


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_from_dict_defaults(self):
        cfg = ModelConfig.from_dict({"provider": "openai", "model": "gpt-4o"})
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.max_tokens == 500
        assert cfg.input_cost_per_mtok == 0.0
        assert cfg.output_cost_per_mtok == 0.0

    def test_from_dict_custom(self):
        cfg = ModelConfig.from_dict({
            "provider": "anthropic",
            "model": "claude-3",
            "max_tokens": 1000,
            "input_cost_per_mtok": 3.0,
            "output_cost_per_mtok": 15.0,
        })
        assert cfg.max_tokens == 1000
        assert cfg.input_cost_per_mtok == 3.0
        assert cfg.output_cost_per_mtok == 15.0

    def test_to_dict(self):
        cfg = ModelConfig(provider="google", model="gemini-2.5-flash")
        d = cfg.to_dict()
        assert d["provider"] == "google"
        assert d["model"] == "gemini-2.5-flash"
        assert "max_tokens" in d

    def test_from_dict_roundtrip(self):
        original = {"provider": "openai", "model": "gpt-4o", "max_tokens": 800}
        cfg = ModelConfig.from_dict(original)
        d = cfg.to_dict()
        assert d["provider"] == "openai"
        assert d["max_tokens"] == 800


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_defaults(self):
        r = EvalResult(eval_id="e1", category="adr", model="m", provider="p")
        assert r.response is None
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.stop_reason is None
        assert r.time_to_first_token_ms is None
        assert r.total_latency_ms == 0
        assert r.cost_per_query == 0.0
        assert r.faithfulness_score is None
        assert r.relevance_score is None
        assert r.judge_model is None
        assert r.error is None

    def test_to_dict(self):
        r = EvalResult(
            eval_id="e1", category="adr", model="m", provider="p",
            response="hello", input_tokens=100, output_tokens=50,
        )
        d = r.to_dict()
        assert d["eval_id"] == "e1"
        assert d["response"] == "hello"
        assert d["input_tokens"] == 100

    def test_from_dict(self):
        data = {
            "eval_id": "e2",
            "category": "sprint",
            "model": "gpt-4o",
            "provider": "openai",
            "response": "world",
            "faithfulness_score": 0.9,
        }
        r = EvalResult.from_dict(data)
        assert r.eval_id == "e2"
        assert r.response == "world"
        assert r.faithfulness_score == 0.9

    def test_from_dict_ignores_extra_keys(self):
        data = {
            "eval_id": "e3",
            "category": "adr",
            "model": "m",
            "provider": "p",
            "unknown_key": "value",
        }
        r = EvalResult.from_dict(data)
        assert r.eval_id == "e3"

    def test_error_result(self):
        r = EvalResult(
            eval_id="e4", category="adr", model="m", provider="p",
            error="connection timeout",
        )
        assert r.error == "connection timeout"
        assert r.response is None

    def test_with_judge_scores(self):
        r = EvalResult(
            eval_id="e5", category="discovery", model="m", provider="p",
            faithfulness_score=0.85, faithfulness_reason="well grounded",
            relevance_score=0.95, relevance_reason="directly answers",
            judge_model="gpt-4o-mini",
        )
        d = r.to_dict()
        assert d["faithfulness_score"] == 0.85
        assert d["relevance_reason"] == "directly answers"
        assert d["judge_model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# AggregateMetrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_construction(self):
        agg = AggregateMetrics(
            provider="openai", model="gpt-4o", runs=10,
            avg_ttft_ms=150.0, avg_latency_ms=2000.0,
        )
        assert agg.runs == 10
        assert agg.avg_ttft_ms == 150.0

    def test_to_dict(self):
        agg = AggregateMetrics(provider="p", model="m", runs=5)
        d = agg.to_dict()
        assert d["runs"] == 5
        assert d["avg_faithfulness"] is None

    def test_with_scores(self):
        agg = AggregateMetrics(
            provider="p", model="m", runs=3,
            avg_faithfulness=0.9, avg_relevance=0.85,
        )
        assert agg.avg_faithfulness == 0.9
        assert agg.avg_relevance == 0.85


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_minimal(self):
        s = RunSummary(
            timestamp="20250101_000000",
            total_questions=10,
            total_model_runs=30,
            total_errors=0,
            total_elapsed_sec=120.0,
            total_estimated_cost=0.05,
        )
        assert s.timestamp == "20250101_000000"
        assert s.overall == {}
        assert s.results == []

    def test_to_dict(self):
        s = RunSummary(
            timestamp="20250101_000000",
            total_questions=5,
            total_model_runs=10,
            total_errors=1,
            total_elapsed_sec=60.0,
            total_estimated_cost=0.01,
            judge_model="gpt-4o-mini",
        )
        d = s.to_dict()
        assert d["judge_model"] == "gpt-4o-mini"
        assert d["total_errors"] == 1

    def test_from_dict(self):
        data = {
            "timestamp": "20250101_000000",
            "total_questions": 5,
            "total_model_runs": 10,
            "total_errors": 0,
            "total_elapsed_sec": 30.0,
            "total_estimated_cost": 0.02,
            "judge_model": "gpt-4o-mini",
            "overall": {"model-a": {"runs": 5}},
            "by_category": {},
            "results": [{"eval_id": "e1"}],
        }
        s = RunSummary.from_dict(data)
        assert s.total_questions == 5
        assert s.overall == {"model-a": {"runs": 5}}
        assert len(s.results) == 1

    def test_from_dict_defaults(self):
        data = {
            "timestamp": "ts",
            "total_questions": 1,
            "total_model_runs": 1,
        }
        s = RunSummary.from_dict(data)
        assert s.total_errors == 0
        assert s.total_elapsed_sec == 0.0
        assert s.judge_model is None

    def test_roundtrip(self):
        s = RunSummary(
            timestamp="ts", total_questions=3, total_model_runs=6,
            total_errors=0, total_elapsed_sec=10.0, total_estimated_cost=0.001,
        )
        d = s.to_dict()
        s2 = RunSummary.from_dict(d)
        assert s2.timestamp == s.timestamp
        assert s2.total_questions == s.total_questions


# ---------------------------------------------------------------------------
# ThresholdConfig
# ---------------------------------------------------------------------------


class TestThresholdConfig:
    def test_from_dict_all_set(self):
        t = ThresholdConfig.from_dict({
            "avg_faithfulness": 0.8,
            "avg_relevance": 0.7,
            "p95_ttft_ms": 500,
            "max_cost_per_query": 0.01,
        })
        assert t.avg_faithfulness == 0.8
        assert t.p95_ttft_ms == 500

    def test_from_dict_partial(self):
        t = ThresholdConfig.from_dict({"avg_faithfulness": 0.9})
        assert t.avg_faithfulness == 0.9
        assert t.avg_relevance is None
        assert t.p95_ttft_ms is None

    def test_from_dict_empty(self):
        t = ThresholdConfig.from_dict({})
        assert t.avg_faithfulness is None

    def test_to_dict(self):
        t = ThresholdConfig(avg_faithfulness=0.8, p95_ttft_ms=1000)
        d = t.to_dict()
        assert d["avg_faithfulness"] == 0.8
        assert d["p95_ttft_ms"] == 1000
        assert d["avg_relevance"] is None


# ---------------------------------------------------------------------------
# MetricCheck
# ---------------------------------------------------------------------------


class TestMetricCheck:
    def test_passing_check(self):
        c = MetricCheck(metric="avg_faithfulness", threshold=0.8, actual=0.9, passed=True)
        assert c.passed is True

    def test_failing_check(self):
        c = MetricCheck(metric="p95_ttft_ms", threshold=500.0, actual=800.0, passed=False)
        assert c.passed is False

    def test_to_dict(self):
        c = MetricCheck(metric="m", threshold=1.0, actual=0.5, passed=False)
        d = c.to_dict()
        assert d["metric"] == "m"
        assert d["passed"] is False


# ---------------------------------------------------------------------------
# ThresholdResult
# ---------------------------------------------------------------------------


class TestThresholdResult:
    def test_all_pass(self):
        checks = [
            MetricCheck(metric="a", threshold=0.8, actual=0.9, passed=True),
            MetricCheck(metric="b", threshold=0.7, actual=0.8, passed=True),
        ]
        r = ThresholdResult(overall_pass=True, per_metric=checks)
        assert r.overall_pass is True
        assert len(r.per_metric) == 2

    def test_some_fail(self):
        checks = [
            MetricCheck(metric="a", threshold=0.8, actual=0.9, passed=True),
            MetricCheck(metric="b", threshold=0.9, actual=0.7, passed=False),
        ]
        r = ThresholdResult(overall_pass=False, per_metric=checks)
        assert r.overall_pass is False

    def test_to_dict(self):
        checks = [MetricCheck(metric="x", threshold=1.0, actual=0.5, passed=False)]
        r = ThresholdResult(overall_pass=False, per_metric=checks)
        d = r.to_dict()
        assert d["overall_pass"] is False
        assert len(d["per_metric"]) == 1
        assert d["per_metric"][0]["metric"] == "x"

    def test_empty_checks(self):
        r = ThresholdResult(overall_pass=True, per_metric=[])
        assert r.overall_pass is True
        assert r.to_dict()["per_metric"] == []

    def test_boundary_values(self):
        c = MetricCheck(metric="m", threshold=0.8, actual=0.8, passed=True)
        assert c.passed is True

    def test_zero_threshold(self):
        c = MetricCheck(metric="m", threshold=0.0, actual=0.0, passed=True)
        assert c.passed is True
