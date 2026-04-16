"""Tests for comparison.py: regression detection, tolerance math, edge cases."""

from mcp_llm_eval.comparison import compare_runs, _apply_tolerance_defaults


# ---------------------------------------------------------------------------
# _apply_tolerance_defaults
# ---------------------------------------------------------------------------


class TestApplyToleranceDefaults:
    def test_all_defaults(self):
        result = _apply_tolerance_defaults({})
        assert result["faithfulness"] == 0.05
        assert result["relevance"] == 0.05
        assert result["ttft_ms"] == 200
        assert result["cost"] == 0.005

    def test_custom_values(self):
        result = _apply_tolerance_defaults({
            "faithfulness": 0.1,
            "relevance": 0.1,
            "ttft_ms": 500,
            "cost": 0.01,
        })
        assert result["faithfulness"] == 0.1
        assert result["relevance"] == 0.1
        assert result["ttft_ms"] == 500
        assert result["cost"] == 0.01

    def test_partial_override(self):
        result = _apply_tolerance_defaults({"faithfulness": 0.1})
        assert result["faithfulness"] == 0.1
        assert result["relevance"] == 0.05  # default


# ---------------------------------------------------------------------------
# compare_runs — no regressions
# ---------------------------------------------------------------------------


class TestCompareRunsNoRegression:
    def _make_summary(self, timestamp, models_data):
        return {"timestamp": timestamp, "overall": models_data}

    def test_identical_runs(self):
        data = {"model-a": {
            "avg_faithfulness": 0.9, "avg_relevance": 0.85,
            "avg_ttft_ms": 150, "avg_cost_per_query": 0.001,
        }}
        result = compare_runs(
            self._make_summary("t1", data),
            self._make_summary("t2", data),
        )
        assert result["has_regressions"] is False
        assert result["regressions"] == []
        assert result["baseline_timestamp"] == "t1"
        assert result["current_timestamp"] == "t2"

    def test_improvement_no_regression(self):
        baseline = {"model-a": {
            "avg_faithfulness": 0.8, "avg_relevance": 0.7,
            "avg_ttft_ms": 200, "avg_cost_per_query": 0.002,
        }}
        current = {"model-a": {
            "avg_faithfulness": 0.9, "avg_relevance": 0.85,
            "avg_ttft_ms": 150, "avg_cost_per_query": 0.001,
        }}
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is False

    def test_within_tolerance(self):
        baseline = {"model-a": {
            "avg_faithfulness": 0.9, "avg_relevance": 0.85,
            "avg_ttft_ms": 150, "avg_cost_per_query": 0.001,
        }}
        current = {"model-a": {
            "avg_faithfulness": 0.86,  # drop of 0.04, within 0.05 tolerance
            "avg_relevance": 0.81,     # drop of 0.04, within 0.05 tolerance
            "avg_ttft_ms": 340,        # increase of 190, within 200 tolerance
            "avg_cost_per_query": 0.005,  # increase of 0.004, within 0.005 tolerance
        }}
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is False

    def test_empty_overall(self):
        result = compare_runs(
            {"timestamp": "t1", "overall": {}},
            {"timestamp": "t2", "overall": {}},
        )
        assert result["has_regressions"] is False
        assert result["models"] == {}


# ---------------------------------------------------------------------------
# compare_runs — regressions detected
# ---------------------------------------------------------------------------


class TestCompareRunsRegressions:
    def _make_summary(self, timestamp, models_data):
        return {"timestamp": timestamp, "overall": models_data}

    def test_faithfulness_regression(self):
        baseline = {"model-a": {"avg_faithfulness": 0.9}}
        current = {"model-a": {"avg_faithfulness": 0.8}}  # drop of 0.1 > 0.05
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is True
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["metric"] == "faithfulness"
        assert result["regressions"][0]["model"] == "model-a"

    def test_relevance_regression(self):
        baseline = {"model-a": {"avg_relevance": 0.9}}
        current = {"model-a": {"avg_relevance": 0.8}}  # drop of 0.1 > 0.05
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is True
        assert result["regressions"][0]["metric"] == "relevance"

    def test_ttft_regression(self):
        baseline = {"model-a": {"avg_ttft_ms": 150}}
        current = {"model-a": {"avg_ttft_ms": 400}}  # increase of 250 > 200
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is True
        assert result["regressions"][0]["metric"] == "avg_ttft_ms"

    def test_cost_regression(self):
        baseline = {"model-a": {"avg_cost_per_query": 0.001}}
        current = {"model-a": {"avg_cost_per_query": 0.01}}  # increase of 0.009 > 0.005
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is True
        assert result["regressions"][0]["metric"] == "avg_cost_per_query"

    def test_multiple_regressions(self):
        baseline = {"model-a": {
            "avg_faithfulness": 0.9, "avg_relevance": 0.9,
            "avg_ttft_ms": 100, "avg_cost_per_query": 0.001,
        }}
        current = {"model-a": {
            "avg_faithfulness": 0.7, "avg_relevance": 0.7,
            "avg_ttft_ms": 500, "avg_cost_per_query": 0.02,
        }}
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is True
        assert len(result["regressions"]) == 4

    def test_multiple_models_mixed(self):
        baseline = {
            "model-a": {"avg_faithfulness": 0.9},
            "model-b": {"avg_faithfulness": 0.9},
        }
        current = {
            "model-a": {"avg_faithfulness": 0.95},  # improved
            "model-b": {"avg_faithfulness": 0.7},   # regressed
        }
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is True
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["model"] == "model-b"


# ---------------------------------------------------------------------------
# compare_runs — custom tolerance
# ---------------------------------------------------------------------------


class TestCompareRunsCustomTolerance:
    def _make_summary(self, timestamp, models_data):
        return {"timestamp": timestamp, "overall": models_data}

    def test_custom_faithfulness_tolerance(self):
        baseline = {"model-a": {"avg_faithfulness": 0.9}}
        current = {"model-a": {"avg_faithfulness": 0.8}}  # drop of 0.1
        # With tolerance of 0.15, this should NOT be a regression
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
            tolerance={"faithfulness": 0.15},
        )
        assert result["has_regressions"] is False

    def test_strict_tolerance(self):
        baseline = {"model-a": {"avg_faithfulness": 0.9}}
        current = {"model-a": {"avg_faithfulness": 0.88}}  # drop of 0.02
        # With tolerance of 0.01, this IS a regression
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
            tolerance={"faithfulness": 0.01},
        )
        assert result["has_regressions"] is True


# ---------------------------------------------------------------------------
# compare_runs — edge cases
# ---------------------------------------------------------------------------


class TestCompareRunsEdgeCases:
    def _make_summary(self, timestamp, models_data):
        return {"timestamp": timestamp, "overall": models_data}

    def test_model_only_in_baseline(self):
        baseline = {"model-a": {"avg_faithfulness": 0.9}}
        current = {}
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        # model-a has no current metrics, so no comparison possible
        assert result["has_regressions"] is False
        assert "model-a" not in result["models"]

    def test_model_only_in_current(self):
        baseline = {}
        current = {"model-a": {"avg_faithfulness": 0.9}}
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        assert result["has_regressions"] is False
        assert "model-a" not in result["models"]

    def test_missing_metric_in_one_side(self):
        baseline = {"model-a": {"avg_faithfulness": 0.9}}
        current = {"model-a": {"avg_relevance": 0.8}}  # different metric
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        # Neither metric can be compared (one side always None)
        assert result["has_regressions"] is False

    def test_boundary_exactly_at_tolerance(self):
        baseline = {"model-a": {"avg_faithfulness": 0.9}}
        current = {"model-a": {"avg_faithfulness": 0.85}}  # drop of exactly 0.05
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        # Drop equals tolerance, not exceeding it — no regression
        assert result["has_regressions"] is False

    def test_delta_values_in_result(self):
        baseline = {"model-a": {
            "avg_faithfulness": 0.9, "avg_relevance": 0.8,
            "avg_ttft_ms": 100, "avg_cost_per_query": 0.001,
        }}
        current = {"model-a": {
            "avg_faithfulness": 0.88, "avg_relevance": 0.85,
            "avg_ttft_ms": 120, "avg_cost_per_query": 0.0012,
        }}
        result = compare_runs(
            self._make_summary("t1", baseline),
            self._make_summary("t2", current),
        )
        m = result["models"]["model-a"]
        assert m["faithfulness"]["delta"] == -0.02
        assert m["relevance"]["delta"] == 0.05
        assert m["avg_ttft_ms"]["delta"] == 20.0
        assert m["avg_cost_per_query"]["delta"] == 0.0002

    def test_missing_timestamp(self):
        result = compare_runs({"overall": {}}, {"overall": {}})
        assert result["baseline_timestamp"] == ""
        assert result["current_timestamp"] == ""
