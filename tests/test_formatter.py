"""Tests for formatter.py: markdown generation, with/without comparison and thresholds."""

from mcp_llm_eval.formatter import (
    format_pr_comment,
    _check_thresholds_inline,
    _compute_threshold_details,
    _format_delta,
    _format_value,
)


# ---------------------------------------------------------------------------
# format_pr_comment — basic output
# ---------------------------------------------------------------------------


class TestFormatPrCommentBasic:
    def _make_summary(self, overall=None):
        return {
            "timestamp": "20250101_000000",
            "total_questions": 10,
            "total_model_runs": 20,
            "total_errors": 0,
            "total_estimated_cost": 0.05,
            "judge_model": "gpt-4o-mini",
            "overall": overall or {},
        }

    def test_header_present(self):
        md = format_pr_comment(self._make_summary())
        assert "## LLM Eval Results" in md

    def test_table_headers(self):
        md = format_pr_comment(self._make_summary({
            "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
        }))
        assert "| Model |" in md
        assert "| Faithfulness |" in md

    def test_model_row(self):
        md = format_pr_comment(self._make_summary({
            "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
        }))
        assert "model-a" in md
        assert "0.90" in md
        assert "0.85" in md
        assert "150ms" in md
        assert "$0.0010" in md

    def test_overall_summary_line(self):
        md = format_pr_comment(self._make_summary({
            "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
        }))
        assert "Total cost: $0.05" in md
        assert "Questions: 10" in md
        assert "Judge: gpt-4o-mini" in md

    def test_regressions_section_present(self):
        md = format_pr_comment(self._make_summary())
        assert "Regressions vs baseline" in md

    def test_no_regressions_message(self):
        md = format_pr_comment(self._make_summary())
        assert "(none detected)" in md

    def test_multiple_models(self):
        md = format_pr_comment(self._make_summary({
            "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
            "model-b": {"avg_faithfulness": 0.7, "avg_relevance": 0.75, "avg_ttft_ms": 200, "avg_cost_per_query": 0.002},
        }))
        assert "model-a" in md
        assert "model-b" in md

    def test_na_for_missing_metrics(self):
        md = format_pr_comment(self._make_summary({
            "model-a": {},
        }))
        assert "N/A" in md


# ---------------------------------------------------------------------------
# format_pr_comment — with comparison
# ---------------------------------------------------------------------------


class TestFormatPrCommentWithComparison:
    def _make_summary(self):
        return {
            "timestamp": "20250101_000000",
            "total_questions": 10,
            "total_model_runs": 20,
            "total_errors": 0,
            "total_estimated_cost": 0.05,
            "judge_model": "gpt-4o-mini",
            "overall": {
                "model-a": {"avg_faithfulness": 0.7, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
            },
        }

    def test_regressions_shown(self):
        comparison = {
            "regressions": [
                {"model": "model-a", "metric": "faithfulness", "baseline": 0.9, "current": 0.7, "delta": -0.2},
            ],
            "has_regressions": True,
        }
        md = format_pr_comment(self._make_summary(), comparison=comparison)
        assert "model-a" in md
        assert "faithfulness" in md
        assert "-0.2000" in md

    def test_no_regressions_with_comparison(self):
        comparison = {"regressions": [], "has_regressions": False}
        md = format_pr_comment(self._make_summary(), comparison=comparison)
        assert "(none detected)" in md


# ---------------------------------------------------------------------------
# format_pr_comment — with thresholds
# ---------------------------------------------------------------------------


class TestFormatPrCommentWithThresholds:
    def _make_summary(self, overall=None):
        return {
            "timestamp": "20250101_000000",
            "total_questions": 10,
            "total_model_runs": 20,
            "total_errors": 0,
            "total_estimated_cost": 0.05,
            "judge_model": "gpt-4o-mini",
            "overall": overall or {
                "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
            },
        }

    def test_threshold_details_section(self):
        thresholds = {"avg_faithfulness": 0.8, "avg_relevance": 0.7}
        md = format_pr_comment(self._make_summary(), thresholds=thresholds)
        assert "Threshold details" in md
        assert "avg_faithfulness" in md
        assert ">= 0.80" in md

    def test_pass_status(self):
        thresholds = {"avg_faithfulness": 0.8}
        md = format_pr_comment(self._make_summary(), thresholds=thresholds)
        assert "\u2705 PASS" in md

    def test_fail_status(self):
        thresholds = {"avg_faithfulness": 0.95}  # higher than 0.9
        md = format_pr_comment(self._make_summary(), thresholds=thresholds)
        assert "\u274c FAIL" in md

    def test_no_threshold_details_without_thresholds(self):
        md = format_pr_comment(self._make_summary())
        assert "Threshold details" not in md

    def test_model_fail_with_thresholds(self):
        overall = {
            "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
            "model-b": {"avg_faithfulness": 0.5, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
        }
        thresholds = {"avg_faithfulness": 0.8}
        md = format_pr_comment(self._make_summary(overall), thresholds=thresholds)
        assert "\u274c FAIL" in md  # model-b fails

    def test_overall_fail(self):
        overall = {
            "model-a": {"avg_faithfulness": 0.5, "avg_relevance": 0.3, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
        }
        thresholds = {"avg_faithfulness": 0.8}
        md = format_pr_comment(self._make_summary(overall), thresholds=thresholds)
        assert "**Overall: FAIL**" in md


# ---------------------------------------------------------------------------
# _check_thresholds_inline
# ---------------------------------------------------------------------------


class TestCheckThresholdsInline:
    def test_all_pass(self):
        overall = {"m": {"avg_faithfulness": 0.9, "avg_relevance": 0.85}}
        thresholds = {"avg_faithfulness": 0.8, "avg_relevance": 0.7}
        result = _check_thresholds_inline(overall, thresholds)
        assert result["m"] is True

    def test_fail(self):
        overall = {"m": {"avg_faithfulness": 0.5}}
        thresholds = {"avg_faithfulness": 0.8}
        result = _check_thresholds_inline(overall, thresholds)
        assert result["m"] is False

    def test_ttft_fail(self):
        overall = {"m": {"avg_ttft_ms": 3000}}
        thresholds = {"p95_ttft_ms": 2000}
        result = _check_thresholds_inline(overall, thresholds)
        assert result["m"] is False

    def test_cost_fail(self):
        overall = {"m": {"avg_cost_per_query": 0.05}}
        thresholds = {"max_cost_per_query": 0.02}
        result = _check_thresholds_inline(overall, thresholds)
        assert result["m"] is False

    def test_no_thresholds(self):
        overall = {"m": {"avg_faithfulness": 0.5}}
        result = _check_thresholds_inline(overall, {})
        assert result["m"] is True


# ---------------------------------------------------------------------------
# _compute_threshold_details
# ---------------------------------------------------------------------------


class TestComputeThresholdDetails:
    def test_faithfulness_check(self):
        overall = {"m": {"avg_faithfulness": 0.9}}
        thresholds = {"avg_faithfulness": 0.8}
        checks = _compute_threshold_details(overall, thresholds)
        assert len(checks) == 1
        assert checks[0]["metric"] == "avg_faithfulness"
        assert checks[0]["passed"] is True

    def test_all_metrics(self):
        overall = {"m": {
            "avg_faithfulness": 0.9, "avg_relevance": 0.85,
            "avg_ttft_ms": 150, "avg_cost_per_query": 0.001,
        }}
        thresholds = {
            "avg_faithfulness": 0.8, "avg_relevance": 0.7,
            "p95_ttft_ms": 2000, "max_cost_per_query": 0.02,
        }
        checks = _compute_threshold_details(overall, thresholds)
        assert len(checks) == 4


# ---------------------------------------------------------------------------
# _format_delta / _format_value
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_format_delta_faithfulness(self):
        assert _format_delta("faithfulness", -0.05) == "-0.0500"

    def test_format_delta_ttft(self):
        assert _format_delta("avg_ttft_ms", 50) == "+50ms"

    def test_format_delta_cost(self):
        assert _format_delta("avg_cost_per_query", 0.001) == "+0.001000"

    def test_format_value_faithfulness(self):
        assert _format_value("faithfulness", 0.9) == "0.9000"

    def test_format_value_ttft(self):
        assert _format_value("avg_ttft_ms", 150) == "150ms"

    def test_format_value_cost(self):
        assert _format_value("avg_cost_per_query", 0.001) == "$0.0010"
