"""Tests for the pure retrieval-metric functions.

Expected values for nDCG are computed directly from the spec formula in
docs/design/v0.5.0-retrieval-eval.md §5 (cases 1-7).
"""

import pytest

from mcp_llm_eval.retrieval_metrics import (
    _dedupe,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_case1_half_recall(self):
        # retrieved=[A,B,C,D], relevant={A,C}, k=2 → 0.5
        assert recall_at_k(["A", "B", "C", "D"], ["A", "C"], 2) == pytest.approx(0.5)

    def test_case2_full_recall_at_k4(self):
        assert recall_at_k(["A", "B", "C", "D"], ["A", "C"], 4) == pytest.approx(1.0)

    def test_case3_reordered_full_recall(self):
        assert recall_at_k(["B", "A", "C"], ["A", "C"], 3) == pytest.approx(1.0)

    def test_case4_no_overlap(self):
        assert recall_at_k(["X", "Y", "Z"], ["A", "B"], 3) == 0.0

    def test_case5_k_larger_than_retrieved(self):
        # retrieved=[A], relevant={A,B,C}, k=5 → 1/3
        assert recall_at_k(["A"], ["A", "B", "C"], 5) == pytest.approx(1 / 3)

    def test_case6_empty_retrieved(self):
        assert recall_at_k([], ["A"], 2) == 0.0

    def test_case7_duplicates_deduped(self):
        # [A,A,C] dedupes to [A,C]; both relevant → recall 1.0
        assert recall_at_k(["A", "A", "C"], ["A", "C"], 3) == pytest.approx(1.0)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["A"], [], 1) == 0.0

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(["A"], ["A"], 0)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_case1(self):
        assert precision_at_k(["A", "B", "C", "D"], ["A", "C"], 2) == pytest.approx(0.5)

    def test_case2_half_precision(self):
        assert precision_at_k(["A", "B", "C", "D"], ["A", "C"], 4) == pytest.approx(0.5)

    def test_case3(self):
        # retrieved=[B,A,C], k=3, 2 hits / 3 = 2/3
        assert precision_at_k(["B", "A", "C"], ["A", "C"], 3) == pytest.approx(2 / 3)

    def test_case4_zero(self):
        assert precision_at_k(["X", "Y", "Z"], ["A", "B"], 3) == 0.0

    def test_case5_denominator_is_k_not_retrieved_len(self):
        # retrieved=[A], relevant={A,B,C}, k=5 → 1/5 = 0.2 (NOT 1.0)
        assert precision_at_k(["A"], ["A", "B", "C"], 5) == pytest.approx(0.2)

    def test_case6_empty(self):
        assert precision_at_k([], ["A"], 2) == 0.0

    def test_duplicates_deduped(self):
        # [A,A,C] → [A,C]; 2 hits / k=3 = 2/3
        assert precision_at_k(["A", "A", "C"], ["A", "C"], 3) == pytest.approx(2 / 3)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(["A"], ["A"], -1)


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------


class TestMeanReciprocalRank:
    def test_first_relevant_at_rank_0(self):
        assert mean_reciprocal_rank(["A", "B", "C"], ["A"]) == 1.0

    def test_first_relevant_at_rank_1(self):
        assert mean_reciprocal_rank(["X", "A", "B"], ["A"]) == pytest.approx(0.5)

    def test_first_relevant_at_rank_2(self):
        assert mean_reciprocal_rank(["X", "Y", "A"], ["A"]) == pytest.approx(1 / 3)

    def test_no_relevant_retrieved(self):
        assert mean_reciprocal_rank(["X", "Y", "Z"], ["A", "B"]) == 0.0

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank([], ["A"]) == 0.0

    def test_empty_relevant(self):
        assert mean_reciprocal_rank(["A"], []) == 0.0

    def test_case3_first_hit_at_rank_1(self):
        # retrieved=[B,A,C], relevant={A,C} → first hit is A at rank 1 → 0.5
        assert mean_reciprocal_rank(["B", "A", "C"], ["A", "C"]) == pytest.approx(0.5)

    def test_duplicate_before_relevant_deduped(self):
        # [A,A,B,C] → dedupes to [A,B,C]; if relevant={C}, rank 2 → 1/3
        assert mean_reciprocal_rank(["A", "A", "B", "C"], ["C"]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    def test_case1(self):
        # retrieved=[A,B,C,D], relevant={A,C}, k=2 → 0.6131
        assert ndcg_at_k(["A", "B", "C", "D"], ["A", "C"], 2) == pytest.approx(
            0.613147, abs=1e-4
        )

    def test_case2(self):
        assert ndcg_at_k(["A", "B", "C", "D"], ["A", "C"], 4) == pytest.approx(
            0.919721, abs=1e-4
        )

    def test_case3(self):
        assert ndcg_at_k(["B", "A", "C"], ["A", "C"], 3) == pytest.approx(
            0.693426, abs=1e-4
        )

    def test_case4_zero(self):
        assert ndcg_at_k(["X", "Y", "Z"], ["A", "B"], 3) == 0.0

    def test_case5_k_exceeds_retrieved(self):
        # retrieved=[A], relevant={A,B,C}, k=5 → 1 / 2.130930 = 0.469279
        assert ndcg_at_k(["A"], ["A", "B", "C"], 5) == pytest.approx(0.469279, abs=1e-4)

    def test_case6_empty_retrieved(self):
        assert ndcg_at_k([], ["A"], 2) == 0.0

    def test_case7_duplicates_deduped(self):
        # [A,A,C] → [A,C]; relevant={A,C}, k=3 → nDCG=1.0
        assert ndcg_at_k(["A", "A", "C"], ["A", "C"], 3) == pytest.approx(1.0)

    def test_perfect_ranking_at_k_equals_relevant(self):
        # All relevant at top → 1.0
        assert ndcg_at_k(["A", "B", "C"], ["A", "B"], 2) == pytest.approx(1.0)

    def test_empty_relevant(self):
        assert ndcg_at_k(["A", "B"], [], 2) == 0.0

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            ndcg_at_k(["A"], ["A"], 0)


# ---------------------------------------------------------------------------
# _dedupe helper
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_preserves_first_occurrence_order(self):
        assert _dedupe(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]

    def test_empty(self):
        assert _dedupe([]) == []

    def test_no_duplicates(self):
        assert _dedupe(["A", "B", "C"]) == ["A", "B", "C"]
