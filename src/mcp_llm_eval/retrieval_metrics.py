"""Pure retrieval metric functions: recall@k, precision@k, MRR, nDCG@k.

All functions take an ordered list of retrieved chunk IDs (rank 0 = top
result), a list of ground-truth relevant chunk IDs, and a cutoff k. All
return a float in [0.0, 1.0]. No I/O, no LLM calls, no side effects.

Conventions:

- Binary relevance only. A chunk is either in the relevant set or not.
  Graded relevance is reserved for a future release.
- ``retrieved`` is deduplicated by chunk_id, first occurrence wins, before
  any metric math. Ill-behaved retrievers that return duplicates will not
  double-count.
- Precision@k uses ``k`` as the denominator regardless of how many chunks
  were actually retrieved. If a retriever returns fewer than ``k`` chunks,
  precision is naturally lower — this is the standard IR convention.
- nDCG uses binary gains with a log2(rank + 2) discount. The ideal DCG caps
  at ``min(k, len(relevant))``.
- ``k <= 0`` raises ``ValueError``.
- Empty ``relevant`` makes recall and nDCG return 0.0 by convention.
"""

from __future__ import annotations

import math


def _dedupe(retrieved: list[str]) -> list[str]:
    """Return ``retrieved`` with later duplicates removed, preserving rank."""
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id in retrieved:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(chunk_id)
    return out


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top-k retrieved."""
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _dedupe(retrieved)[:k]
    hits = sum(1 for c in top if c in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant.

    Denominator is always ``k`` — if fewer than ``k`` chunks were retrieved,
    precision is correspondingly lower.
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _dedupe(retrieved)[:k]
    hits = sum(1 for c in top if c in relevant_set)
    return hits / k


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant chunk in the ordered list.

    Returns 0.0 if no relevant chunk is retrieved or if ``relevant`` is empty.
    The name is conventional — this is a single-query reciprocal rank;
    averaging across queries is the caller's responsibility.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _dedupe(retrieved)
    for rank, chunk_id in enumerate(top):
        if chunk_id in relevant_set:
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Normalised discounted cumulative gain at k with binary gains.

    DCG@k  = sum_{i=0..min(k,len(retrieved))-1}  rel_i / log2(i + 2)
    IDCG@k = sum_{i=0..min(k,|relevant|)-1}       1 / log2(i + 2)
    nDCG@k = DCG@k / IDCG@k  (or 0.0 if IDCG@k == 0)
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _dedupe(retrieved)[:k]
    dcg = 0.0
    for i, chunk_id in enumerate(top):
        if chunk_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
