"""Tests for RetrievalAdapter protocol and BM25Adapter."""

import json
import os
import tempfile

import pytest

from mcp_llm_eval.retrieval import BM25Adapter, _tokenize
from mcp_llm_eval.types import RetrievedChunk


SMALL_CORPUS = [
    {"chunk_id": "c1", "content": "Python is a popular programming language",
     "metadata": {"topic": "py"}},
    {"chunk_id": "c2", "content": "Poetry manages Python dependencies",
     "metadata": {"topic": "py"}},
    {"chunk_id": "c3", "content": "The James Webb Space Telescope orbits at L2",
     "metadata": {"topic": "space"}},
    {"chunk_id": "c4", "content": "Hubble orbits Earth in low orbit",
     "metadata": {"topic": "space"}},
    {"chunk_id": "c5", "content": "The EU AI Act classifies systems by risk tier",
     "metadata": {"topic": "policy"}},
]


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello, World!") == ["hello", "world"]

    def test_lowercases(self):
        assert _tokenize("PyPI") == ["pypi"]

    def test_keeps_apostrophes(self):
        assert _tokenize("it's Python's") == ["it's", "python's"]

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


# ---------------------------------------------------------------------------
# BM25Adapter construction
# ---------------------------------------------------------------------------


class TestBM25AdapterConstruction:
    def test_from_memory_corpus(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        assert adapter is not None

    def test_from_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for chunk in SMALL_CORPUS:
                f.write(json.dumps(chunk) + "\n")
            f.flush()
            try:
                adapter = BM25Adapter.from_jsonl(f.name)
                assert adapter is not None
            finally:
                os.unlink(f.name)

    def test_both_args_raises(self):
        with pytest.raises(ValueError, match="exactly one of"):
            BM25Adapter(corpus=SMALL_CORPUS, corpus_path="x")

    def test_neither_arg_raises(self):
        with pytest.raises(ValueError, match="exactly one of"):
            BM25Adapter()

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="corpus is empty"):
            BM25Adapter(corpus=[])

    def test_duplicate_chunk_id_raises(self):
        corpus = [
            {"chunk_id": "c1", "content": "first"},
            {"chunk_id": "c1", "content": "second"},
        ]
        with pytest.raises(ValueError, match="Duplicate chunk_id: c1"):
            BM25Adapter(corpus=corpus)

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            BM25Adapter(corpus=[{"chunk_id": "c1"}])

    def test_jsonl_malformed_reports_lineno(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"chunk_id": "c1", "content": "ok"}\n')
            f.write('not json here\n')
            f.flush()
            try:
                with pytest.raises(ValueError, match="Line 2:"):
                    BM25Adapter.from_jsonl(f.name)
            finally:
                os.unlink(f.name)

    def test_jsonl_missing_chunk_id_reports_lineno(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"content": "ok"}\n')
            f.flush()
            try:
                with pytest.raises(ValueError, match="Line 1: missing chunk_id"):
                    BM25Adapter.from_jsonl(f.name)
            finally:
                os.unlink(f.name)

    def test_jsonl_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            BM25Adapter.from_jsonl("/nonexistent/path.jsonl")


# ---------------------------------------------------------------------------
# BM25Adapter.retrieve
# ---------------------------------------------------------------------------


class TestBM25Retrieve:
    def test_returns_retrieved_chunk_type(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        results = adapter.retrieve("Python", 2)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_respects_k_cap(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        results = adapter.retrieve("Python", 2)
        assert len(results) == 2

    def test_relevant_chunk_ranked_first(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        results = adapter.retrieve("Python programming", 5)
        # The Python-related chunks should outrank the space/policy chunks
        top_ids = [r.chunk_id for r in results[:2]]
        assert "c1" in top_ids or "c2" in top_ids

    def test_ordered_descending_score(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        results = adapter.retrieve("Webb Telescope space", 5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        assert adapter.retrieve("", 3) == []
        assert adapter.retrieve("   ", 3) == []

    def test_unrelated_query_returns_k_with_zero_scores(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        results = adapter.retrieve("xyzzy", 3)
        assert len(results) == 3
        assert all(r.score == 0.0 for r in results)

    def test_k_larger_than_corpus_returns_all(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        results = adapter.retrieve("Python", 99)
        assert len(results) == len(SMALL_CORPUS)

    def test_k_zero_raises(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        with pytest.raises(ValueError, match="k must be positive"):
            adapter.retrieve("Python", 0)

    def test_k_negative_raises(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        with pytest.raises(ValueError, match="k must be positive"):
            adapter.retrieve("Python", -1)

    def test_deterministic_ordering(self):
        adapter1 = BM25Adapter(corpus=SMALL_CORPUS)
        adapter2 = BM25Adapter(corpus=SMALL_CORPUS)
        r1 = adapter1.retrieve("Python dependencies", 5)
        r2 = adapter2.retrieve("Python dependencies", 5)
        assert [x.chunk_id for x in r1] == [x.chunk_id for x in r2]
        assert [x.score for x in r1] == [x.score for x in r2]

    def test_tiebreak_by_original_index(self):
        # Two identical-content chunks should tie; first in corpus wins.
        corpus = [
            {"chunk_id": "first", "content": "alpha beta gamma"},
            {"chunk_id": "second", "content": "alpha beta gamma"},
            {"chunk_id": "third", "content": "completely different words"},
        ]
        adapter = BM25Adapter(corpus=corpus)
        results = adapter.retrieve("alpha", 2)
        assert results[0].chunk_id == "first"
        assert results[1].chunk_id == "second"

    def test_metadata_preserved(self):
        adapter = BM25Adapter(corpus=SMALL_CORPUS)
        results = adapter.retrieve("Python", 1)
        assert results[0].metadata.get("topic") == "py"
