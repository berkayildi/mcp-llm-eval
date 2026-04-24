"""Retrieval adapter protocol and in-memory BM25 implementation.

The ``RetrievalAdapter`` protocol is a minimal synchronous interface:
``retrieve(query, k) -> list[RetrievedChunk]`` where index 0 is the top
result. v0.5.0 ships one implementation, ``BM25Adapter``, backed by the
``rank_bm25`` package.

Tokenisation is intentionally naive: lowercase, split on non-word characters
(keeping apostrophes inside words). For production use, subclass
``BM25Adapter`` and override ``_tokenize``.

Tie-breaking is deterministic: descending BM25 score, then ascending
original corpus index, so the same corpus + query always produces the same
ranking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from .types import RetrievedChunk


class RetrievalAdapter(Protocol):
    """Synchronous retrieval interface.

    Implementations must return at most ``k`` chunks, ordered by descending
    relevance, and must return ``[]`` for an empty query without raising.
    """

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        ...


def _get_bm25() -> Any:
    """Lazily import the BM25Okapi class from rank_bm25."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError(
            "rank_bm25 not installed. In-memory retrieval requires it: "
            "pip install rank_bm25"
        ) from e
    return BM25Okapi


_TOKEN_SPLIT_RE = re.compile(r"[^\w']+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters; drop empty tokens."""
    return [t for t in _TOKEN_SPLIT_RE.split(text.lower()) if t]


class BM25Adapter:
    """In-memory BM25 retrieval adapter.

    Construct with either an in-memory list of chunk dicts or a JSONL path,
    not both. Each chunk dict must have ``chunk_id`` (str) and ``content``
    (str); ``metadata`` (dict) is optional.
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]] | None = None,
        corpus_path: str | None = None,
    ) -> None:
        if (corpus is None) == (corpus_path is None):
            raise ValueError(
                "BM25Adapter requires exactly one of corpus or corpus_path"
            )

        if corpus_path is not None:
            corpus = self._load_jsonl(corpus_path)

        assert corpus is not None  # for type-checkers; validated above
        if not corpus:
            raise ValueError("corpus is empty")

        seen_ids: set[str] = set()
        self._chunks: list[dict[str, Any]] = []
        for i, chunk in enumerate(corpus):
            if not isinstance(chunk, dict):
                raise ValueError(f"corpus[{i}] must be a mapping")
            if "chunk_id" not in chunk or "content" not in chunk:
                raise ValueError(
                    f"corpus[{i}] missing required keys: chunk_id and content"
                )
            cid = chunk["chunk_id"]
            if cid in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {cid}")
            seen_ids.add(cid)
            self._chunks.append({
                "chunk_id": cid,
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
            })

        tokenised = [_tokenize(c["content"]) for c in self._chunks]
        BM25Okapi = _get_bm25()
        self._bm25 = BM25Okapi(tokenised)

    @classmethod
    def from_jsonl(cls, path: str) -> BM25Adapter:
        """Construct from a JSONL corpus file."""
        return cls(corpus_path=path)

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        chunks: list[dict[str, Any]] = []
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunk = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {lineno}: invalid JSON ({e})") from e
            if not isinstance(chunk, dict):
                raise ValueError(f"Line {lineno}: expected JSON object, got {type(chunk).__name__}")
            if "chunk_id" not in chunk:
                raise ValueError(f"Line {lineno}: missing chunk_id")
            if "content" not in chunk:
                raise ValueError(f"Line {lineno}: missing content")
            metadata = chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Line {lineno}: metadata must be a mapping")
            chunks.append(chunk)
        return chunks

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            raise ValueError("k must be positive")
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        indexed = [(float(s), i) for i, s in enumerate(scores)]
        indexed.sort(key=lambda t: (-t[0], t[1]))
        top = indexed[:k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[i]["chunk_id"],
                content=self._chunks[i]["content"],
                score=score,
                metadata=dict(self._chunks[i]["metadata"]),
            )
            for score, i in top
        ]
