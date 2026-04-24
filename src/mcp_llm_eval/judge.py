"""LLM-as-judge scorers for generation, retrieval, and RAG evaluation.

All public APIs, thresholds, and reports use a float in [0.0, 1.0] for every
judge score. The v0.4.x ``evaluate_response`` asks the judge for 0-1 floats
directly (faithfulness + relevance). The v0.5.0 judges
(``judge_context_relevance``, ``judge_citation_faithfulness``) ask the judge
for an integer 1-5 with anchor descriptions — integer buckets are measurably
more reliable than asking for arbitrary floats — and normalise to 0-1 with
``(score - 1) / 4``: 1 → 0.0, 2 → 0.25, 3 → 0.5, 4 → 0.75, 5 → 1.0.

All judge calls run at ``temperature=0``; deterministic scoring is a hard
requirement.

Judge model precedence (see ``_resolve_judge_model``):
    explicit argument > ``MCP_LLM_EVAL_JUDGE_MODEL`` env var > ``DEFAULT_JUDGE_MODEL``.
"""

from __future__ import annotations

import json
import os
from typing import Any

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator scoring LLM responses for a question-answering assistant.
You will receive: the context (system prompt), the question asked, \
a reference expected response, and the model's actual response.

Score the response on two dimensions, each 0.0 to 1.0:

**Faithfulness** — Is the response grounded in the provided context?
- 1.0: Every claim is supported by or directly inferable from the context
- 0.5: Mostly grounded but includes some unsupported claims or mild extrapolation
- 0.0: Contains hallucinated facts, invented specifics, or contradicts the context

**Relevance** — Does the response actually answer the question?
- 1.0: Directly and completely addresses the question with actionable content
- 0.5: Partially addresses the question or is tangentially related
- 0.0: Does not address the question at all or is off-topic

Return ONLY a JSON object with exactly these keys:
{"faithfulness_score": <float>, "faithfulness_reason": "<string>", \
"relevance_score": <float>, "relevance_reason": "<string>"}"""

JUDGE_USER_TEMPLATE = """\
## Context (System Prompt)
{context}

## Question Asked
{question}

## Reference Expected Response
{expected_response}

## Model's Actual Response
{response}

Score the model's actual response. Return only the JSON object."""


CONTEXT_RELEVANCE_SYSTEM_PROMPT = """\
You are an expert evaluator scoring retrieval quality for a
question-answering system.

You will receive a user query and a single retrieved text chunk. Score how
relevant the chunk is to the query on an integer scale from 1 to 5:

**5 — Highly relevant.** The chunk directly answers the query or contains
the key information needed to answer it.
**4 — Relevant.** The chunk is clearly on-topic and contributes useful
supporting information, even if it is not the complete answer.
**3 — Marginally relevant.** The chunk touches on the topic but would not,
on its own, be enough to answer the query.
**2 — Weakly relevant.** The chunk shares surface-level keywords with the
query but does not contain information that would help answer it.
**1 — Not relevant.** The chunk is off-topic, unrelated, or misleading.

Return ONLY a JSON object with exactly these keys:
{"score": <integer 1-5>, "reason": "<one-sentence justification>"}"""

CONTEXT_RELEVANCE_USER_TEMPLATE = """\
## Query
{query}

## Retrieved Chunk
{chunk}

Score the chunk's relevance to the query. Return only the JSON object."""


CITATION_FAITHFULNESS_SYSTEM_PROMPT = """\
You are an expert evaluator scoring factual faithfulness of answers
produced by a retrieval-augmented generation (RAG) system.

You will receive the generated answer and the list of chunks that were
cited as its sources. Score how faithfully the answer is supported by the
cited chunks on an integer scale from 1 to 5:

**5 — Fully faithful.** Every factual claim in the answer is directly
supported by the cited chunks. No fabricated details.
**4 — Mostly faithful.** Nearly all claims are supported; any unsupported
content is minor (e.g., a connective phrase, a rewording) and does not
introduce new facts.
**3 — Partially faithful.** Some claims are supported by the chunks and
some are not. The unsupported claims are not clearly contradicted by the
chunks but are not present in them either.
**2 — Weakly faithful.** A substantial portion of the answer is
unsupported or extrapolated well beyond what the chunks state.
**1 — Unfaithful.** The answer contains fabricated facts, contradicts the
cited chunks, or is essentially unsupported.

Return ONLY a JSON object with exactly these keys:
{"score": <integer 1-5>, "reason": "<one-sentence justification>"}"""

CITATION_FAITHFULNESS_USER_TEMPLATE = """\
## Generated Answer
{answer}

## Cited Chunks
{chunks}

Score how faithfully the answer is supported by the cited chunks. Return
only the JSON object."""


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
JUDGE_MODEL_ENV_VAR = "MCP_LLM_EVAL_JUDGE_MODEL"


def _get_openai_client() -> Any:
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "OpenAI SDK not installed. The judge requires it: pip install openai"
        ) from e
    return openai.OpenAI()


def _resolve_judge_model(override: str | None) -> str:
    """Pick the judge model: explicit arg > env var > default."""
    if override:
        return override
    env_val = os.environ.get(JUDGE_MODEL_ENV_VAR)
    if env_val:
        return env_val
    return DEFAULT_JUDGE_MODEL


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Strip optional markdown code fences from a judge response and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return json.loads(raw)


def _scale_1_to_5(score: int | float) -> float:
    """Map a 1-5 judge score to a float in [0.0, 1.0] via (score - 1) / 4."""
    s = float(score)
    if s < 1.0 or s > 5.0:
        raise ValueError(f"Judge score out of range [1, 5]: {score}")
    return round((s - 1.0) / 4.0, 4)


def evaluate_response(
    client: Any,
    context: str,
    question: str,
    expected_response: str,
    response: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = 0,
) -> dict[str, Any]:
    """Run the LLM judge on a single response.

    Returns dict with faithfulness_score, faithfulness_reason,
    relevance_score, relevance_reason, judge_model.
    """
    user_msg = JUDGE_USER_TEMPLATE.format(
        context=context,
        question=question,
        expected_response=expected_response,
        response=response,
    )

    completion = client.chat.completions.create(
        model=judge_model,
        max_tokens=300,
        temperature=temperature,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = completion.choices[0].message.content
    scores = _parse_judge_json(raw)

    return {
        "faithfulness_score": float(scores["faithfulness_score"]),
        "faithfulness_reason": scores["faithfulness_reason"],
        "relevance_score": float(scores["relevance_score"]),
        "relevance_reason": scores["relevance_reason"],
        "judge_model": judge_model,
    }


def judge_context_relevance(
    client: Any,
    query: str,
    chunk: str,
    judge_model: str | None = None,
    temperature: float = 0,
) -> dict[str, Any]:
    """Score how relevant a single retrieved chunk is to a query.

    Returns {"score": float 0-1, "reason": str, "raw_score": int 1-5,
    "judge_model": str}.
    """
    model = _resolve_judge_model(judge_model)
    user_msg = CONTEXT_RELEVANCE_USER_TEMPLATE.format(query=query, chunk=chunk)

    completion = client.chat.completions.create(
        model=model,
        max_tokens=150,
        temperature=temperature,
        messages=[
            {"role": "system", "content": CONTEXT_RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = completion.choices[0].message.content
    parsed = _parse_judge_json(raw)
    raw_score = int(parsed["score"])
    return {
        "score": _scale_1_to_5(raw_score),
        "reason": parsed["reason"],
        "raw_score": raw_score,
        "judge_model": model,
    }


def _render_cited_chunks(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(f"[chunk {i + 1}]\n{c}" for i, c in enumerate(chunks))


def judge_citation_faithfulness(
    client: Any,
    answer: str,
    cited_chunks: list[str],
    judge_model: str | None = None,
    temperature: float = 0,
) -> dict[str, Any]:
    """Score how faithfully a generated answer is supported by its cited chunks.

    Short-circuits with score 0.0 if ``cited_chunks`` is empty — no API call
    is made, since an answer with no citations cannot be supported.
    Returns {"score": float 0-1, "reason": str, "raw_score": int 1-5,
    "judge_model": str}.
    """
    model = _resolve_judge_model(judge_model)

    if not cited_chunks:
        return {
            "score": 0.0,
            "reason": "No chunks cited",
            "raw_score": 1,
            "judge_model": model,
        }

    user_msg = CITATION_FAITHFULNESS_USER_TEMPLATE.format(
        answer=answer,
        chunks=_render_cited_chunks(cited_chunks),
    )

    completion = client.chat.completions.create(
        model=model,
        max_tokens=200,
        temperature=temperature,
        messages=[
            {"role": "system", "content": CITATION_FAITHFULNESS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = completion.choices[0].message.content
    parsed = _parse_judge_json(raw)
    raw_score = int(parsed["score"])
    return {
        "score": _scale_1_to_5(raw_score),
        "reason": parsed["reason"],
        "raw_score": raw_score,
        "judge_model": model,
    }
