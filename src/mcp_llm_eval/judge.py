"""LLM-as-judge scorer: faithfulness and relevance on a 0-1 scale."""

from __future__ import annotations

import json
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


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def _get_openai_client() -> Any:
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "OpenAI SDK not installed. The judge requires it: pip install openai"
        ) from e
    return openai.OpenAI()


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

    raw = completion.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    scores = json.loads(raw)

    return {
        "faithfulness_score": float(scores["faithfulness_score"]),
        "faithfulness_reason": scores["faithfulness_reason"],
        "relevance_score": float(scores["relevance_score"]),
        "relevance_reason": scores["relevance_reason"],
        "judge_model": judge_model,
    }
