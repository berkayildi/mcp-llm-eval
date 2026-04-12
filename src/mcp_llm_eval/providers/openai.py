"""Streaming runner for OpenAI models (chat.completions.create stream)."""

from __future__ import annotations

import time
from typing import Any


def _get_client() -> Any:
    try:
        import openai
    except ImportError:
        raise ImportError(
            "OpenAI SDK not installed. Install it with: pip install openai"
        )
    return openai.OpenAI()


def run(
    client: Any,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Run a streaming completion against an OpenAI model.

    Returns dict with: response, input_tokens, output_tokens, stop_reason,
    time_to_first_token_ms, total_latency_ms.
    """
    t_start = time.monotonic()
    t_first_token = None
    full_response = ""
    input_tokens = output_tokens = 0
    stop_reason = None

    stream = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if t_first_token is None:
                    t_first_token = time.monotonic()
                full_response += delta.content
            if chunk.choices[0].finish_reason:
                stop_reason = chunk.choices[0].finish_reason

    t_end = time.monotonic()
    return {
        "response": full_response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "time_to_first_token_ms": round((t_first_token - t_start) * 1000) if t_first_token else None,
        "total_latency_ms": round((t_end - t_start) * 1000),
    }
