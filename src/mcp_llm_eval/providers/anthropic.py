"""Streaming runner for Anthropic models (messages.stream, TTFT capture)."""

from __future__ import annotations

import time
from typing import Any


def _get_client() -> Any:
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK not installed. Install it with: pip install anthropic"
        )
    return anthropic.Anthropic()


def run(
    client: Any,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Run a streaming completion against an Anthropic model.

    Returns dict with: response, input_tokens, output_tokens, stop_reason,
    time_to_first_token_ms, total_latency_ms.
    """
    t_start = time.monotonic()
    t_first_token = None
    full_response = ""
    input_tokens = output_tokens = 0
    stop_reason = None

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        for text in stream.text_stream:
            if t_first_token is None:
                t_first_token = time.monotonic()
            full_response += text

        try:
            final = stream.get_final_message()
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens
            stop_reason = final.stop_reason
        except Exception:
            pass

    t_end = time.monotonic()
    return {
        "response": full_response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "time_to_first_token_ms": round((t_first_token - t_start) * 1000) if t_first_token else None,
        "total_latency_ms": round((t_end - t_start) * 1000),
    }
