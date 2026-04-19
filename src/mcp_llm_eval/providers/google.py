"""Streaming runner for Google GenAI models (generate_content_stream)."""

from __future__ import annotations

import time
from typing import Any

# Lazy module-level reference; set on first use by _get_client() or run().
genai = None


def _ensure_genai() -> Any:
    global genai
    if genai is None:
        try:
            from google import genai as _genai
            genai = _genai
        except ImportError as e:
            raise ImportError(
                "Google GenAI SDK not installed. Install it with: pip install google-genai"
            ) from e
    return genai


def _get_client() -> Any:
    _ensure_genai()
    return genai.Client()


def run(
    client: Any,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Run a streaming completion against a Google GenAI model.

    Returns dict with: response, input_tokens, output_tokens, stop_reason,
    time_to_first_token_ms, total_latency_ms.
    """
    _ensure_genai()

    t_start = time.monotonic()
    t_first_token = None
    full_response = ""
    input_tokens = output_tokens = 0
    stop_reason = None

    contents = [genai.types.Content(
        role="user",
        parts=[genai.types.Part(text=question)],
    )]

    response_stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        ),
    )

    last_chunk = None
    for chunk in response_stream:
        if chunk.text:
            if t_first_token is None:
                t_first_token = time.monotonic()
            full_response += chunk.text
        last_chunk = chunk

    try:
        if last_chunk and last_chunk.usage_metadata:
            input_tokens = last_chunk.usage_metadata.prompt_token_count
            output_tokens = last_chunk.usage_metadata.candidates_token_count
        if last_chunk and last_chunk.candidates:
            stop_reason = str(last_chunk.candidates[0].finish_reason)
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
