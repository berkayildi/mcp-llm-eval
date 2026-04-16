#!/bin/bash
set -e

# Load environment variables if .env exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "=== mcp-llm-eval e2e test ==="
echo ""
echo "Required keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY"
echo "  ANTHROPIC_API_KEY — runs Haiku as eval target"
echo "  OPENAI_API_KEY — runs GPT-4o-mini as eval target + judge"
echo "  GOOGLE_API_KEY — runs Gemini 2.5 Flash-Lite as eval target"
echo ""

# Step 1: Run evaluation with 3 models across 3 providers
echo "[1/4] Running evaluation with 3 models (Haiku, GPT-4o-mini, Gemini Flash-Lite)..."
mcp-llm-eval run \
  --dataset eval/dataset.json \
  --models '[{"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "max_tokens": 300, "input_cost_per_mtok": 1.0, "output_cost_per_mtok": 5.0}, {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 300, "input_cost_per_mtok": 0.15, "output_cost_per_mtok": 0.60}, {"provider": "google", "model": "gemini-2.5-flash-lite", "max_tokens": 300, "input_cost_per_mtok": 0.075, "output_cost_per_mtok": 0.30}]' \
  --judge-model gpt-4o-mini \
  --output-dir eval/results

echo ""

# Step 2: Check thresholds (should pass with lenient thresholds)
echo "[2/4] Checking thresholds (should PASS)..."
mcp-llm-eval check \
  --results eval/results/latest_summary.json \
  --fail-under-faithfulness 0.5 \
  --fail-under-relevance 0.5 \
  --fail-over-ttft 5000 \
  --fail-over-cost 0.10
echo "Threshold check: PASSED"

echo ""

# Step 3: Check thresholds (should fail with impossible thresholds)
echo "[3/4] Checking thresholds (should FAIL with exit code 1)..."
if mcp-llm-eval check \
  --results eval/results/latest_summary.json \
  --fail-under-faithfulness 0.99 \
  --fail-under-relevance 0.99 2>/dev/null; then
  echo "ERROR: Expected failure but got success"
  exit 1
else
  echo "Threshold check: CORRECTLY FAILED (exit code 1)"
fi

echo ""

# Step 4: Generate PR comment
echo "[4/4] Generating PR comment..."
mcp-llm-eval comment \
  --summary eval/results/latest_summary.json \
  --output eval/results/pr_comment.md
echo "PR comment written to eval/results/pr_comment.md"

echo ""
echo "=== E2E test complete ==="
echo "Results: eval/results/"
