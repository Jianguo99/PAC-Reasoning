#!/usr/bin/env bash
# Serve the non-thinking and thinking models with vLLM (OpenAI-compatible API).
# Do not pass --reasoning-parser: the experiments use the full raw output.
#   NONTHINKING_GPU=0 THINKING_GPU=1 bash scripts/reproduce/serve.sh
set -euo pipefail

NONTHINKING_MODEL=${NONTHINKING_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
THINKING_MODEL=${THINKING_MODEL:-Qwen/Qwen3-4B-Thinking-2507}
NONTHINKING_PORT=${NONTHINKING_PORT:-8001}
THINKING_PORT=${THINKING_PORT:-8002}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

CUDA_VISIBLE_DEVICES=${NONTHINKING_GPU:-0} vllm serve "$NONTHINKING_MODEL" \
    --port "$NONTHINKING_PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.9 &
CUDA_VISIBLE_DEVICES=${THINKING_GPU:-1} vllm serve "$THINKING_MODEL" \
    --port "$THINKING_PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.9 &
wait
