#!/usr/bin/env bash
# Step 2: verbalized uncertainty scores and semantic losses (offline vLLM, one GPU each).
# Stop the servers from serve.sh first, or point CUDA_VISIBLE_DEVICES at a free GPU.
set -euo pipefail

RESULTS=${RESULTS:-results}
NONTHINKING_MODEL=${NONTHINKING_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
DATASETS=${DATASETS:-"math500 zebralogic arena-hard"}

for dataset in $DATASETS; do
    python scripts/verbalized_confidence.py --model "$NONTHINKING_MODEL" \
        --input "$RESULTS/$dataset/nonthinking.jsonl"
    python scripts/semantic_loss.py --model Qwen/Qwen3-Embedding-4B \
        --thinking "$RESULTS/$dataset/thinking.jsonl" \
        --nonthinking "$RESULTS/$dataset/nonthinking.jsonl" \
        --output "$RESULTS/$dataset/semantic_loss.jsonl"
done
