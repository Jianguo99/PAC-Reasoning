#!/usr/bin/env bash
# Step 1: answers of both models (plus the CoD / NoThinking baselines) on every benchmark.
# Needs the two servers from serve.sh.  ZebraLogic grading needs access to the gated
# allenai/ZebraLogicBench-private dataset; Arena-Hard needs ARENA_HARD_QUESTIONS.
set -euo pipefail

RESULTS=${RESULTS:-results}
NONTHINKING_MODEL=${NONTHINKING_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
THINKING_MODEL=${THINKING_MODEL:-Qwen/Qwen3-4B-Thinking-2507}
NONTHINKING_URL=${NONTHINKING_URL:-http://localhost:8001/v1}
THINKING_URL=${THINKING_URL:-http://localhost:8002/v1}
ARENA_HARD_QUESTIONS=${ARENA_HARD_QUESTIONS:-data/arena-hard-v2.0/question.jsonl}
DATASETS=${DATASETS:-"math500 zebralogic arena-hard"}

# Recommended sampling settings of the Qwen3-2507 models (Appendix G).
NONTHINKING_SAMPLING="--temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0"
THINKING_SAMPLING="--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0"

# Per-benchmark generation settings used in the paper's runs.
extra_args() {
    case "$1" in
        math500|aime|gpqa) echo "--max-tokens 38912 --presence-penalty 1.5 --frequency-penalty 0.2" ;;
        zebralogic)        echo "--max-tokens 32768" ;;
        arena-hard)        echo "--max-tokens 32000 --presence-penalty 0.5 --data-file $ARENA_HARD_QUESTIONS" ;;
    esac
}

for dataset in $DATASETS; do
    args=$(extra_args "$dataset")
    python scripts/generate.py --dataset "$dataset" --model "$NONTHINKING_MODEL" --base-url "$NONTHINKING_URL" \
        $NONTHINKING_SAMPLING $args --output "$RESULTS/$dataset/nonthinking.jsonl"
    python scripts/generate.py --dataset "$dataset" --model "$THINKING_MODEL" --base-url "$THINKING_URL" \
        $THINKING_SAMPLING $args --output "$RESULTS/$dataset/thinking.jsonl"
done

# Efficient-reasoning baselines of Table 1: the thinking model with a cheaper prompt.
for style in cod nothinking; do
    python scripts/generate.py --dataset math500 --model "$THINKING_MODEL" --base-url "$THINKING_URL" \
        $THINKING_SAMPLING $(extra_args math500) --prompt-style "$style" \
        --output "$RESULTS/math500/thinking_$style.jsonl"
done
