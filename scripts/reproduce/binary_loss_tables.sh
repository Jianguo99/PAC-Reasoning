#!/usr/bin/env bash
# Step 3b: binary-loss results - Table 1 (MATH-500, epsilon = 0.03) and
# Table 5 (MATH-500 and ZebraLogic, epsilon = 0.03 / 0.08), alpha = 0.05, 100 runs.
set -euo pipefail

RESULTS=${RESULTS:-results}

run() {  # dataset ratio epsilon score
    python scripts/run_pac.py \
        --thinking "$RESULTS/$1/thinking.jsonl" --nonthinking "$RESULTS/$1/nonthinking.jsonl" \
        --loss binary --score "$4" --epsilon "$3" --calib-ratio "$2" \
        --methods pac naive nonthinking --alpha 0.05 --pi 0.5 --n-trials 100 \
        --output "$RESULTS/$1/binary_$4.json"
}

# See semantic_loss_figure.sh for why MATH-500 uses the "logprob" variant.
run math500    0.6 0.03 logprob
run math500    0.6 0.03 verbalized
run zebralogic 0.5 0.08 logits
run zebralogic 0.5 0.08 verbalized

# Table 1 baselines: always answer with the thinking model under a cheaper prompt.
for style in cod nothinking; do
    python scripts/run_pac.py \
        --thinking "$RESULTS/math500/thinking.jsonl" --nonthinking "$RESULTS/math500/thinking_$style.jsonl" \
        --loss binary --score logprob --epsilon 0.03 --calib-ratio 0.6 --methods nonthinking \
        --n-trials 100 --output "$RESULTS/math500/binary_baseline_$style.json"
done
