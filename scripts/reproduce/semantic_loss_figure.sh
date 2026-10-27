#!/usr/bin/env bash
# Step 3a: Figure 1 - error, ECP and STP under the semantic loss (alpha = 0.05, 100 runs).
set -euo pipefail

RESULTS=${RESULTS:-results}
FIGURES=${FIGURES:-figures}

# dataset  calibration ratio (Table 3)  epsilon range  logits-score variant
# The MATH-500 runs in the paper only recorded mean token log-probabilities, so
# their logits score is 1 - mean log-prob ("logprob"); the others use 1 - mean prob.
while read -r dataset ratio eps_start eps_stop eps_step logits; do
    for score in "$logits" verbalized; do
        python scripts/run_pac.py \
            --thinking "$RESULTS/$dataset/thinking.jsonl" \
            --nonthinking "$RESULTS/$dataset/nonthinking.jsonl" \
            --loss semantic --semantic-loss-file "$RESULTS/$dataset/semantic_loss.jsonl" \
            --score "$score" --epsilon-range "$eps_start" "$eps_stop" "$eps_step" \
            --calib-ratio "$ratio" --alpha 0.05 --pi 0.5 --n-trials 100 \
            --output "$RESULTS/$dataset/semantic_$score.json"
    done
    python scripts/plot_curves.py \
        --results "$RESULTS/$dataset/semantic_$logits.json" "$RESULTS/$dataset/semantic_verbalized.json" \
        --labels "Logits score" "Verbalized score" \
        --output-dir "$FIGURES" --prefix "${dataset}_semantic"
done <<'EOF'
math500    0.6 0.035 0.055 0.002 logprob
zebralogic 0.5 0.040 0.060 0.002 logits
arena-hard 0.6 0.040 0.120 0.005 logits
EOF
