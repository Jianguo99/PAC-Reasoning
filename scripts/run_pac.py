"""Run PAC reasoning (and baselines) over repeated random calibration/test splits.

Binary loss on MATH-500 at epsilon = 0.03 (Table 1):

    python scripts/run_pac.py \
        --thinking results/math500/thinking.jsonl \
        --nonthinking results/math500/nonthinking.jsonl \
        --loss binary --score logits --epsilon 0.03 --calib-ratio 0.6 \
        --methods pac naive --output results/math500/binary_logits.json

Semantic loss over a range of epsilon (Figure 1):

    python scripts/run_pac.py ... --loss semantic \
        --semantic-loss-file results/arena-hard/semantic_loss.jsonl \
        --epsilon-range 0.04 0.12 0.005 --calib-ratio 0.6
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pac_reasoning.calibration import CALIBRATORS
from pac_reasoning.experiment import run_trials, summarize
from pac_reasoning.records import SCORES, load_routing_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("inputs")
    io.add_argument("--thinking", required=True, help="generation records of the thinking model")
    io.add_argument("--nonthinking", required=True, help="generation records of the non-thinking model")
    io.add_argument("--loss", choices=("binary", "semantic"), default="binary")
    io.add_argument("--semantic-loss-file", help="output of scripts/semantic_loss.py")
    io.add_argument("--score", choices=SCORES, default="logits",
                    help="logits: 1 - mean token prob; logprob: 1 - mean token log-prob; "
                         "verbalized: scripts/verbalized_confidence.py; file: --score-file")
    io.add_argument("--score-file", help="JSONL with {id, score} (or {id, verbalized_uncertainty})")
    io.add_argument("--score-before-generation", action="store_true",
                    help="the score only looks at the query (e.g. a router), so routed queries "
                         "do not pay for the non-thinking answer")

    eps = p.add_mutually_exclusive_group(required=True)
    eps.add_argument("--epsilon", type=float, nargs="+", help="target risk level(s)")
    eps.add_argument("--epsilon-range", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
                     help="numpy.arange(START, STOP, STEP)")

    alg = p.add_argument_group("calibration")
    alg.add_argument("--methods", nargs="+", default=["pac"], choices=sorted(CALIBRATORS))
    alg.add_argument("--alpha", type=float, default=0.05)
    alg.add_argument("--pi", type=float, default=0.5, help="auditing probability")
    alg.add_argument("--m", type=int, help="number of draws (default n / pi)")
    alg.add_argument("--bound", choices=("clt", "hoeffding"), default="clt")
    alg.add_argument("--calib-ratio", type=float, default=0.5)
    alg.add_argument("--n-trials", type=int, default=100)
    alg.add_argument("--seed", type=int, default=0)

    p.add_argument("--output", help="JSON file with the configuration, summaries and per-trial results")
    return p.parse_args()


def fmt(summary, key, scale=1.0, digits=4):
    if key not in summary:
        return "-"
    return f"{summary[key] * scale:.{digits}f} ± {summary[key + '_std'] * scale:.{digits}f}"


def main():
    args = parse_args()
    data = load_routing_data(
        args.thinking, args.nonthinking, loss=args.loss, score=args.score,
        semantic_loss_path=args.semantic_loss_file, score_path=args.score_file,
    )
    epsilons = args.epsilon if args.epsilon else np.arange(*args.epsilon_range).round(10).tolist()
    print(f"{len(data)} queries | {args.loss} loss | {args.score} score | "
          f"non-thinking-only loss {data.losses.mean():.4f}")

    header = f"{'method':12s} {'epsilon':>8s}  {'error':>17s}  {'ECP (%)':>15s}  {'STP (%)':>15s}  {'Pass@1':>15s}"
    print(header)
    print("-" * len(header))

    results = []
    for method in args.methods:
        kwargs = {"method": method, "calib_ratio": args.calib_ratio,
                  "score_before_generation": args.score_before_generation}
        if method == "pac":
            kwargs.update(alpha=args.alpha, pi=args.pi, m=args.m, bound=args.bound)
        for eps in epsilons:
            trials = run_trials(data, eps, n_trials=args.n_trials, seed=args.seed, **kwargs)
            s = summarize(trials)
            results.append({"method": method, "summary": s, "trials": trials})
            print(f"{method:12s} {eps:8.4f}  {fmt(s, 'error'):>17s}  {fmt(s, 'ecp', digits=2):>15s}  "
                  f"{fmt(s, 'stp', digits=2):>15s}  {fmt(s, 'accuracy', digits=3):>15s}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"config": vars(args), "n_queries": len(data), "results": results}, f, indent=1)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
