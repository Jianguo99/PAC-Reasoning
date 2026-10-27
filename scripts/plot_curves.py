"""Plot error / ECP / STP against epsilon from one or more ``run_pac.py`` outputs.

    python scripts/plot_curves.py \
        --results results/arena-hard/semantic_logits.json results/arena-hard/semantic_verbalized.json \
        --labels "Logits score" "Verbalized score" \
        --output-dir figures --prefix arena-hard_semantic

writes ``figures/arena-hard_semantic_{error,ecp,stp}.pdf``.
"""

from __future__ import annotations

import argparse
import json
import os

from pac_reasoning.plotting import METRICS, plot_curves


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", help="one per results file (default: file name)")
    p.add_argument("--method", default="pac")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--prefix", default="curves")
    p.add_argument("--format", default="pdf")
    args = p.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(r))[0] for r in args.results]
    if len(labels) != len(args.results):
        p.error("--labels must have one entry per results file")

    curves = {}
    for label, path in zip(labels, args.results):
        with open(path) as f:
            results = json.load(f)["results"]
        summaries = [r["summary"] for r in results if r["method"] == args.method]
        if not summaries:
            p.error(f"{path} has no results for method {args.method!r}")
        curves[label] = sorted(summaries, key=lambda s: s["epsilon"])

    for metric in METRICS:
        out = os.path.join(args.output_dir, f"{args.prefix}_{metric}.{args.format}")
        plot_curves(curves, metric, path=out)
        print("saved", out)


if __name__ == "__main__":
    main()
