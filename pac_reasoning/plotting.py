"""Figures in the style of the paper (error / ECP / STP against epsilon)."""

from __future__ import annotations

import os

import numpy as np

# Orange then blue, as in Figure 1 (logits score, verbalized score).
PALETTE = ["C1", "C0", "C2", "C3", "C4", "C5", "C6", "C7"]

METRICS = {
    "error": "Error",
    "ecp": "ECP (%)",
    "stp": "STP (%)",
}


def plot_curves(curves: dict, metric: str, path: str | None = None, ax=None):
    """Plot ``metric`` against epsilon, one line per entry of ``curves``.

    Args:
        curves: ``{label: [summary, ...]}`` where each summary comes from
            :func:`pac_reasoning.experiment.summarize`.
        metric: ``"error"``, ``"ecp"`` or ``"stp"``.  The error plot adds the
            diagonal ``error = epsilon``.
    """
    import matplotlib.pyplot as plt

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 4.2))
    all_eps = []
    for i, (label, summaries) in enumerate(curves.items()):
        eps = np.array([s["epsilon"] for s in summaries])
        mean = np.array([s[metric] for s in summaries])
        std = np.array([s[f"{metric}_std"] for s in summaries])
        color = PALETTE[i % len(PALETTE)]
        ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(eps, mean, color=color, lw=2.5, label=label)
        all_eps.extend(eps)
    if metric == "error" and all_eps:
        grid = np.array(sorted(set(all_eps)))
        ax.plot(grid, grid, "--", color="black", lw=2, alpha=0.7, label="Diagonal")

    ax.set_xlabel(r"$\epsilon$", fontsize=22)
    ax.set_ylabel(METRICS.get(metric, metric), fontsize=22)
    ax.tick_params(labelsize=15)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(14)
    if metric == "error":
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(14)
    ax.grid(True, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=13, framealpha=0.85)

    if own_figure:
        fig.tight_layout()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
    return ax
