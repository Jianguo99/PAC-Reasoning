"""Repeated calibration/test experiments and the evaluation metrics.

Metrics (all computed on the test split):
  * ``error``  - empirical risk of the composite model, mean of ``l * 1{U < u_hat}``.
  * ``ecp``    - expert call percentage, share of test queries routed to the thinking model.
  * ``stp``    - saved token percentage relative to always using the thinking model.
                 The non-thinking model always answers first, so a routed query
                 costs both models' tokens.
  * ``slp``    - saved latency percentage, same accounting as ``stp``.
  * ``accuracy`` - Pass@1 of the composite model (only when correctness labels exist).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import CALIBRATORS


@dataclass
class RoutingData:
    """Per-query arrays for one (dataset, thinking model, non-thinking model) triple."""

    losses: np.ndarray
    scores: np.ndarray
    thinking_tokens: np.ndarray
    nonthinking_tokens: np.ndarray
    thinking_latency: np.ndarray | None = None
    nonthinking_latency: np.ndarray | None = None
    thinking_correct: np.ndarray | None = None
    nonthinking_correct: np.ndarray | None = None
    ids: list | None = field(default=None, repr=False)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and name != "ids":
                setattr(self, name, np.asarray(value, dtype=float))
        n = len(self.losses)
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"{name} has length {len(value)}, expected {n}")

    def __len__(self):
        return len(self.losses)


def split_indices(n: int, calib_ratio: float, seed: int):
    """Random calibration/test split; returns (calib_idx, test_idx)."""
    if not 0 < calib_ratio < 1:
        raise ValueError("calib_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_calib = int(n * calib_ratio)
    return idx[:n_calib], idx[n_calib:]


def _saving(full_cost, actual_cost) -> float:
    return float((full_cost - actual_cost) / full_cost * 100) if full_cost > 0 else float("nan")


def evaluate_routing(data: RoutingData, test_idx, routed, score_before_generation: bool = False) -> dict:
    """Metrics of a routing decision on the queries ``test_idx``.

    Args:
        routed: boolean mask over ``test_idx``, ``True`` = thinking model answers.
        score_before_generation: set for scores computed from the query alone
            (e.g. a learned router).  Routed queries then cost only the
            thinking model's tokens.
    """
    routed = np.asarray(routed, dtype=bool)
    weak_share = np.where(routed & score_before_generation, 0.0, 1.0)

    def cost(strong, weak):
        return float(np.sum(weak[test_idx] * weak_share + strong[test_idx] * routed))

    out = {
        "error": float(np.mean(data.losses[test_idx] * ~routed)),
        "nonthinking_error": float(np.mean(data.losses[test_idx])),
        "ecp": float(routed.mean() * 100),
        "stp": _saving(
            data.thinking_tokens[test_idx].sum(), cost(data.thinking_tokens, data.nonthinking_tokens)
        ),
    }
    if data.thinking_latency is not None and data.nonthinking_latency is not None:
        out["slp"] = _saving(
            data.thinking_latency[test_idx].sum(), cost(data.thinking_latency, data.nonthinking_latency)
        )
    if data.thinking_correct is not None and data.nonthinking_correct is not None:
        composite = np.where(routed, data.thinking_correct[test_idx], data.nonthinking_correct[test_idx])
        out["accuracy"] = float(composite.mean())
        out["thinking_accuracy"] = float(data.thinking_correct[test_idx].mean())
        out["nonthinking_accuracy"] = float(data.nonthinking_correct[test_idx].mean())
    return out


def run_trial(
    data: RoutingData,
    epsilon: float,
    seed: int,
    method: str = "pac",
    calib_ratio: float = 0.5,
    score_before_generation: bool = False,
    **calib_kwargs,
) -> dict:
    """One random split: calibrate on the calibration part, evaluate on the rest."""
    calib_idx, test_idx = split_indices(len(data), calib_ratio, seed)
    calibrate = CALIBRATORS[method]
    if method == "pac":
        calib_kwargs.setdefault("rng", seed)
    calib = calibrate(data.losses[calib_idx], data.scores[calib_idx], epsilon, **calib_kwargs)
    routed = calib.route(data.scores[test_idx])
    result = evaluate_routing(data, test_idx, routed, score_before_generation)
    result.update(threshold=calib.threshold, seed=seed, epsilon=epsilon)
    return result


def run_trials(data: RoutingData, epsilon: float, n_trials: int = 100, seed: int = 0, **kwargs) -> list[dict]:
    """``n_trials`` independent splits with seeds ``seed, seed + 1, ...``."""
    return [run_trial(data, epsilon, seed=seed + t, **kwargs) for t in range(n_trials)]


def summarize(trials: list[dict]) -> dict:
    """Mean and standard deviation of every numeric metric across trials."""
    keys = [k for k in trials[0] if k not in ("seed", "epsilon")]
    summary = {"epsilon": trials[0]["epsilon"], "n_trials": len(trials)}
    for k in keys:
        values = np.array([t[k] for t in trials], dtype=float)
        if np.all(np.isfinite(values)):
            summary[k], summary[f"{k}_std"] = float(np.mean(values)), float(np.std(values))
        else:  # e.g. thresholds of +-inf ("never" / "always" call the thinking model)
            summary[k], summary[f"{k}_std"] = None, None
    return summary
