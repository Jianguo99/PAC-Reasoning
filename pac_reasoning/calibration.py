"""Threshold calibration for PAC reasoning.

Setup. For every query we have
  * ``U``  - an uncertainty score of the non-thinking model's answer;
  * ``l``  - the loss incurred if the non-thinking answer is kept instead of
             the thinking model's answer.

The composite model answers with the non-thinking model when ``U < u_hat`` and
falls back to the thinking model when ``U >= u_hat``.  Its risk at threshold
``u`` is ``L(u) = E[l * 1{U <= u}]``.

``pac_calibrate`` picks ``u_hat = max{u : UCB_alpha(u) <= epsilon}`` from an
importance-sampled upper confidence bound (Algorithm 1 of the paper; the
Hoeffding variant is Algorithm 3), so that the risk of the composite model is
at most ``epsilon`` with probability at least ``1 - alpha`` (see Section 3 of
the paper for the exact statement and assumptions).

``naive_calibrate`` is the "naive control" baseline: it picks the largest
threshold whose *empirical* calibration risk is at most ``epsilon`` and has no
statistical guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Calibration:
    """Result of a calibration run."""

    threshold: float
    """Queries with ``score >= threshold`` are routed to the thinking model."""

    grid: np.ndarray
    """Candidate thresholds that were evaluated."""

    risk_bound: np.ndarray
    """Risk upper bound (PAC) or empirical risk (naive) at each grid value."""

    def route(self, scores) -> np.ndarray:
        """Boolean mask, ``True`` where the thinking model should be called."""
        return np.asarray(scores, dtype=float) >= self.threshold


def _check_inputs(losses, scores):
    losses = np.asarray(losses, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if losses.ndim != 1 or losses.shape != scores.shape:
        raise ValueError("losses and scores must be 1-D arrays of the same length")
    if losses.size < 2:
        raise ValueError("need at least two calibration points")
    if np.any(~np.isfinite(scores)) or np.any(~np.isfinite(losses)):
        raise ValueError("losses and scores must be finite")
    return losses, scores


def pac_calibrate(
    losses,
    scores,
    epsilon: float,
    alpha: float = 0.05,
    pi: float = 0.5,
    m: int | None = None,
    bound: str = "clt",
    loss_bound: float = 1.0,
    rng=None,
) -> Calibration:
    """Choose the routing threshold with a PAC guarantee.

    Args:
        losses: loss of keeping the non-thinking answer, one per calibration point.
        scores: uncertainty score of the non-thinking answer (higher = less sure).
        epsilon: target risk level.
        alpha: allowed failure probability (confidence level is ``1 - alpha``).
        pi: probability of auditing a sampled point, i.e. of querying the
            thinking model for its label.  ``pi = 1`` audits every point.
        m: number of draws with replacement.  Defaults to ``n / pi``.
        bound: ``"clt"`` (Algorithm 1) or ``"hoeffding"`` (Algorithm 3).
        loss_bound: upper bound ``B`` on the loss, only used by ``"hoeffding"``.
        rng: seed or ``numpy.random.Generator``.

    Returns:
        A :class:`Calibration`.  If no threshold satisfies the bound, the
        threshold is ``-inf`` and every query is routed to the thinking model.
    """
    losses, scores = _check_inputs(losses, scores)
    if not 0 < pi <= 1:
        raise ValueError("pi must be in (0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if bound not in ("clt", "hoeffding"):
        raise ValueError(f"unknown bound {bound!r}")

    n = losses.size
    m = int(round(n / pi)) if m is None else int(m)
    if m < 2:
        raise ValueError("m must be at least 2")
    rng = np.random.default_rng(rng)

    # Steps 3-12: sample with replacement, audit each draw with prob. pi and
    # re-weight the audited losses by 1/pi so that E[z_j * 1{U <= u}] = L(u).
    draws = rng.choice(n, size=m, replace=True)
    audited = rng.random(m) < pi
    draw_losses = losses[draws]
    if bound == "hoeffding":
        draw_losses = np.minimum(draw_losses, loss_bound)
    z = draw_losses * audited / pi
    draw_scores = scores[draws]

    # Z_j(u) = z_j * 1{U_{i_j} <= u}.  Sorting the draws by score turns the
    # sum over j into a prefix sum, so every grid value costs O(log m).
    grid = np.sort(scores)
    order = np.argsort(draw_scores, kind="stable")
    z_sorted = z[order]
    n_below = np.searchsorted(draw_scores[order], grid, side="right")
    mean = np.concatenate(([0.0], np.cumsum(z_sorted)))[n_below] / m

    if bound == "clt":
        sum_sq = np.concatenate(([0.0], np.cumsum(z_sorted**2)))[n_below]
        var = np.clip((sum_sq - m * mean**2) / (m - 1), 0.0, None)
        risk_bound = mean + norm.ppf(1 - alpha) * np.sqrt(var) / np.sqrt(m)
    else:
        r = loss_bound / pi
        risk_bound = mean + np.sqrt(r**2 * np.log(2 / alpha) / (2 * m))

    feasible = np.flatnonzero(risk_bound <= epsilon)
    threshold = float(grid[feasible[-1]]) if feasible.size else -np.inf
    return Calibration(threshold=threshold, grid=grid, risk_bound=risk_bound)


def naive_calibrate(losses, scores, epsilon: float) -> Calibration:
    """Naive control: largest threshold with empirical calibration risk <= epsilon.

    This baseline plugs in the empirical risk instead of an upper confidence
    bound, so it does not control the risk with high probability.
    """
    losses, scores = _check_inputs(losses, scores)
    n = losses.size

    # The last grid value lies just above max(U), i.e. "never call the thinking model".
    grid = np.unique(scores)
    grid = np.append(grid, np.nextafter(grid[-1], np.inf))

    order = np.argsort(scores, kind="stable")
    n_below = np.searchsorted(scores[order], grid, side="left")  # #{U < u}
    risk = np.concatenate(([0.0], np.cumsum(losses[order])))[n_below] / n

    feasible = np.flatnonzero(risk <= epsilon)
    idx = feasible[-1] if feasible.size else 0
    return Calibration(threshold=float(grid[idx]), grid=grid, risk_bound=risk)


def nonthinking_only(losses, scores, epsilon: float) -> Calibration:
    """Never call the thinking model.

    Used for baselines that only change how the cheap model is prompted
    (NoThinking, Chain of Draft): pass their outputs as the non-thinking model.
    """
    losses, _ = _check_inputs(losses, scores)
    return Calibration(threshold=np.inf, grid=np.array([np.inf]), risk_bound=np.array([losses.mean()]))


CALIBRATORS = {
    "pac": pac_calibrate,
    "naive": naive_calibrate,
    "nonthinking": nonthinking_only,
}
