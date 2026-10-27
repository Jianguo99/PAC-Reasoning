import numpy as np
import pytest
from scipy.stats import norm

from pac_reasoning.calibration import naive_calibrate, nonthinking_only, pac_calibrate


def reference_pac_threshold(losses, scores, epsilon, alpha, pi, m, seed):
    """Direct (loop) implementation of Algorithm 1, as in the paper."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(losses), size=m, replace=True)
    weights = (rng.random(m) < pi) / pi
    z = norm.ppf(1 - alpha)
    grid = np.sort(scores)
    ucb = []
    for u in grid:
        x = losses[draws] * weights * (scores[draws] <= u)
        ucb.append(x.mean() + z * x.std(ddof=1) / np.sqrt(m))
    ok = np.flatnonzero(np.array(ucb) <= epsilon)
    return grid[ok[-1]] if ok.size else -np.inf


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    losses = (rng.random(n) < 0.3 * scores).astype(float)  # P(loss | U = u) = 0.3 u
    return losses, scores


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("pi", [0.3, 0.5, 1.0])
def test_matches_reference(seed, pi):
    losses, scores = synthetic(300, seed)
    for eps in (0.02, 0.05, 0.1):
        m = int(round(300 / pi))
        got = pac_calibrate(losses, scores, eps, alpha=0.05, pi=pi, rng=seed).threshold
        assert got == reference_pac_threshold(losses, scores, eps, 0.05, pi, m, seed)


def test_threshold_grows_with_epsilon():
    losses, scores = synthetic(400, 0)
    thresholds = [pac_calibrate(losses, scores, eps, rng=1).threshold for eps in np.linspace(0.005, 0.2, 20)]
    assert np.all(np.diff(thresholds) >= 0)


def test_zero_loss_keeps_everything_but_the_top_score():
    scores = np.linspace(0, 1, 50)
    calib = pac_calibrate(np.zeros(50), scores, 0.01, rng=0)
    assert calib.threshold == scores.max()


def test_infeasible_routes_everything():
    calib = pac_calibrate(np.ones(50), np.linspace(0, 1, 50), 1e-6, rng=0)
    assert calib.threshold == -np.inf
    assert calib.route([0.0, 0.5, 1.0]).all()


def test_hoeffding_is_more_conservative():
    losses, scores = synthetic(500, 3)
    clt = pac_calibrate(losses, scores, 0.08, rng=0).threshold
    hoeffding = pac_calibrate(losses, scores, 0.08, bound="hoeffding", rng=0).threshold
    assert hoeffding <= clt


def test_bound_controls_risk_on_the_calibration_set():
    # The m draws are taken with replacement from the calibration set, so the
    # bound holds (up to the CLT approximation) for the risk averaged over the
    # calibration points themselves.
    epsilon, alpha, n_trials = 0.05, 0.05, 300
    violations = 0
    for t in range(n_trials):
        losses, scores = synthetic(500, 1000 + t)
        u = pac_calibrate(losses, scores, epsilon, alpha=alpha, rng=t).threshold
        violations += np.mean(losses * (scores < u)) > epsilon
    assert violations / n_trials <= alpha + 0.04


def test_naive_matches_definition():
    losses, scores = synthetic(200, 7)
    eps = 0.04
    calib = naive_calibrate(losses, scores, eps)
    grid = np.append(np.unique(scores), np.nextafter(scores.max(), np.inf))
    risks = np.array([np.mean(losses * (scores < u)) for u in grid])
    assert calib.threshold == grid[np.flatnonzero(risks <= eps)[-1]]
    np.testing.assert_allclose(calib.risk_bound, risks)


def test_nonthinking_only_never_routes():
    calib = nonthinking_only(np.ones(10), np.linspace(0, 1, 10), 0.1)
    assert not calib.route(np.linspace(0, 1, 10)).any()


def test_input_validation():
    with pytest.raises(ValueError):
        pac_calibrate([0, 1], [0.1], 0.1)
    with pytest.raises(ValueError):
        pac_calibrate([0, 1], [0.1, 0.2], 0.1, pi=0)
    with pytest.raises(ValueError):
        pac_calibrate([0, 1], [0.1, np.nan], 0.1)
