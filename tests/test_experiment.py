import numpy as np
import pytest

from pac_reasoning.experiment import RoutingData, evaluate_routing, run_trials, split_indices, summarize
from pac_reasoning.losses import binary_loss, semantic_loss


def test_split_is_deterministic_and_disjoint():
    a, b = split_indices(100, 0.6, seed=3)
    c, d = split_indices(100, 0.6, seed=3)
    assert len(a) == 60 and len(b) == 40
    assert set(a).isdisjoint(b) and set(a) | set(b) == set(range(100))
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)


def make_data():
    return RoutingData(
        losses=[1, 0, 1, 0],
        scores=[0.9, 0.1, 0.2, 0.8],
        thinking_tokens=[100, 100, 100, 100],
        nonthinking_tokens=[10, 10, 10, 10],
        thinking_latency=[10, 10, 10, 10],
        nonthinking_latency=[1, 1, 1, 1],
        thinking_correct=[1, 1, 1, 0],
        nonthinking_correct=[0, 1, 0, 0],
    )


def test_evaluate_routing_accounting():
    data = make_data()
    idx = np.arange(4)
    routed = np.array([True, False, False, True])
    m = evaluate_routing(data, idx, routed)
    assert m["error"] == pytest.approx(0.25)  # only query 2 keeps a lossy answer
    assert m["ecp"] == pytest.approx(50.0)
    # cost = 4 * 10 (non-thinking always runs) + 2 * 100 (routed) = 240 of 400
    assert m["stp"] == pytest.approx(40.0)
    assert m["slp"] == pytest.approx(40.0)
    assert m["accuracy"] == pytest.approx(0.5)

    pre = evaluate_routing(data, idx, routed, score_before_generation=True)
    # routed queries skip the non-thinking model: 2 * 10 + 2 * 100 = 220 of 400
    assert pre["stp"] == pytest.approx(45.0)


def test_run_trials_and_summary():
    rng = np.random.default_rng(0)
    n = 300
    scores = rng.random(n)
    data = RoutingData(
        losses=(rng.random(n) < 0.3 * scores).astype(float),
        scores=scores,
        thinking_tokens=rng.integers(1000, 5000, n),
        nonthinking_tokens=rng.integers(100, 500, n),
    )
    trials = run_trials(data, 0.05, n_trials=20, calib_ratio=0.5)
    assert [t["seed"] for t in trials] == list(range(20))
    s = summarize(trials)
    assert s["n_trials"] == 20 and 0 <= s["ecp"] <= 100 and "error_std" in s
    assert run_trials(data, 0.05, n_trials=3) == run_trials(data, 0.05, n_trials=3)


def test_losses():
    np.testing.assert_array_equal(binary_loss([1, 1, 0, 0], [1, 0, 1, 0]), [0, 1, 0, 0])
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[1.0, 0.0], [-1.0, -1.0]])
    np.testing.assert_allclose(semantic_loss(a, b), [0.0, 2.0])
