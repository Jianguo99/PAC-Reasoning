"""PAC reasoning on synthetic data: calibrate a threshold, route, and check the risk.

    python examples/quickstart.py
"""

import numpy as np

from pac_reasoning import RoutingData, binary_loss, pac_calibrate, run_trials, split_indices, summarize

rng = np.random.default_rng(0)
n = 1000

# The non-thinking model is wrong more often when it is uncertain.
uncertainty = rng.beta(2, 5, n)
nonthinking_correct = rng.random(n) > 0.4 * uncertainty
thinking_correct = rng.random(n) > 0.05
thinking_tokens = rng.integers(2000, 8000, n)
nonthinking_tokens = rng.integers(200, 800, n)

loss = binary_loss(thinking_correct, nonthinking_correct)
print(f"loss of always using the non-thinking model: {loss.mean():.4f}")

# One calibration/test split.
calib_idx, test_idx = split_indices(n, calib_ratio=0.5, seed=0)
calib = pac_calibrate(loss[calib_idx], uncertainty[calib_idx], epsilon=0.05, alpha=0.05, pi=0.5, rng=0)
routed = calib.route(uncertainty[test_idx])
print(f"threshold u_hat = {calib.threshold:.4f}")
print(f"test error = {np.mean(loss[test_idx] * ~routed):.4f} (target 0.05), "
      f"thinking model called on {routed.mean():.1%} of the test queries")

# 100 random splits, as in the paper.
data = RoutingData(losses=loss, scores=uncertainty,
                   thinking_tokens=thinking_tokens, nonthinking_tokens=nonthinking_tokens,
                   thinking_correct=thinking_correct, nonthinking_correct=nonthinking_correct)
for eps in (0.02, 0.05, 0.1):
    s = summarize(run_trials(data, eps, n_trials=100, calib_ratio=0.5))
    print(f"epsilon={eps:.2f}: error {s['error']:.4f} ± {s['error_std']:.4f} | "
          f"ECP {s['ecp']:.1f}% | STP {s['stp']:.1f}% | Pass@1 {s['accuracy']:.3f}")
