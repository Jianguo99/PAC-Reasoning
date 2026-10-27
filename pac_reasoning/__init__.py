"""PAC reasoning: efficient reasoning with a provable bound on the performance loss."""

from .calibration import Calibration, naive_calibrate, nonthinking_only, pac_calibrate
from .experiment import RoutingData, evaluate_routing, run_trial, run_trials, split_indices, summarize
from .losses import binary_loss, semantic_loss
from .records import load_routing_data

__version__ = "1.0.0"

__all__ = [
    "Calibration",
    "RoutingData",
    "binary_loss",
    "evaluate_routing",
    "load_routing_data",
    "naive_calibrate",
    "nonthinking_only",
    "pac_calibrate",
    "run_trial",
    "run_trials",
    "semantic_loss",
    "split_indices",
    "summarize",
]
