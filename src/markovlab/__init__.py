"""Robustness-first hidden Markov model utilities for factor research."""

from .alignment import (
    AlignmentMetric,
    align_by_distribution_distance,
    align_by_mean_distance,
    distribution_distance_matrix,
    reorder_states,
)
from .diagnostics import (
    expected_durations,
    hard_state_agreement,
    model_disagreement,
    normalized_entropy,
)
from .first_passage import (
    expected_hitting_time,
    first_passage_probability,
    mixture_first_passage_probability,
)
from .hmm import HMMFit, fit_hmm
from .inference import HMMInference, forecast_state_probabilities, infer_hmm
from .parallel import joint_probabilities, joint_state_labels, joint_transition
from .walkforward import WalkForwardResult, expanding_walk_forward

__all__ = [
    "AlignmentMetric",
    "HMMFit",
    "HMMInference",
    "WalkForwardResult",
    "fit_hmm",
    "infer_hmm",
    "forecast_state_probabilities",
    "expanding_walk_forward",
    "align_by_mean_distance",
    "align_by_distribution_distance",
    "distribution_distance_matrix",
    "reorder_states",
    "joint_probabilities",
    "joint_transition",
    "joint_state_labels",
    "normalized_entropy",
    "hard_state_agreement",
    "model_disagreement",
    "expected_durations",
    "first_passage_probability",
    "mixture_first_passage_probability",
    "expected_hitting_time",
]
