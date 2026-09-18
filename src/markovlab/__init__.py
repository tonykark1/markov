"""Robustness-first hidden Markov model utilities for factor research."""

from .alignment import align_by_mean_distance, reorder_states
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

__all__ = [
    "HMMFit",
    "fit_hmm",
    "align_by_mean_distance",
    "reorder_states",
    "normalized_entropy",
    "hard_state_agreement",
    "model_disagreement",
    "expected_durations",
    "first_passage_probability",
    "mixture_first_passage_probability",
    "expected_hitting_time",
]
