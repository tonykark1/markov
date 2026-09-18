"""Small, transparent tools for factor-regime HMM research."""

from .first_passage import expected_hitting_time, first_passage_probability
from .hmm import HMMFit, fit_hmm

__all__ = [
    "HMMFit",
    "fit_hmm",
    "first_passage_probability",
    "expected_hitting_time",
]
