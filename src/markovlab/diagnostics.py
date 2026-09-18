"""Diagnostics that separate state uncertainty from model uncertainty."""

from __future__ import annotations

import numpy as np


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Row-wise entropy normalized to [0, 1]."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must be T x K with K >= 2")
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def hard_state_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of dates for which aligned hard states agree."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("state arrays must have the same shape")
    return float(np.mean(a == b))


def expected_durations(transition: np.ndarray) -> np.ndarray:
    """Geometric-duration expectation 1 / (1 - p_ii)."""
    p = np.asarray(transition, dtype=float)
    diagonal = np.diag(p)
    return 1.0 / np.maximum(1.0 - diagonal, 1e-15)


def model_disagreement(probability_stack: np.ndarray) -> np.ndarray:
    """Cross-model SD for a common aligned state probability.

    Input shape is M x T, where M is the number of specifications.
    """
    x = np.asarray(probability_stack, dtype=float)
    if x.ndim != 2:
        raise ValueError("probability_stack must have shape models x time")
    return np.std(x, axis=0, ddof=1) if x.shape[0] > 1 else np.zeros(x.shape[1])
