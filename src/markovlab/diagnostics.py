"""Diagnostics that separate state uncertainty from model uncertainty."""

from __future__ import annotations

import numpy as np


def _validate_transition(transition: np.ndarray) -> np.ndarray:
    p = np.asarray(transition, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] == 0:
        raise ValueError("transition must be a non-empty square matrix")
    if not np.all(np.isfinite(p)):
        raise ValueError("transition must contain only finite values")
    if np.any(p < -1e-12):
        raise ValueError("transition probabilities must be non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transition rows must sum to one")
    return p


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy normalized to [0, 1]."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must be T x K with K >= 2")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    totals = p.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each probability row must have positive mass")
    p = p / totals
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])
    return -terms.sum(axis=1) / np.log(p.shape[1])


def hard_state_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of dates for which already-aligned hard states agree."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) == 0:
        raise ValueError("state arrays must be non-empty 1D arrays with equal shape")
    return float(np.mean(a == b))


def expected_durations(transition: np.ndarray) -> np.ndarray:
    """Geometric-duration expectation 1 / (1 - p_ii)."""
    p = _validate_transition(transition)
    diagonal = np.diag(p)
    out = np.empty_like(diagonal)
    absorbing = np.isclose(diagonal, 1.0, atol=1e-12)
    out[absorbing] = np.inf
    out[~absorbing] = 1.0 / (1.0 - diagonal[~absorbing])
    return out


def model_disagreement(probability_stack: np.ndarray) -> np.ndarray:
    """Cross-model SD for a common aligned state probability.

    Parameters
    ----------
    probability_stack:
        M x T array, where M is the number of model specifications.
    """
    x = np.asarray(probability_stack, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("probability_stack must be a non-empty models x time array")
    if not np.all(np.isfinite(x)):
        raise ValueError("probability_stack must be finite")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if x.shape[0] == 1:
        return np.zeros(x.shape[1])
    return np.std(x, axis=0, ddof=1)
