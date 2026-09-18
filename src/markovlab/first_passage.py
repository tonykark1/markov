"""First-passage and hitting-time utilities for finite Markov chains."""

from __future__ import annotations

import numpy as np


def _validate(transition: np.ndarray) -> np.ndarray:
    p = np.asarray(transition, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("transition must be square")
    if np.any(p < -1e-12) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transition rows must be non-negative and sum to one")
    return p


def first_passage_probability(
    transition: np.ndarray,
    start_state: int,
    target_state: int,
    horizon: int,
) -> float:
    """Probability of reaching ``target_state`` within ``horizon`` steps.

    If start_state already equals target_state, the returned probability is 1.
    """
    p = _validate(transition)
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if start_state == target_state:
        return 1.0

    transient = [i for i in range(len(p)) if i != target_state]
    q = p[np.ix_(transient, transient)]
    e = np.zeros(len(transient))
    e[transient.index(start_state)] = 1.0
    survival = float(e @ np.linalg.matrix_power(q, horizon) @ np.ones(len(transient)))
    return float(np.clip(1.0 - survival, 0.0, 1.0))


def expected_hitting_time(
    transition: np.ndarray,
    start_state: int,
    target_state: int,
) -> float:
    """Expected steps to first reach ``target_state`` from ``start_state``."""
    p = _validate(transition)
    if start_state == target_state:
        return 0.0

    transient = [i for i in range(len(p)) if i != target_state]
    q = p[np.ix_(transient, transient)]
    fundamental = np.linalg.inv(np.eye(len(q)) - q)
    times = fundamental @ np.ones(len(q))
    return float(times[transient.index(start_state)])


def mixture_first_passage_probability(
    transition: np.ndarray,
    current_probabilities: np.ndarray,
    target_state: int,
    horizon: int,
) -> float:
    """First-passage probability when the current state itself is uncertain."""
    p = _validate(transition)
    alpha = np.asarray(current_probabilities, dtype=float)
    if alpha.shape != (len(p),):
        raise ValueError("current_probabilities must have one entry per state")
    alpha = alpha / alpha.sum()

    transient = [i for i in range(len(p)) if i != target_state]
    q = p[np.ix_(transient, transient)]
    survival = float(alpha[transient] @ np.linalg.matrix_power(q, horizon) @ np.ones(len(transient)))
    return float(np.clip(1.0 - survival, 0.0, 1.0))
