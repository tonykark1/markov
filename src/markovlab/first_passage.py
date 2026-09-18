"""First-passage and hitting-time utilities for finite Markov chains."""

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


def _validate_state(state: int, n_states: int, name: str) -> int:
    if not isinstance(state, (int, np.integer)) or not 0 <= state < n_states:
        raise ValueError(f"{name} must be an integer in [0, {n_states - 1}]")
    return int(state)


def _validate_horizon(horizon: int) -> int:
    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    return int(horizon)


def _reachable_transient_states(
    p: np.ndarray,
    start_state: int,
    target_state: int,
) -> list[int]:
    """States reachable from start without passing through the target."""
    seen = {start_state}
    stack = [start_state]
    while stack:
        current = stack.pop()
        for nxt in np.flatnonzero(p[current] > 0):
            nxt = int(nxt)
            if nxt == target_state or nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return sorted(seen)


def first_passage_probability(
    transition: np.ndarray,
    start_state: int,
    target_state: int,
    horizon: int,
) -> float:
    """Probability of reaching ``target_state`` within ``horizon`` steps."""
    p = _validate_transition(transition)
    start_state = _validate_state(start_state, len(p), "start_state")
    target_state = _validate_state(target_state, len(p), "target_state")
    horizon = _validate_horizon(horizon)

    if start_state == target_state:
        return 1.0
    if horizon == 0:
        return 0.0

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
    """Expected steps to first reach ``target_state`` from ``start_state``.

    Returns ``np.inf`` when the target is not reached almost surely from the
    supplied starting state.
    """
    p = _validate_transition(transition)
    start_state = _validate_state(start_state, len(p), "start_state")
    target_state = _validate_state(target_state, len(p), "target_state")
    if start_state == target_state:
        return 0.0

    transient = _reachable_transient_states(p, start_state, target_state)
    q = p[np.ix_(transient, transient)]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(q)))) if len(q) else 0.0
    if spectral_radius >= 1.0 - 1e-12:
        return float(np.inf)

    times = np.linalg.solve(np.eye(len(q)) - q, np.ones(len(q)))
    return float(times[transient.index(start_state)])


def mixture_first_passage_probability(
    transition: np.ndarray,
    current_probabilities: np.ndarray,
    target_state: int,
    horizon: int,
) -> float:
    """First-passage probability when the current state itself is uncertain."""
    p = _validate_transition(transition)
    target_state = _validate_state(target_state, len(p), "target_state")
    horizon = _validate_horizon(horizon)

    alpha = np.asarray(current_probabilities, dtype=float)
    if alpha.shape != (len(p),):
        raise ValueError("current_probabilities must have one entry per state")
    if not np.all(np.isfinite(alpha)) or np.any(alpha < 0):
        raise ValueError("current_probabilities must be finite and non-negative")
    total = alpha.sum()
    if total <= 0:
        raise ValueError("current_probabilities must have positive mass")
    alpha = alpha / total

    if horizon == 0:
        return float(alpha[target_state])

    transient = [i for i in range(len(p)) if i != target_state]
    q = p[np.ix_(transient, transient)]
    survival = float(
        alpha[transient] @ np.linalg.matrix_power(q, horizon) @ np.ones(len(transient))
    )
    return float(np.clip(1.0 - survival, 0.0, 1.0))
