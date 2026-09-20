"""Utilities for transparent combinations of independently fitted latent-state chains."""

from __future__ import annotations

from itertools import product

import numpy as np


def _validate_probability_matrix(probabilities: np.ndarray, *, name: str) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"{name} must have shape T x K with K >= 2")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"{name} rows must have positive total mass")
    return p / row_sums


def _validate_transition(transition: np.ndarray, *, name: str) -> np.ndarray:
    p = np.asarray(transition, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or len(p) < 2:
        raise ValueError(f"{name} must be a square K x K matrix with K >= 2")
    if not np.all(np.isfinite(p)) or np.any(p < -1e-12):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"{name} rows must sum to one")
    return np.clip(p, 0.0, 1.0)


def joint_probabilities(*chains: np.ndarray) -> np.ndarray:
    """Combine independent chain probabilities into joint-state probabilities.

    The function makes the independence assumption explicit. If chain A has K states
    and chain B has J states, the output has K*J columns ordered lexicographically by
    component-state index, matching ``numpy.kron`` ordering.
    """
    if len(chains) < 2:
        raise ValueError("at least two latent chains are required")

    normalized = [
        _validate_probability_matrix(chain, name=f"chain[{i}]") for i, chain in enumerate(chains)
    ]
    n_obs = normalized[0].shape[0]
    if any(chain.shape[0] != n_obs for chain in normalized[1:]):
        raise ValueError("all chains must contain the same number of observations")

    joint = normalized[0]
    for chain in normalized[1:]:
        joint = np.einsum("ti,tj->tij", joint, chain).reshape(n_obs, -1)
    return joint / joint.sum(axis=1, keepdims=True)


def joint_transition(*transitions: np.ndarray) -> np.ndarray:
    """Kronecker-product transition matrix for independent Markov chains."""
    if len(transitions) < 2:
        raise ValueError("at least two transition matrices are required")

    matrices = [
        _validate_transition(transition, name=f"transition[{i}]")
        for i, transition in enumerate(transitions)
    ]
    joint = matrices[0]
    for transition in matrices[1:]:
        joint = np.kron(joint, transition)
    return joint / joint.sum(axis=1, keepdims=True)


def joint_state_labels(*label_sets: list[str] | tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Return joint-state labels in the same ordering used by the Kronecker products."""
    if len(label_sets) < 2:
        raise ValueError("at least two label sets are required")
    if any(len(labels) < 2 for labels in label_sets):
        raise ValueError("each label set must contain at least two states")
    return tuple(product(*label_sets))
