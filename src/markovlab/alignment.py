"""State-alignment utilities for comparing HMM specifications."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_by_mean_distance(
    reference_means: np.ndarray,
    candidate_means: np.ndarray,
) -> np.ndarray:
    """Return candidate-state indices ordered to match reference states.

    State labels are arbitrary. This function uses a minimum-cost Hungarian
    assignment on Euclidean distances between state mean vectors.
    """
    reference = np.asarray(reference_means, dtype=float)
    candidate = np.asarray(candidate_means, dtype=float)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("mean arrays must be two-dimensional")
    if reference.shape != candidate.shape:
        raise ValueError("reference_means and candidate_means must have equal shape")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("mean arrays must be finite")

    cost = np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.empty(len(row_ind), dtype=int)
    order[row_ind] = col_ind
    return order


def reorder_states(array: np.ndarray, order: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reorder a state axis using an alignment returned by ``align_by_mean_distance``."""
    x = np.asarray(array)
    order = np.asarray(order, dtype=int)
    if order.ndim != 1 or sorted(order.tolist()) != list(range(len(order))):
        raise ValueError("order must be a permutation of 0..K-1")
    if x.shape[axis] != len(order):
        raise ValueError("state axis length does not match order")
    return np.take(x, order, axis=axis)
