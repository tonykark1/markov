"""State-alignment utilities for comparing HMM specifications.

Mean-only alignment is retained for backward compatibility. The distribution-aware
alignment functions compare both state means and covariance structure. For Gaussian
emissions the distances are exact distribution distances. For Student-t emissions,
pass ``HMMFit.covariances`` to obtain a moment-matched Gaussian comparison.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment

AlignmentMetric = Literal["symmetric_kl", "bhattacharyya", "wasserstein"]


def _validate_means(reference_means: np.ndarray, candidate_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_means, dtype=float)
    candidate = np.asarray(candidate_means, dtype=float)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("mean arrays must be two-dimensional")
    if reference.shape != candidate.shape:
        raise ValueError("reference_means and candidate_means must have equal shape")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("mean arrays must be finite")
    return reference, candidate


def _validate_covariances(covariances: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    cov = np.asarray(covariances, dtype=float)
    n_states, dimension = shape
    if cov.shape != (n_states, dimension, dimension):
        raise ValueError(f"{name} must have shape (n_states, dimension, dimension)")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(cov, np.swapaxes(cov, 1, 2), rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals <= 0):
        raise ValueError(f"{name} must be positive definite")
    return cov


def _logdet_spd(covariance: np.ndarray) -> float:
    chol, _ = cho_factor(covariance, lower=True, check_finite=True)
    return float(2.0 * np.log(np.diag(chol)).sum())


def _gaussian_kl(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    """KL[N(a)||N(b)]."""
    dimension = len(mean_a)
    chol_b = cho_factor(covariance_b, lower=True, check_finite=True)
    trace_term = float(np.trace(cho_solve(chol_b, covariance_a, check_finite=True)))
    delta = mean_b - mean_a
    quadratic = float(delta @ cho_solve(chol_b, delta, check_finite=True))
    return 0.5 * (
        trace_term
        + quadratic
        - dimension
        + _logdet_spd(covariance_b)
        - _logdet_spd(covariance_a)
    )


def _symmetric_kl(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    return 0.5 * (
        _gaussian_kl(mean_a, covariance_a, mean_b, covariance_b)
        + _gaussian_kl(mean_b, covariance_b, mean_a, covariance_a)
    )


def _bhattacharyya(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    pooled = 0.5 * (covariance_a + covariance_b)
    chol_pooled = cho_factor(pooled, lower=True, check_finite=True)
    delta = mean_b - mean_a
    quadratic = float(delta @ cho_solve(chol_pooled, delta, check_finite=True))
    determinant_term = 0.5 * (
        _logdet_spd(pooled)
        - 0.5 * (_logdet_spd(covariance_a) + _logdet_spd(covariance_b))
    )
    return 0.125 * quadratic + determinant_term


def _spd_sqrt(covariance: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(covariance)
    if np.any(eigvals <= 0):
        raise ValueError("covariance must be positive definite")
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _wasserstein(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    """Gaussian 2-Wasserstein distance."""
    root_b = _spd_sqrt(covariance_b)
    middle_root = _spd_sqrt(root_b @ covariance_a @ root_b)
    squared = float(
        np.sum((mean_a - mean_b) ** 2)
        + np.trace(covariance_a + covariance_b - 2.0 * middle_root)
    )
    return float(np.sqrt(max(squared, 0.0)))


def distribution_distance_matrix(
    reference_means: np.ndarray,
    reference_covariances: np.ndarray,
    candidate_means: np.ndarray,
    candidate_covariances: np.ndarray,
    *,
    metric: AlignmentMetric = "symmetric_kl",
) -> np.ndarray:
    """Return pairwise state distances using means and covariance structure.

    ``symmetric_kl`` and ``bhattacharyya`` are exact for Gaussian emissions.
    ``wasserstein`` is the Gaussian 2-Wasserstein distance. For Student-t states,
    passing finite covariance matrices yields a moment-matched Gaussian proxy.
    """
    reference, candidate = _validate_means(reference_means, candidate_means)
    reference_cov = _validate_covariances(reference_covariances, reference.shape, "reference_covariances")
    candidate_cov = _validate_covariances(candidate_covariances, candidate.shape, "candidate_covariances")

    functions = {
        "symmetric_kl": _symmetric_kl,
        "bhattacharyya": _bhattacharyya,
        "wasserstein": _wasserstein,
    }
    if metric not in functions:
        raise ValueError("metric must be 'symmetric_kl', 'bhattacharyya', or 'wasserstein'")
    distance = functions[metric]

    n_states = len(reference)
    cost = np.empty((n_states, n_states), dtype=float)
    for i in range(n_states):
        for j in range(n_states):
            cost[i, j] = distance(
                reference[i],
                reference_cov[i],
                candidate[j],
                candidate_cov[j],
            )
    return cost


def align_by_mean_distance(
    reference_means: np.ndarray,
    candidate_means: np.ndarray,
) -> np.ndarray:
    """Return candidate-state indices ordered to match reference states by mean distance."""
    reference, candidate = _validate_means(reference_means, candidate_means)
    cost = np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.empty(len(row_ind), dtype=int)
    order[row_ind] = col_ind
    return order


def align_by_distribution_distance(
    reference_means: np.ndarray,
    reference_covariances: np.ndarray,
    candidate_means: np.ndarray,
    candidate_covariances: np.ndarray,
    *,
    metric: AlignmentMetric = "symmetric_kl",
) -> np.ndarray:
    """Hungarian-align states using a full mean/covariance distribution distance."""
    cost = distribution_distance_matrix(
        reference_means,
        reference_covariances,
        candidate_means,
        candidate_covariances,
        metric=metric,
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.empty(len(row_ind), dtype=int)
    order[row_ind] = col_ind
    return order


def reorder_states(array: np.ndarray, order: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reorder a state axis using an alignment returned by an alignment function."""
    x = np.asarray(array)
    order = np.asarray(order, dtype=int)
    if order.ndim != 1 or sorted(order.tolist()) != list(range(len(order))):
        raise ValueError("order must be a permutation of 0..K-1")
    if x.shape[axis] != len(order):
        raise ValueError("state axis length does not match order")
    return np.take(x, order, axis=axis)
