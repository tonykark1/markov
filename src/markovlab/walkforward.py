"""Leakage-resistant expanding-window HMM evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .alignment import (
    align_by_distribution_distance,
    align_by_mean_distance,
    reorder_states,
)
from .hmm import Family, fit_hmm
from .inference import infer_hmm

WalkForwardAlignment = Literal["mean", "symmetric_kl", "bhattacharyya", "wasserstein"]


@dataclass(frozen=True)
class WalkForwardResult:
    """One-step-ahead state probabilities from an expanding-window experiment."""

    oos_index: np.ndarray
    predicted: np.ndarray
    filtered: np.ndarray
    converged: np.ndarray
    train_loglik: np.ndarray
    aligned_state_means: np.ndarray
    aligned_state_covariances: np.ndarray
    alignment_metric: WalkForwardAlignment


def expanding_walk_forward(
    x: np.ndarray,
    *,
    initial_train: int,
    n_states: int = 2,
    family: Family = "student_t",
    nu: float = 5.0,
    n_init: int = 4,
    max_iter: int = 120,
    tol: float = 1e-6,
    sticky_kappa: float = 0.0,
    covariance_shrinkage: float = 0.02,
    min_covar: float = 1e-6,
    random_state: int | None = None,
    align_states: bool = True,
    alignment_metric: WalkForwardAlignment = "symmetric_kl",
) -> WalkForwardResult:
    """Run an expanding-window, one-step-ahead HMM experiment.

    Every OOS observation is standardized using training-window statistics only.
    The HMM is re-estimated from observations strictly before the OOS row, then that
    row is filtered under fixed parameters.

    When ``align_states`` is true, each refit is Hungarian-aligned to the preceding
    refit in the original data scale. The default ``symmetric_kl`` alignment uses
    both state means and covariance matrices. ``bhattacharyya`` and ``wasserstein``
    are also available. ``mean`` reproduces the earlier mean-only behavior.

    For Student-t states, covariance-sensitive alignment uses ``HMMFit.covariances``
    and therefore compares a moment-matched Gaussian representation of each state.
    """
    observations = np.asarray(x, dtype=float)
    if observations.ndim != 2 or len(observations) < 4:
        raise ValueError("x must be a 2D array with at least four observations")
    if not np.all(np.isfinite(observations)):
        raise ValueError("x must contain only finite values")
    if not isinstance(initial_train, (int, np.integer)):
        raise ValueError("initial_train must be an integer")
    if initial_train < 3 or initial_train >= len(observations):
        raise ValueError("initial_train must lie in [3, n_observations - 1]")
    if alignment_metric not in {"mean", "symmetric_kl", "bhattacharyya", "wasserstein"}:
        raise ValueError(
            "alignment_metric must be 'mean', 'symmetric_kl', 'bhattacharyya', or 'wasserstein'"
        )

    predicted_rows: list[np.ndarray] = []
    filtered_rows: list[np.ndarray] = []
    convergence: list[bool] = []
    logliks: list[float] = []
    mean_rows: list[np.ndarray] = []
    covariance_rows: list[np.ndarray] = []
    reference_means: np.ndarray | None = None
    reference_covariances: np.ndarray | None = None

    for t in range(initial_train, len(observations)):
        train = observations[:t]
        center = train.mean(axis=0)
        scale = train.std(axis=0, ddof=1)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("every feature must have positive training-window variance")

        z_train = (train - center) / scale
        seed = None if random_state is None else int(random_state + t)
        fit = fit_hmm(
            z_train,
            n_states=n_states,
            family=family,
            nu=nu,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            sticky_kappa=sticky_kappa,
            covariance_shrinkage=covariance_shrinkage,
            min_covar=min_covar,
            random_state=seed,
        )

        z_current = ((observations[t] - center) / scale)[None, :]
        inference = infer_hmm(fit, z_current, continuation=True)

        raw_means = center + fit.means * scale
        scale_matrix = np.diag(scale)
        raw_covariances = np.asarray(
            [scale_matrix @ covariance @ scale_matrix for covariance in fit.covariances]
        )

        if align_states and reference_means is not None:
            if alignment_metric == "mean":
                order = align_by_mean_distance(reference_means, raw_means)
            else:
                assert reference_covariances is not None
                order = align_by_distribution_distance(
                    reference_means,
                    reference_covariances,
                    raw_means,
                    raw_covariances,
                    metric=alignment_metric,
                )
        else:
            order = np.arange(n_states)

        aligned_means = raw_means[order]
        aligned_covariances = raw_covariances[order]
        predicted_rows.append(reorder_states(inference.predicted[0], order, axis=0))
        filtered_rows.append(reorder_states(inference.filtered[0], order, axis=0))
        convergence.append(fit.converged)
        logliks.append(fit.loglik)
        mean_rows.append(aligned_means)
        covariance_rows.append(aligned_covariances)
        reference_means = aligned_means
        reference_covariances = aligned_covariances

    return WalkForwardResult(
        oos_index=np.arange(initial_train, len(observations), dtype=int),
        predicted=np.vstack(predicted_rows),
        filtered=np.vstack(filtered_rows),
        converged=np.asarray(convergence, dtype=bool),
        train_loglik=np.asarray(logliks, dtype=float),
        aligned_state_means=np.stack(mean_rows),
        aligned_state_covariances=np.stack(covariance_rows),
        alignment_metric=alignment_metric,
    )
