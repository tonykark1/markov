"""Numerically robust Gaussian and fixed-nu Student-t HMM estimation.

The API explicitly separates predicted, filtered, and smoothed state probabilities.
The Student-t parameterization stores *scale matrices*; ``covariances`` exposes the
actual covariance matrices when they exist (nu > 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln, logsumexp

Family = Literal["gaussian", "student_t"]


@dataclass(frozen=True)
class HMMFit:
    loglik: float
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    scale_matrices: np.ndarray
    predicted: np.ndarray
    filtered: np.ndarray
    smoothed: np.ndarray
    family: Family
    nu: float | None
    converged: bool
    n_iter: int
    loglik_history: tuple[float, ...]

    @property
    def covariances(self) -> np.ndarray:
        """Return state covariance matrices.

        For Gaussian emissions the scale matrix is the covariance matrix. For a
        multivariate Student-t with nu > 2, Cov(X) = nu / (nu - 2) * scale.
        """
        if self.family == "student_t":
            assert self.nu is not None
            return self.scale_matrices * (self.nu / (self.nu - 2.0))
        return self.scale_matrices.copy()


def _validate_fit_inputs(
    x: np.ndarray,
    n_states: int,
    family: Family,
    nu: float,
    n_init: int,
    max_iter: int,
    tol: float,
    sticky_kappa: float,
    covariance_shrinkage: float,
    min_covar: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("x must be a 2D array with at least three observations")
    if x.shape[1] < 1:
        raise ValueError("x must contain at least one feature")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if not isinstance(n_states, (int, np.integer)) or n_states < 1 or n_states > len(x):
        raise ValueError("n_states must be an integer in [1, n_observations]")
    if family not in {"gaussian", "student_t"}:
        raise ValueError("family must be 'gaussian' or 'student_t'")
    if family == "student_t" and (not np.isfinite(nu) or nu <= 2):
        raise ValueError("nu must be finite and > 2 for finite covariance")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not np.isfinite(sticky_kappa) or sticky_kappa < 0:
        raise ValueError("sticky_kappa must be finite and non-negative")
    if not np.isfinite(covariance_shrinkage) or not 0 <= covariance_shrinkage <= 1:
        raise ValueError("covariance_shrinkage must lie in [0, 1]")
    if not np.isfinite(min_covar) or min_covar <= 0:
        raise ValueError("min_covar must be finite and positive")
    return x


def _regularize_covariance(
    covariance: np.ndarray,
    shrinkage: float,
    min_covar: float,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    if not np.all(np.isfinite(covariance)):
        raise np.linalg.LinAlgError("non-finite covariance estimate")

    diagonal_target = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal_target

    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, min_covar)
    covariance = (eigvecs * eigvals) @ eigvecs.T
    return (covariance + covariance.T) / 2.0


def _log_gaussian_emission(
    x: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    n_obs, dim = x.shape
    n_states = len(means)
    out = np.empty((n_obs, n_states))

    for k in range(n_states):
        chol, lower = cho_factor(scales[k], lower=True, check_finite=True)
        z = x - means[k]
        solved = cho_solve((chol, lower), z.T, check_finite=True).T
        quad = np.einsum("ij,ij->i", z, solved)
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        out[:, k] = -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)
    return out


def _log_student_t_emission(
    x: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_obs, dim = x.shape
    n_states = len(means)
    out = np.empty((n_obs, n_states))
    delta = np.empty((n_obs, n_states))
    const = (
        gammaln((nu + dim) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * dim * np.log(nu * np.pi)
    )

    for k in range(n_states):
        chol, lower = cho_factor(scales[k], lower=True, check_finite=True)
        z = x - means[k]
        solved = cho_solve((chol, lower), z.T, check_finite=True).T
        q = np.einsum("ij,ij->i", z, solved)
        delta[:, k] = q
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        out[:, k] = const - 0.5 * logdet - 0.5 * (nu + dim) * np.log1p(q / nu)
    return out, delta


def _forward_backward(
    log_emission: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run log-domain forward-backward recursion.

    Returns
    -------
    loglik, predicted, filtered, smoothed, xi
    """
    t_count, n_states = log_emission.shape
    log_transition = np.log(np.clip(transition, 1e-300, 1.0))
    log_initial = np.log(np.clip(initial, 1e-300, 1.0))

    log_alpha = np.empty((t_count, n_states))
    predicted = np.empty((t_count, n_states))

    predicted[0] = initial
    log_alpha[0] = log_initial + log_emission[0]
    for t in range(1, t_count):
        log_pred = logsumexp(log_alpha[t - 1][:, None] + log_transition, axis=0)
        predicted[t] = np.exp(log_pred - logsumexp(log_pred))
        log_alpha[t] = log_emission[t] + log_pred

    loglik = float(logsumexp(log_alpha[-1]))
    filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    log_beta = np.zeros((t_count, n_states))
    for t in range(t_count - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_transition + log_emission[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1,
        )

    log_gamma = log_alpha + log_beta
    smoothed = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))

    xi = np.empty((t_count - 1, n_states, n_states))
    for t in range(t_count - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_transition
            + log_emission[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        xi[t] = np.exp(log_xi - logsumexp(log_xi))

    return loglik, predicted, filtered, smoothed, xi


def _kmeans_initialize(
    x: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    n_iter: int = 30,
) -> np.ndarray:
    """Small dependency-free k-means used only for initialization."""
    chosen = rng.choice(len(x), size=n_states, replace=False)
    centers = x[chosen].copy()

    for _ in range(n_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for k in range(n_states):
            members = x[labels == k]
            if len(members):
                new_centers[k] = members.mean(axis=0)
            else:
                new_centers[k] = x[rng.integers(len(x))]
        if np.allclose(new_centers, centers, rtol=0.0, atol=1e-8):
            break
        centers = new_centers
    return labels


def _initialize(
    x: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    covariance_shrinkage: float,
    min_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_obs, dim = x.shape
    if n_states == 1:
        labels = np.zeros(n_obs, dtype=int)
    else:
        labels = _kmeans_initialize(x, n_states, rng)

    means = np.empty((n_states, dim))
    global_cov = np.atleast_2d(np.cov(x, rowvar=False))
    global_cov = _regularize_covariance(global_cov, covariance_shrinkage, min_covar)
    scales = np.empty((n_states, dim, dim))

    for k in range(n_states):
        members = x[labels == k]
        if len(members) == 0:
            means[k] = x[rng.integers(n_obs)]
            scales[k] = global_cov
            continue
        means[k] = members.mean(axis=0)
        if len(members) <= 1:
            state_cov = global_cov
        else:
            state_cov = np.atleast_2d(np.cov(members, rowvar=False))
        scales[k] = _regularize_covariance(
            state_cov, covariance_shrinkage, min_covar
        )

    if n_states == 1:
        transition = np.ones((1, 1))
    else:
        transition = np.full((n_states, n_states), 0.1 / (n_states - 1))
        np.fill_diagonal(transition, 0.9)
    initial = np.full(n_states, 1.0 / n_states)
    return initial, transition, means, scales


def fit_hmm(
    x: np.ndarray,
    n_states: int = 2,
    family: Family = "student_t",
    nu: float = 5.0,
    n_init: int = 8,
    max_iter: int = 200,
    tol: float = 1e-6,
    sticky_kappa: float = 0.0,
    covariance_shrinkage: float = 0.02,
    min_covar: float = 1e-6,
    random_state: int | None = None,
) -> HMMFit:
    """Fit a full-scale-matrix HMM by EM.

    Notes
    -----
    ``filtered`` contains P(S_t | Y_1:t) and is suitable for real-time state
    inference conditional on fitted parameters. ``smoothed`` contains
    P(S_t | Y_1:T) and therefore uses future observations. ``predicted`` contains
    the one-step prior probabilities before observing Y_t.
    """
    x = _validate_fit_inputs(
        x,
        n_states,
        family,
        nu,
        n_init,
        max_iter,
        tol,
        sticky_kappa,
        covariance_shrinkage,
        min_covar,
    )
    rng = np.random.default_rng(random_state)
    best: HMMFit | None = None

    for _ in range(n_init):
        initial, transition, means, scales = _initialize(
            x,
            n_states,
            rng,
            covariance_shrinkage,
            min_covar,
        )
        history: list[float] = []
        converged = False
        predicted = filtered = smoothed = xi = None

        for iteration in range(1, max_iter + 1):
            if family == "gaussian":
                log_emission = _log_gaussian_emission(x, means, scales)
                delta = None
            else:
                log_emission, delta = _log_student_t_emission(x, means, scales, nu)

            loglik, predicted, filtered, smoothed, xi = _forward_backward(
                log_emission,
                initial,
                transition,
            )
            history.append(loglik)

            if len(history) >= 2:
                change = abs(history[-1] - history[-2])
                if change <= tol * (1.0 + abs(history[-2])):
                    converged = True
                    break

            # On the final allowed iteration, return the coherent E-step above
            # rather than performing an M-step that is not followed by re-evaluation.
            if iteration == max_iter:
                break

            initial = np.maximum(smoothed[0], 1e-15)
            initial /= initial.sum()

            counts = xi.sum(axis=0) + np.eye(n_states) * sticky_kappa
            counts = np.maximum(counts, 1e-15)
            transition = counts / counts.sum(axis=1, keepdims=True)

            dim = x.shape[1]
            for k in range(n_states):
                if family == "gaussian":
                    weights = smoothed[:, k]
                    denom = weights.sum()
                    means[k] = (weights[:, None] * x).sum(axis=0) / denom
                    z = x - means[k]
                    state_scale = (z * weights[:, None]).T @ z / denom
                else:
                    assert delta is not None
                    tau = (nu + dim) / (nu + delta[:, k])
                    mean_weights = smoothed[:, k] * tau
                    means[k] = (mean_weights[:, None] * x).sum(axis=0) / mean_weights.sum()
                    z = x - means[k]
                    scale_weights = smoothed[:, k] * tau
                    state_scale = (z * scale_weights[:, None]).T @ z / smoothed[:, k].sum()

                scales[k] = _regularize_covariance(
                    state_scale,
                    covariance_shrinkage,
                    min_covar,
                )

        assert predicted is not None
        assert filtered is not None
        assert smoothed is not None

        current = HMMFit(
            loglik=float(history[-1]),
            initial=initial.copy(),
            transition=transition.copy(),
            means=means.copy(),
            scale_matrices=scales.copy(),
            predicted=predicted.copy(),
            filtered=filtered.copy(),
            smoothed=smoothed.copy(),
            family=family,
            nu=nu if family == "student_t" else None,
            converged=converged,
            n_iter=len(history),
            loglik_history=tuple(float(value) for value in history),
        )
        if best is None or current.loglik > best.loglik:
            best = current

    assert best is not None
    return best
