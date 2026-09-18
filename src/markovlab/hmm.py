"""Transparent Gaussian and fixed-nu Student-t HMM estimation.

The implementation is deliberately compact and dependency-light. It is intended for
research diagnostics, not as a replacement for a production HMM library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaln

Family = Literal["gaussian", "student_t"]


@dataclass(frozen=True)
class HMMFit:
    loglik: float
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    filtered: np.ndarray
    family: Family
    nu: float | None


def _scaled_forward_backward(
    emission: np.ndarray, initial: np.ndarray, transition: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    t_count, n_states = emission.shape
    alpha = np.empty((t_count, n_states))
    scale = np.empty(t_count)

    alpha[0] = initial * emission[0]
    scale[0] = max(alpha[0].sum(), 1e-300)
    alpha[0] /= scale[0]

    for t in range(1, t_count):
        alpha[t] = (alpha[t - 1] @ transition) * emission[t]
        scale[t] = max(alpha[t].sum(), 1e-300)
        alpha[t] /= scale[t]

    beta = np.ones((t_count, n_states))
    for t in range(t_count - 2, -1, -1):
        beta[t] = transition @ (emission[t + 1] * beta[t + 1])
        beta[t] /= scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.empty((t_count - 1, n_states, n_states))
    for t in range(t_count - 1):
        x = alpha[t][:, None] * transition * (emission[t + 1] * beta[t + 1])[None, :]
        xi[t] = x / max(x.sum(), 1e-300)

    return float(np.log(scale).sum()), gamma, xi


def _gaussian_emission(x: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    n_obs, dim = x.shape
    n_states = len(means)
    out = np.empty((n_obs, n_states))

    for k in range(n_states):
        sigma = (covs[k] + covs[k].T) / 2 + np.eye(dim) * 1e-6
        inv = np.linalg.inv(sigma)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix must be positive definite")
        z = x - means[k]
        quad = np.einsum("ij,jk,ik->i", z, inv, z)
        logp = -0.5 * (dim * np.log(2 * np.pi) + logdet + quad)
        out[:, k] = np.exp(np.clip(logp, -700, 100))

    return np.maximum(out, 1e-300)


def _student_t_emission(
    x: np.ndarray, means: np.ndarray, covs: np.ndarray, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    n_obs, dim = x.shape
    n_states = len(means)
    out = np.empty((n_obs, n_states))
    delta = np.empty((n_obs, n_states))
    const = gammaln((nu + dim) / 2) - gammaln(nu / 2) - 0.5 * dim * np.log(nu * np.pi)

    for k in range(n_states):
        sigma = (covs[k] + covs[k].T) / 2 + np.eye(dim) * 1e-6
        inv = np.linalg.inv(sigma)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix must be positive definite")
        z = x - means[k]
        q = np.einsum("ij,jk,ik->i", z, inv, z)
        delta[:, k] = q
        logp = const - 0.5 * logdet - 0.5 * (nu + dim) * np.log1p(q / nu)
        out[:, k] = np.exp(np.clip(logp, -700, 100))

    return np.maximum(out, 1e-300), delta


def _initialize(x: np.ndarray, n_states: int, rng: np.random.Generator):
    """Lightweight randomized initialization without sklearn."""
    n_obs, dim = x.shape
    chosen = rng.choice(n_obs, size=n_states, replace=False)
    means = x[chosen].copy()
    base = np.cov(x, rowvar=False)
    if dim == 1:
        base = np.array([[float(base)]])
    base = np.asarray(base) + np.eye(dim) * 0.05
    covs = np.repeat(base[None, :, :], n_states, axis=0)
    transition = np.full((n_states, n_states), (1 - 0.9) / max(n_states - 1, 1))
    np.fill_diagonal(transition, 0.9 if n_states > 1 else 1.0)
    initial = np.full(n_states, 1 / n_states)
    return initial, transition, means, covs


def fit_hmm(
    x: np.ndarray,
    n_states: int = 2,
    family: Family = "student_t",
    nu: float = 5.0,
    n_init: int = 8,
    max_iter: int = 200,
    tol: float = 1e-6,
    sticky_kappa: float = 0.0,
    random_state: int | None = None,
) -> HMMFit:
    """Fit a full-covariance HMM by EM.

    Parameters
    ----------
    x:
        T x d observation matrix. Standardize upstream when variables have very
        different scales.
    family:
        ``"gaussian"`` or fixed-dof ``"student_t"``.
    sticky_kappa:
        Optional pseudo-count added to self transitions. Use as a sensitivity
        parameter rather than an automatic improvement.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 3:
        raise ValueError("x must be a 2D array with at least three observations")
    if n_states < 1 or n_states > len(x):
        raise ValueError("invalid n_states")
    if family not in {"gaussian", "student_t"}:
        raise ValueError("family must be 'gaussian' or 'student_t'")
    if family == "student_t" and nu <= 2:
        raise ValueError("nu must be > 2 for finite covariance")

    rng = np.random.default_rng(random_state)
    best: HMMFit | None = None

    for _ in range(n_init):
        initial, transition, means, covs = _initialize(x, n_states, rng)
        previous = -np.inf

        for _ in range(max_iter):
            if family == "gaussian":
                emission = _gaussian_emission(x, means, covs)
                delta = None
            else:
                emission, delta = _student_t_emission(x, means, covs, nu)

            loglik, gamma, xi = _scaled_forward_backward(emission, initial, transition)
            initial = np.maximum(gamma[0], 1e-12)
            initial /= initial.sum()

            counts = xi.sum(axis=0) + np.eye(n_states) * sticky_kappa
            counts = np.maximum(counts, 1e-12)
            transition = counts / counts.sum(axis=1, keepdims=True)

            dim = x.shape[1]
            for k in range(n_states):
                if family == "gaussian":
                    mean_weights = gamma[:, k]
                    means[k] = (mean_weights[:, None] * x).sum(axis=0) / mean_weights.sum()
                    z = x - means[k]
                    covs[k] = (z * mean_weights[:, None]).T @ z / mean_weights.sum()
                else:
                    assert delta is not None
                    tau = (nu + dim) / (nu + delta[:, k])
                    mean_weights = gamma[:, k] * tau
                    means[k] = (mean_weights[:, None] * x).sum(axis=0) / mean_weights.sum()
                    z = x - means[k]
                    cov_weights = gamma[:, k] * tau
                    covs[k] = (z * cov_weights[:, None]).T @ z / gamma[:, k].sum()
                covs[k] += np.eye(dim) * 1e-5

            if np.isfinite(previous) and abs(loglik - previous) <= tol * (1 + abs(previous)):
                break
            previous = loglik

        current = HMMFit(
            loglik=float(loglik),
            initial=initial.copy(),
            transition=transition.copy(),
            means=means.copy(),
            covariances=covs.copy(),
            filtered=gamma.copy(),
            family=family,
            nu=nu if family == "student_t" else None,
        )
        if best is None or current.loglik > best.loglik:
            best = current

    assert best is not None
    return best
