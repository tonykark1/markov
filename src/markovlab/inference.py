"""Out-of-sample state inference for an already fitted HMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hmm import (
    HMMFit,
    _forward_backward,
    _log_gaussian_emission,
    _log_student_t_emission,
)


@dataclass(frozen=True)
class HMMInference:
    """State probabilities for data evaluated under fixed HMM parameters."""

    loglik: float
    predicted: np.ndarray
    filtered: np.ndarray
    smoothed: np.ndarray


def _validate_probability_vector(
    probabilities: np.ndarray,
    n_states: int,
    *,
    name: str,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (n_states,):
        raise ValueError(f"{name} must have shape ({n_states},)")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    total = float(p.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return p / total


def infer_hmm(
    fit: HMMFit,
    x: np.ndarray,
    *,
    initial_probabilities: np.ndarray | None = None,
    continuation: bool = False,
) -> HMMInference:
    """Infer states for observations using fixed fitted parameters.

    Parameters
    ----------
    fit:
        Previously fitted HMM.
    x:
        New observations with the same feature order and scaling used to fit the model.
    initial_probabilities:
        Prior state probabilities before the first row of ``x``. If omitted, the fitted
        model's original initial probabilities are used unless ``continuation=True``.
    continuation:
        If true, initialize the first new observation with the one-step prior from the
        final in-sample filtered state. This is the appropriate mode when ``x`` directly
        follows the sample used to estimate ``fit``.
    """
    observations = np.asarray(x, dtype=float)
    if observations.ndim != 2 or len(observations) < 1:
        raise ValueError("x must be a non-empty 2D array")
    if observations.shape[1] != fit.means.shape[1]:
        raise ValueError("x feature count must match the fitted HMM")
    if not np.all(np.isfinite(observations)):
        raise ValueError("x must contain only finite values")
    if continuation and initial_probabilities is not None:
        raise ValueError("do not supply initial_probabilities when continuation=True")

    n_states = fit.transition.shape[0]
    if continuation:
        initial = fit.filtered[-1] @ fit.transition
    elif initial_probabilities is None:
        initial = fit.initial.copy()
    else:
        initial = _validate_probability_vector(
            initial_probabilities,
            n_states,
            name="initial_probabilities",
        )

    initial = _validate_probability_vector(initial, n_states, name="initial probabilities")

    if fit.family == "gaussian":
        log_emission = _log_gaussian_emission(
            observations,
            fit.means,
            fit.scale_matrices,
        )
    else:
        assert fit.nu is not None
        log_emission, _ = _log_student_t_emission(
            observations,
            fit.means,
            fit.scale_matrices,
            fit.nu,
        )

    loglik, predicted, filtered, smoothed, _ = _forward_backward(
        log_emission,
        initial,
        fit.transition,
    )
    return HMMInference(
        loglik=loglik,
        predicted=predicted,
        filtered=filtered,
        smoothed=smoothed,
    )


def forecast_state_probabilities(
    fit: HMMFit,
    horizon: int = 1,
    *,
    current_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """Forecast latent-state occupancy probabilities ``horizon`` steps ahead."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")

    n_states = fit.transition.shape[0]
    current = fit.filtered[-1] if current_probabilities is None else current_probabilities
    current = _validate_probability_vector(current, n_states, name="current_probabilities")
    if horizon == 0:
        return current
    return current @ np.linalg.matrix_power(fit.transition, horizon)
