import numpy as np
import pytest

from markovlab.hmm import (
    _forward_backward,
    _log_gaussian_emission,
    fit_hmm,
)


def synthetic_two_regime(seed=4):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(-1.0, 0.4, (80, 2)), rng.normal(1.0, 0.4, (80, 2))]


def test_hmm_returns_predicted_filtered_and_smoothed_probabilities():
    fit = fit_hmm(
        synthetic_two_regime(),
        n_states=2,
        family="student_t",
        n_init=3,
        random_state=4,
    )
    assert fit.predicted.shape == (160, 2)
    assert fit.filtered.shape == (160, 2)
    assert fit.smoothed.shape == (160, 2)
    assert np.allclose(fit.predicted.sum(axis=1), 1.0)
    assert np.allclose(fit.filtered.sum(axis=1), 1.0)
    assert np.allclose(fit.smoothed.sum(axis=1), 1.0)


def test_filtered_is_not_smoothed_and_uses_only_current_information():
    log_emission = np.log(np.array([[0.9, 0.1], [0.1, 0.9], [0.95, 0.05]]))
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    _, _, filtered, smoothed, _ = _forward_backward(log_emission, initial, transition)
    assert not np.allclose(filtered[1], smoothed[1])
    manual = initial * np.exp(log_emission[0])
    manual /= manual.sum()
    assert np.allclose(filtered[0], manual)


def test_predicted_probability_matches_previous_filter_transition():
    log_emission = np.log(np.array([[0.8, 0.2], [0.3, 0.7]]))
    initial = np.array([0.6, 0.4])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    _, predicted, filtered, _, _ = _forward_backward(log_emission, initial, transition)
    assert np.allclose(predicted[0], initial)
    assert np.allclose(predicted[1], filtered[0] @ transition)


def test_final_loglik_is_consistent_with_returned_parameters():
    x = synthetic_two_regime()
    fit = fit_hmm(x, n_states=2, family="gaussian", n_init=2, random_state=11)
    log_emission = _log_gaussian_emission(x, fit.means, fit.scale_matrices)
    loglik, *_ = _forward_backward(log_emission, fit.initial, fit.transition)
    assert np.isclose(loglik, fit.loglik)


def test_student_t_scale_and_covariance_are_distinct():
    fit = fit_hmm(
        synthetic_two_regime(),
        n_states=2,
        family="student_t",
        nu=5,
        n_init=2,
        random_state=3,
    )
    assert np.allclose(fit.covariances, fit.scale_matrices * (5 / 3))


def test_gaussian_covariance_equals_scale():
    fit = fit_hmm(
        synthetic_two_regime(),
        n_states=2,
        family="gaussian",
        n_init=2,
        random_state=3,
    )
    assert np.allclose(fit.covariances, fit.scale_matrices)


def test_reproducible_with_same_seed():
    x = synthetic_two_regime()
    a = fit_hmm(x, n_states=2, family="gaussian", n_init=3, random_state=17)
    b = fit_hmm(x, n_states=2, family="gaussian", n_init=3, random_state=17)
    assert np.isclose(a.loglik, b.loglik)
    assert np.allclose(a.transition, b.transition)
    assert np.allclose(a.means, b.means)


def test_one_state_model_is_supported():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 1))
    fit = fit_hmm(x, n_states=1, family="gaussian", n_init=2, random_state=1)
    assert fit.transition.shape == (1, 1)
    assert np.allclose(fit.transition, [[1.0]])
    assert np.allclose(fit.filtered, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_init": 0},
        {"max_iter": 0},
        {"tol": 0},
        {"sticky_kappa": -1},
        {"covariance_shrinkage": -0.1},
        {"covariance_shrinkage": 1.1},
        {"min_covar": 0},
    ],
)
def test_invalid_optimizer_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        fit_hmm(synthetic_two_regime(), **kwargs)


def test_nonfinite_data_raise():
    x = synthetic_two_regime()
    x[3, 0] = np.nan
    with pytest.raises(ValueError):
        fit_hmm(x)


def test_invalid_student_t_nu_raises():
    with pytest.raises(ValueError):
        fit_hmm(synthetic_two_regime(), family="student_t", nu=2)


def test_fit_metadata_is_populated():
    fit = fit_hmm(
        synthetic_two_regime(),
        n_states=2,
        family="gaussian",
        n_init=2,
        random_state=8,
    )
    assert fit.n_iter >= 1
    assert len(fit.loglik_history) == fit.n_iter
    assert isinstance(fit.converged, bool)
    assert np.isfinite(fit.loglik_history).all()


def test_covariances_are_positive_definite():
    fit = fit_hmm(
        synthetic_two_regime(),
        n_states=2,
        family="student_t",
        n_init=2,
        random_state=9,
    )
    for matrix in fit.scale_matrices:
        assert np.all(np.linalg.eigvalsh(matrix) > 0)
