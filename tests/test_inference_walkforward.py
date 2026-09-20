import numpy as np
import pytest

from markovlab.hmm import fit_hmm
from markovlab.inference import forecast_state_probabilities, infer_hmm
from markovlab.walkforward import expanding_walk_forward


def _two_regime_sample(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(-1.0, 0.35, size=(18, 1))
    right = rng.normal(1.0, 0.35, size=(18, 1))
    return np.vstack([left, right])


def test_inference_continuation_uses_next_state_prior():
    x = _two_regime_sample()
    fit = fit_hmm(
        x[:30],
        n_states=2,
        family="gaussian",
        n_init=2,
        max_iter=80,
        random_state=11,
    )
    inference = infer_hmm(fit, x[30:32], continuation=True)
    expected_prior = fit.filtered[-1] @ fit.transition
    assert np.allclose(inference.predicted[0], expected_prior)
    assert np.allclose(inference.filtered.sum(axis=1), 1.0)
    assert np.allclose(inference.smoothed.sum(axis=1), 1.0)


def test_inference_rejects_conflicting_initialization_modes():
    x = _two_regime_sample()
    fit = fit_hmm(x[:24], n_states=2, family="gaussian", n_init=1, random_state=2)
    with pytest.raises(ValueError, match="continuation"):
        infer_hmm(
            fit,
            x[24:25],
            continuation=True,
            initial_probabilities=np.array([0.5, 0.5]),
        )


def test_forecast_state_probabilities_matches_matrix_power():
    x = _two_regime_sample()
    fit = fit_hmm(x, n_states=2, family="gaussian", n_init=2, random_state=5)
    current = np.array([0.25, 0.75])
    got = forecast_state_probabilities(fit, 3, current_probabilities=current)
    expected = current @ np.linalg.matrix_power(fit.transition, 3)
    assert np.allclose(got, expected)
    assert np.allclose(forecast_state_probabilities(fit, 0, current_probabilities=current), current)


def test_walkforward_shapes_and_probability_rows():
    x = _two_regime_sample()
    result = expanding_walk_forward(
        x,
        initial_train=18,
        n_states=2,
        family="gaussian",
        n_init=1,
        max_iter=40,
        random_state=13,
    )
    assert result.predicted.shape == (18, 2)
    assert result.filtered.shape == (18, 2)
    assert result.aligned_state_means.shape == (18, 2, 1)
    assert np.allclose(result.predicted.sum(axis=1), 1.0)
    assert np.allclose(result.filtered.sum(axis=1), 1.0)


def test_walkforward_does_not_use_future_observations():
    x = _two_regime_sample(seed=17)
    changed_future = x.copy()
    changed_future[30:] = 1000.0

    kwargs = dict(
        initial_train=18,
        n_states=2,
        family="gaussian",
        n_init=1,
        max_iter=35,
        random_state=19,
    )
    original = expanding_walk_forward(x, **kwargs)
    modified = expanding_walk_forward(changed_future, **kwargs)

    unaffected = original.oos_index < 30
    assert np.allclose(original.predicted[unaffected], modified.predicted[unaffected])
    assert np.allclose(original.filtered[unaffected], modified.filtered[unaffected])


def test_walkforward_rejects_zero_variance_training_feature():
    x = np.ones((12, 1))
    with pytest.raises(ValueError, match="positive training-window variance"):
        expanding_walk_forward(x, initial_train=6, n_states=2, family="gaussian")
