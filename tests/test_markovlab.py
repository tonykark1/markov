import numpy as np

from markovlab.diagnostics import expected_durations, hard_state_agreement, normalized_entropy
from markovlab.first_passage import (
    expected_hitting_time,
    first_passage_probability,
    mixture_first_passage_probability,
)
from markovlab.hmm import fit_hmm


def test_entropy_bounds_and_certainty():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    h = normalized_entropy(p)
    assert np.isclose(h[0], 0.0, atol=1e-10)
    assert np.isclose(h[1], 1.0, atol=1e-10)


def test_first_passage_two_state_chain():
    p = np.array([[0.8, 0.2], [0.1, 0.9]])
    assert np.isclose(first_passage_probability(p, 0, 1, 1), 0.2)
    assert np.isclose(first_passage_probability(p, 0, 1, 2), 0.36)
    assert np.isclose(expected_hitting_time(p, 0, 1), 5.0)


def test_mixture_first_passage_counts_current_target_mass():
    p = np.array([[0.8, 0.2], [0.1, 0.9]])
    alpha = np.array([0.75, 0.25])
    got = mixture_first_passage_probability(p, alpha, 1, 1)
    assert np.isclose(got, 0.25 + 0.75 * 0.2)


def test_duration_and_agreement_helpers():
    p = np.array([[0.9, 0.1], [0.25, 0.75]])
    d = expected_durations(p)
    assert np.allclose(d, [10.0, 4.0])
    assert hard_state_agreement(np.array([0, 1, 1]), np.array([0, 0, 1])) == 2 / 3


def test_hmm_returns_valid_probabilities():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(-1.0, 0.4, (80, 2)), rng.normal(1.0, 0.4, (80, 2))]
    fit = fit_hmm(x, n_states=2, family="student_t", nu=5, n_init=3, random_state=4)
    assert fit.filtered.shape == (160, 2)
    assert np.allclose(fit.filtered.sum(axis=1), 1.0)
    assert np.allclose(fit.transition.sum(axis=1), 1.0)
    assert np.isfinite(fit.loglik)
