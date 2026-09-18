import numpy as np
import pytest

from markovlab.first_passage import (
    expected_hitting_time,
    first_passage_probability,
    mixture_first_passage_probability,
)


def chain():
    return np.array([[0.8, 0.2], [0.1, 0.9]])


def test_first_passage_two_state_chain():
    p = chain()
    assert np.isclose(first_passage_probability(p, 0, 1, 1), 0.2)
    assert np.isclose(first_passage_probability(p, 0, 1, 2), 0.36)
    assert np.isclose(expected_hitting_time(p, 0, 1), 5.0)


def test_horizon_zero():
    p = chain()
    assert first_passage_probability(p, 0, 1, 0) == 0.0
    assert first_passage_probability(p, 1, 1, 0) == 1.0


def test_mixture_first_passage_counts_current_target_mass():
    p = chain()
    alpha = np.array([0.75, 0.25])
    got = mixture_first_passage_probability(p, alpha, 1, 1)
    assert np.isclose(got, 0.25 + 0.75 * 0.2)
    assert np.isclose(mixture_first_passage_probability(p, alpha, 1, 0), 0.25)


def test_unreachable_target_has_infinite_expected_hitting_time():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isinf(expected_hitting_time(p, 0, 1))
    assert first_passage_probability(p, 0, 1, 100) == 0.0


@pytest.mark.parametrize(
    "state_name,state",
    [("start", -1), ("start", 2), ("target", -1), ("target", 2)],
)
def test_invalid_states_raise(state_name, state):
    p = chain()
    kwargs = {"start_state": 0, "target_state": 1, "horizon": 2}
    kwargs[f"{state_name}_state"] = state
    with pytest.raises(ValueError):
        first_passage_probability(p, **kwargs)


def test_negative_horizon_raises_for_both_apis():
    p = chain()
    with pytest.raises(ValueError):
        first_passage_probability(p, 0, 1, -1)
    with pytest.raises(ValueError):
        mixture_first_passage_probability(p, np.array([0.5, 0.5]), 1, -1)


def test_invalid_mixture_probabilities_raise():
    p = chain()
    for alpha in [
        np.array([-1.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([np.nan, 1.0]),
    ]:
        with pytest.raises(ValueError):
            mixture_first_passage_probability(p, alpha, 1, 3)


def test_invalid_transition_matrix_raises():
    with pytest.raises(ValueError):
        first_passage_probability(np.array([[0.5, 0.5], [0.2, 0.2]]), 0, 1, 2)
