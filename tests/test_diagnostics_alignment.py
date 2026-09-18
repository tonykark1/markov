import numpy as np
import pytest

from markovlab.alignment import align_by_mean_distance, reorder_states
from markovlab.diagnostics import (
    expected_durations,
    hard_state_agreement,
    model_disagreement,
    normalized_entropy,
)


def test_entropy_bounds_and_certainty():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    h = normalized_entropy(p)
    assert np.isclose(h[0], 0.0, atol=1e-10)
    assert np.isclose(h[1], 1.0, atol=1e-10)


def test_entropy_rejects_invalid_rows():
    for p in [
        np.array([[0.0, 0.0]]),
        np.array([[-0.1, 1.1]]),
        np.array([[np.nan, 1.0]]),
    ]:
        with pytest.raises(ValueError):
            normalized_entropy(p)


def test_expected_duration_handles_absorbing_state():
    p = np.array([[1.0, 0.0], [0.25, 0.75]])
    d = expected_durations(p)
    assert np.isinf(d[0])
    assert np.isclose(d[1], 4.0)


def test_hard_state_agreement():
    assert hard_state_agreement(np.array([0, 1, 1]), np.array([0, 0, 1])) == 2 / 3
    with pytest.raises(ValueError):
        hard_state_agreement(np.array([]), np.array([]))


def test_model_disagreement():
    x = np.array([[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
    got = model_disagreement(x)
    assert got.shape == (2,)
    assert np.all(got > 0)


def test_model_disagreement_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        model_disagreement(np.array([[1.2, -0.2]]))


def test_hungarian_state_alignment_recovers_permutation():
    reference = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    candidate = reference[[2, 0, 1]]
    order = align_by_mean_distance(reference, candidate)
    assert np.array_equal(order, [1, 2, 0])
    aligned = reorder_states(candidate, order, axis=0)
    assert np.allclose(aligned, reference)


def test_alignment_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        align_by_mean_distance(np.zeros((2, 2)), np.zeros((3, 2)))
