import numpy as np
import pytest

from markovlab.parallel import joint_probabilities, joint_state_labels, joint_transition


def test_joint_probabilities_match_rowwise_kronecker_product():
    a = np.array([[0.8, 0.2], [0.4, 0.6]])
    b = np.array([[0.25, 0.75], [0.9, 0.1]])
    got = joint_probabilities(a, b)
    expected = np.vstack([np.kron(a[0], b[0]), np.kron(a[1], b[1])])
    assert np.allclose(got, expected)
    assert np.allclose(got.sum(axis=1), 1.0)


def test_joint_transition_matches_kronecker_product():
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.7, 0.3], [0.4, 0.6]])
    got = joint_transition(a, b)
    expected = np.kron(a, b)
    assert np.allclose(got, expected)
    assert np.allclose(got.sum(axis=1), 1.0)


def test_joint_labels_use_same_lexicographic_order():
    got = joint_state_labels(("Momentum", "Value"), ("Quality", "Cyclical"))
    assert got == (
        ("Momentum", "Quality"),
        ("Momentum", "Cyclical"),
        ("Value", "Quality"),
        ("Value", "Cyclical"),
    )


def test_parallel_helpers_validate_inputs():
    with pytest.raises(ValueError, match="at least two"):
        joint_probabilities(np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="same number"):
        joint_probabilities(
            np.array([[0.5, 0.5]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
        )
    with pytest.raises(ValueError, match="sum to one"):
        joint_transition(
            np.array([[0.8, 0.1], [0.2, 0.8]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
        )
