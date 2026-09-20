import numpy as np
import pytest

from markovlab.alignment import (
    align_by_distribution_distance,
    distribution_distance_matrix,
)


def _covariances(values: list[float]) -> np.ndarray:
    return np.asarray([[[value]] for value in values], dtype=float)


@pytest.mark.parametrize("metric", ["symmetric_kl", "bhattacharyya", "wasserstein"])
def test_distribution_alignment_uses_covariance_when_means_are_identical(metric):
    reference_means = np.zeros((2, 1))
    candidate_means = np.zeros((2, 1))
    reference_covariances = _covariances([1.0, 9.0])
    candidate_covariances = _covariances([9.0, 1.0])

    order = align_by_distribution_distance(
        reference_means,
        reference_covariances,
        candidate_means,
        candidate_covariances,
        metric=metric,
    )
    assert np.array_equal(order, [1, 0])


@pytest.mark.parametrize("metric", ["symmetric_kl", "bhattacharyya", "wasserstein"])
def test_distribution_alignment_recovers_joint_mean_covariance_permutation(metric):
    reference_means = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    reference_covariances = np.array(
        [
            [[1.0, 0.2], [0.2, 2.0]],
            [[3.0, -0.4], [-0.4, 1.0]],
            [[0.8, 0.1], [0.1, 0.7]],
        ]
    )
    permutation = np.array([2, 0, 1])
    candidate_means = reference_means[permutation]
    candidate_covariances = reference_covariances[permutation]

    order = align_by_distribution_distance(
        reference_means,
        reference_covariances,
        candidate_means,
        candidate_covariances,
        metric=metric,
    )
    assert np.array_equal(order, [1, 2, 0])


def test_symmetric_kl_is_zero_for_identical_distributions():
    means = np.array([[0.0, 1.0], [2.0, -1.0]])
    covariances = np.array(
        [
            [[1.0, 0.2], [0.2, 2.0]],
            [[2.0, 0.1], [0.1, 0.8]],
        ]
    )
    cost = distribution_distance_matrix(
        means,
        covariances,
        means,
        covariances,
        metric="symmetric_kl",
    )
    assert np.allclose(np.diag(cost), 0.0, atol=1e-10)
    assert np.all(cost >= -1e-10)


def test_distribution_alignment_rejects_non_positive_definite_covariance():
    means = np.zeros((2, 1))
    bad = _covariances([1.0, -1.0])
    good = _covariances([1.0, 2.0])
    with pytest.raises(ValueError, match="positive definite"):
        align_by_distribution_distance(means, bad, means, good)


def test_distribution_alignment_rejects_unknown_metric():
    means = np.zeros((2, 1))
    covariances = _covariances([1.0, 2.0])
    with pytest.raises(ValueError, match="metric"):
        distribution_distance_matrix(
            means,
            covariances,
            means,
            covariances,
            metric="not_a_metric",
        )
