import pytest
import numpy as np
import pandas as pd
from evaluation.quantization.joint_space import JointSpace


def test_joint_space_build_contingency_table():
    """Should build 2D contingency table from clusters and bins."""
    text_clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    tabular_bins = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    joint_space = JointSpace()
    contingency = joint_space.build_contingency_table(text_clusters, tabular_bins)

    assert contingency.shape == (3, 3)
    assert contingency.sum() == 9


def test_joint_space_compute_joint_probabilities():
    """Should normalize contingency table to probabilities."""
    # Create uniform distribution: all combinations equally likely
    text_clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    tabular_bins = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    joint_space = JointSpace()
    joint_space.build_contingency_table(text_clusters, tabular_bins)
    joint_prob = joint_space.compute_joint_probabilities()

    # Should sum to 1
    assert np.abs(joint_prob.sum() - 1.0) < 1e-6

    # Each cell should be 1/9 (uniform distribution)
    assert np.allclose(joint_prob, 1/9)


def test_joint_space_marginal_distributions():
    """Should compute correct marginal distributions."""
    text_clusters = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    tabular_bins = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])

    joint_space = JointSpace()
    joint_space.build_contingency_table(text_clusters, tabular_bins)
    joint_space.compute_joint_probabilities()

    # Marginal text: [2/9, 3/9, 4/9]
    expected_text = np.array([2/9, 3/9, 4/9])
    np.testing.assert_allclose(joint_space.marginal_text, expected_text)

    # Marginal tabular: [4/9, 3/9, 2/9]
    expected_tabular = np.array([4/9, 3/9, 2/9])
    np.testing.assert_allclose(joint_space.marginal_tabular, expected_tabular)


def test_joint_space_mutual_information():
    """Should compute mutual information."""
    # Perfect correlation
    text_clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    tabular_bins = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    joint_space = JointSpace()
    joint_space.build_contingency_table(text_clusters, tabular_bins)
    mi = joint_space.get_mutual_information()

    # Perfect correlation should have high MI
    assert mi > 1.0


def test_joint_space_mutual_information_independent():
    """Should return ~0 MI for independent variables."""
    # Independent: text and tabular have no relationship
    text_clusters = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    tabular_bins = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    joint_space = JointSpace()
    joint_space.build_contingency_table(text_clusters, tabular_bins)
    mi = joint_space.get_mutual_information()

    # Independent should have near-zero MI
    assert mi < 0.1


def test_joint_space_compute_joint_probabilities_error():
    """Should raise error if contingency table not built first."""
    joint_space = JointSpace()

    with pytest.raises(ValueError, match="Must call build_contingency_table"):
        joint_space.compute_joint_probabilities()


def test_joint_space_mutual_information_error():
    """Should raise error if contingency table not built first."""
    joint_space = JointSpace()

    with pytest.raises(ValueError, match="Must call build_contingency_table"):
        joint_space.get_mutual_information()
