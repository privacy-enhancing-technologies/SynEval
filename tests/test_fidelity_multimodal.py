#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pytest

from evaluation.fidelity import compute_jsd, evaluate_fidelity_multimodal
from evaluation.quantization.semantic_quantizer import SemanticQuantizer


class TestJointSpectralDivergence:
    """Test Joint Spectral Divergence computation."""

    def test_compute_jsd_identical_distributions(self):
        """Test JSD between identical distributions returns 0."""
        P = np.array([[0.25, 0.25], [0.25, 0.25]])
        Q = np.array([[0.25, 0.25], [0.25, 0.25]])

        jsd = compute_jsd(P, Q)

        assert jsd == pytest.approx(0.0, abs=1e-10), "JSD should be 0 for identical distributions"

    def test_compute_jsd_different_distributions(self):
        """Test JSD between different distributions returns positive value."""
        # P is uniform
        P = np.array([[0.25, 0.25], [0.25, 0.25]])
        # Q is concentrated in one corner
        Q = np.array([[0.7, 0.1], [0.1, 0.1]])

        jsd = compute_jsd(P, Q)

        assert jsd > 0, "JSD should be positive for different distributions"
        assert jsd <= 1.0, "JSD should be bounded by 1.0"

    def test_compute_jsd_symmetry(self):
        """Test that JSD is symmetric: JSD(P, Q) = JSD(Q, P)."""
        P = np.array([[0.4, 0.1], [0.1, 0.4]])
        Q = np.array([[0.25, 0.25], [0.25, 0.25]])

        jsd_pq = compute_jsd(P, Q)
        jsd_qp = compute_jsd(Q, P)

        assert jsd_pq == pytest.approx(jsd_qp, abs=1e-10), "JSD should be symmetric"

    def test_compute_jsd_normalization(self):
        """Test that JSD handles non-normalized distributions correctly."""
        # Non-normalized distributions (should be normalized internally)
        P = np.array([[2.0, 2.0], [2.0, 2.0]])  # Sums to 8.0
        Q = np.array([[1.0, 1.0], [1.0, 1.0]])  # Sums to 4.0

        jsd = compute_jsd(P, Q)

        # After normalization, they're identical, so JSD should be ~0
        assert jsd == pytest.approx(0.0, abs=1e-10), "JSD should normalize distributions"

    def test_compute_jsd_extreme_divergence(self):
        """Test JSD with completely different distributions."""
        # P concentrated at top-left
        P = np.array([[0.9, 0.05], [0.05, 0.0]])
        # Q concentrated at bottom-right
        Q = np.array([[0.0, 0.05], [0.05, 0.9]])

        jsd = compute_jsd(P, Q)

        # Should be high divergence, but bounded
        assert jsd > 0.5, "JSD should be high for very different distributions"
        assert jsd <= 1.0, "JSD should be bounded by 1.0"


class TestFidelityMultimodal:
    """Test multimodal fidelity evaluation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample multimodal data for testing."""
        np.random.seed(42)

        # Create real data: correlated text and tabular
        real_df = pd.DataFrame(
            {
                "text": [
                    "positive sentiment",
                    "positive sentiment",
                    "negative sentiment",
                    "negative sentiment",
                ]
                * 25,
                "value": [10, 12, 5, 6] * 25,  # High values with positive, low with negative
            }
        )

        return real_df

    @pytest.fixture
    def good_synth_data(self, sample_data):
        """Create synthetic data that preserves joint distribution."""
        # Good synthetic: preserves correlation
        synth_df = pd.DataFrame(
            {
                "text": [
                    "positive sentiment",
                    "positive sentiment",
                    "negative sentiment",
                    "negative sentiment",
                ]
                * 25,
                "value": [11, 10, 6, 5] * 25,  # Similar correlation
            }
        )
        return synth_df

    @pytest.fixture
    def bad_synth_data(self, sample_data):
        """Create synthetic data that breaks joint distribution (Tilted Data scenario)."""
        # Bad synthetic: marginals look OK but correlation is broken
        synth_df = pd.DataFrame(
            {
                "text": [
                    "positive sentiment",
                    "positive sentiment",
                    "negative sentiment",
                    "negative sentiment",
                ]
                * 25,
                "value": [5, 6, 10, 12] * 25,  # Reversed correlation!
            }
        )
        return synth_df

    @pytest.fixture
    def quantizer(self, sample_data):
        """Create a quantizer for testing."""
        quantizer = SemanticQuantizer(
            text_columns=["text"],
            tabular_columns=["value"],
            text_clusters=5,
            tabular_bins=5,
            adaptive=False
        )
        # Fit on sample data
        quantizer.fit(sample_data)
        return quantizer

    def test_evaluate_fidelity_multimodal_good_synthetic(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test that good synthetic data gets low JSD."""
        result = evaluate_fidelity_multimodal(sample_data, good_synth_data, quantizer)

        assert "joint_spectral_divergence" in result
        assert "marginal_text_divergence" in result
        assert "marginal_tabular_divergence" in result
        assert "interpretation" in result

        # Good synthetic should have low JSD
        assert (
            result["joint_spectral_divergence"] < 0.3
        ), "Good synthetic should have low JSD"
        assert isinstance(result["interpretation"], str)

    def test_evaluate_fidelity_multimodal_bad_synthetic(
        self, sample_data, bad_synth_data, quantizer
    ):
        """Test that Tilted Data (broken joint) gets high JSD."""
        result = evaluate_fidelity_multimodal(sample_data, bad_synth_data, quantizer)

        assert "joint_spectral_divergence" in result
        assert "marginal_text_divergence" in result
        assert "marginal_tabular_divergence" in result
        assert "interpretation" in result

        # Bad synthetic (tilted) should have higher joint JSD than marginal JSD
        jsd = result["joint_spectral_divergence"]
        marginal_text = result["marginal_text_divergence"]
        marginal_tabular = result["marginal_tabular_divergence"]

        # The joint should show the problem (higher divergence)
        assert (
            jsd > marginal_text or jsd > marginal_tabular
        ), "Joint JSD should be higher for broken correlations"

    def test_evaluate_fidelity_multimodal_interpretation(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test interpretation text generation."""
        result = evaluate_fidelity_multimodal(sample_data, good_synth_data, quantizer)

        interpretation = result["interpretation"]

        # Should contain meaningful interpretation
        assert len(interpretation) > 0
        assert isinstance(interpretation, str)

        # Should have reasonable categories
        valid_interpretations = [
            "Excellent",
            "Good",
            "Fair",
            "Poor",
            "Very poor",
            "excellent",
            "good",
            "fair",
            "poor",
        ]
        assert any(
            phrase in interpretation for phrase in valid_interpretations
        ), "Should contain quality assessment"

    def test_evaluate_fidelity_multimodal_return_structure(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test that return structure matches expected format."""
        result = evaluate_fidelity_multimodal(sample_data, good_synth_data, quantizer)

        # Check all required keys exist
        required_keys = [
            "joint_spectral_divergence",
            "marginal_text_divergence",
            "marginal_tabular_divergence",
            "interpretation",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check types
        assert isinstance(result["joint_spectral_divergence"], (float, np.floating))
        assert isinstance(result["marginal_text_divergence"], (float, np.floating))
        assert isinstance(result["marginal_tabular_divergence"], (float, np.floating))
        assert isinstance(result["interpretation"], str)

        # Check bounds
        assert (
            0 <= result["joint_spectral_divergence"] <= 1
        ), "JSD should be in [0, 1]"
        assert (
            0 <= result["marginal_text_divergence"] <= 1
        ), "Marginal JSD should be in [0, 1]"
        assert (
            0 <= result["marginal_tabular_divergence"] <= 1
        ), "Marginal JSD should be in [0, 1]"

    def test_compute_jsd_with_zeros(self):
        """Test JSD computation handles zeros in distributions correctly."""
        # Distribution with zeros
        P = np.array([[0.5, 0.5], [0.0, 0.0]])
        Q = np.array([[0.0, 0.0], [0.5, 0.5]])

        jsd = compute_jsd(P, Q)

        # Should handle zeros gracefully (no NaN or inf)
        assert not np.isnan(jsd), "JSD should not be NaN"
        assert not np.isinf(jsd), "JSD should not be infinite"
        assert jsd > 0, "JSD should be positive for different distributions"
        assert jsd <= 1.0, "JSD should be bounded by 1.0"
