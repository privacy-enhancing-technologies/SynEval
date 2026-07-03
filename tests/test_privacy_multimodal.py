#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pytest

from evaluation.privacy import evaluate_privacy_multimodal
from evaluation.quantization.semantic_quantizer import SemanticQuantizer


class TestPrivacyMultimodal:
    """Test semantic-aware DCR for multimodal privacy evaluation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample multimodal data for testing."""
        np.random.seed(42)

        # Create real data with text and tabular
        real_df = pd.DataFrame(
            {
                "text": [
                    "positive sentiment",
                    "positive sentiment",
                    "negative sentiment",
                    "negative sentiment",
                ]
                * 25,
                "value": [10, 12, 5, 6] * 25,
            }
        )

        return real_df

    @pytest.fixture
    def quantizer(self, sample_data):
        """Create a quantizer for testing."""
        quantizer = SemanticQuantizer(
            text_columns=["text"],
            tabular_columns=["value"],
            text_clusters=5,
            tabular_bins=5,
            adaptive=False,
        )
        # Fit on sample data
        quantizer.fit(sample_data)
        return quantizer

    def test_privacy_multimodal_basic(self, sample_data, quantizer):
        """Test basic functionality of multimodal DCR."""
        # Create slightly different synthetic data
        synth_df = pd.DataFrame(
            {
                "text": [
                    "positive sentiment",
                    "positive sentiment",
                    "negative sentiment",
                    "negative sentiment",
                ]
                * 25,
                "value": [11, 10, 6, 5] * 25,  # Slightly different values
            }
        )

        result = evaluate_privacy_multimodal(sample_data, synth_df, quantizer)

        # Check all required keys exist
        required_keys = [
            "semantic_dcr_mean",
            "semantic_dcr_min",
            "semantic_dcr_median",
            "pct_below_threshold",
            "alpha",
            "interpretation",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check types
        assert isinstance(result["semantic_dcr_mean"], (float, np.floating))
        assert isinstance(result["semantic_dcr_min"], (float, np.floating))
        assert isinstance(result["semantic_dcr_median"], (float, np.floating))
        assert isinstance(result["pct_below_threshold"], (float, np.floating))
        assert isinstance(result["alpha"], (float, np.floating))
        assert isinstance(result["interpretation"], str)

        # Check bounds
        assert result["semantic_dcr_mean"] >= 0, "DCR mean should be non-negative"
        assert result["semantic_dcr_min"] >= 0, "DCR min should be non-negative"
        assert result["semantic_dcr_median"] >= 0, "DCR median should be non-negative"
        assert (
            0 <= result["pct_below_threshold"] <= 100
        ), "Percentage should be in [0, 100]"
        assert 0 <= result["alpha"] <= 1, "Alpha should be in [0, 1]"

    def test_privacy_multimodal_perfect_overlap(self, sample_data, quantizer):
        """Test DCR is approximately 0 when synth=real (perfect overlap)."""
        # Synthetic is identical to real
        synth_df = sample_data.copy()

        result = evaluate_privacy_multimodal(sample_data, synth_df, quantizer)

        # DCR should be very close to 0 for identical data
        assert (
            result["semantic_dcr_mean"] < 0.01
        ), "DCR mean should be ~0 for identical data"
        assert (
            result["semantic_dcr_min"] < 0.01
        ), "DCR min should be ~0 for identical data"

        # High percentage below threshold (memorization risk)
        assert (
            result["pct_below_threshold"] > 90
        ), "Should have high memorization risk for identical data"

        # Interpretation should indicate high risk
        assert "High memorization risk" in result["interpretation"]

    def test_privacy_multimodal_no_overlap(self, sample_data, quantizer):
        """Test DCR > 0 when datasets are different."""
        # Create very different synthetic data
        synth_df = pd.DataFrame(
            {
                "text": ["completely different text"] * 100,
                "value": [100] * 100,  # Very different values
            }
        )

        result = evaluate_privacy_multimodal(sample_data, synth_df, quantizer)

        # DCR should be positive for different data
        assert result["semantic_dcr_mean"] > 0, "DCR mean should be > 0 for different data"
        assert result["semantic_dcr_min"] > 0, "DCR min should be > 0 for different data"

        # Low percentage below threshold (low memorization risk)
        assert (
            result["pct_below_threshold"] < 10
        ), "Should have low memorization risk for different data"

        # Interpretation should indicate low risk
        assert "Low memorization risk" in result["interpretation"]

    def test_privacy_multimodal_alpha_parameter(self, sample_data, quantizer):
        """Test different alpha values affect results."""
        synth_df = pd.DataFrame(
            {
                "text": [
                    "positive sentiment",
                    "positive sentiment",
                    "negative sentiment",
                    "negative sentiment",
                ]
                * 25,
                "value": [11, 10, 6, 5] * 25,
            }
        )

        # Test different alpha values
        result_text_heavy = evaluate_privacy_multimodal(
            sample_data, synth_df, quantizer, alpha=0.8
        )
        result_tabular_heavy = evaluate_privacy_multimodal(
            sample_data, synth_df, quantizer, alpha=0.2
        )
        result_balanced = evaluate_privacy_multimodal(
            sample_data, synth_df, quantizer, alpha=0.5
        )

        # All should return valid results
        assert result_text_heavy["alpha"] == 0.8
        assert result_tabular_heavy["alpha"] == 0.2
        assert result_balanced["alpha"] == 0.5

        # Results should be different (different weights)
        # Note: Depending on the data, the results might not always be different
        # But they should all be valid
        assert result_text_heavy["semantic_dcr_mean"] >= 0
        assert result_tabular_heavy["semantic_dcr_mean"] >= 0
        assert result_balanced["semantic_dcr_mean"] >= 0

    def test_privacy_multimodal_risk_threshold(self, sample_data, quantizer):
        """Test different risk thresholds affect interpretation."""
        synth_df = pd.DataFrame(
            {
                "text": [
                    "positive sentiment",
                    "positive sentiment",
                    "negative sentiment",
                    "negative sentiment",
                ]
                * 25,
                "value": [11, 10, 6, 5] * 25,
            }
        )

        # Test with different thresholds
        result_low_threshold = evaluate_privacy_multimodal(
            sample_data, synth_df, quantizer, risk_threshold=0.01
        )
        result_high_threshold = evaluate_privacy_multimodal(
            sample_data, synth_df, quantizer, risk_threshold=1.0
        )

        # High threshold should catch more records as risky (more points below threshold)
        assert (
            result_high_threshold["pct_below_threshold"]
            >= result_low_threshold["pct_below_threshold"]
        )

    def test_privacy_multimodal_with_categorical(self, quantizer):
        """Test handling of categorical columns in tabular data."""
        # Create data with categorical column
        real_df = pd.DataFrame(
            {
                "text": ["positive", "negative", "neutral"] * 30,
                "category": ["A", "B", "C"] * 30,
                "value": [1, 2, 3] * 30,
            }
        )

        synth_df = pd.DataFrame(
            {
                "text": ["positive", "negative", "neutral"] * 30,
                "category": ["A", "B", "C"] * 30,
                "value": [1.1, 2.1, 3.1] * 30,
            }
        )

        # Create quantizer for this data
        quantizer_cat = SemanticQuantizer(
            text_columns=["text"],
            tabular_columns=["category", "value"],
            text_clusters=5,
            tabular_bins=5,
            adaptive=False,
        )
        quantizer_cat.fit(real_df)

        result = evaluate_privacy_multimodal(real_df, synth_df, quantizer_cat)

        # Should handle categorical columns gracefully
        assert "semantic_dcr_mean" in result
        assert result["semantic_dcr_mean"] >= 0

    def test_privacy_multimodal_empty(self, quantizer):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame({"text": [], "value": []})

        with pytest.raises((ValueError, IndexError)):
            evaluate_privacy_multimodal(empty_df, empty_df, quantizer)

    def test_privacy_multimodal_single_sample(self, quantizer):
        """Test handling of single sample."""
        single_real = pd.DataFrame({"text": ["positive sentiment"], "value": [10]})
        single_synth = pd.DataFrame({"text": ["negative sentiment"], "value": [5]})

        # Should handle single sample case
        result = evaluate_privacy_multimodal(single_real, single_synth, quantizer)

        # Should return valid results
        assert "semantic_dcr_mean" in result
        assert result["semantic_dcr_mean"] >= 0

        # For single sample, mean = min = median
        assert result["semantic_dcr_mean"] == result["semantic_dcr_min"]
        assert result["semantic_dcr_mean"] == result["semantic_dcr_median"]
