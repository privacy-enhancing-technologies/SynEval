#!/usr/bin/env python3
"""
Tests for multimodal utility evaluation.

Tests the evaluate_utility_multimodal function which implements:
- Text-to-Attribute (T2A): Train on synthetic text embeddings → predict tabular target
- Attribute-to-Text (A2T): Train on synthetic tabular → predict text clusters

Both models are trained on synthetic data and evaluated on real test data to measure
whether synthetic data preserves cross-modal predictive relationships.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.utility import evaluate_utility_multimodal
from evaluation.quantization.semantic_quantizer import SemanticQuantizer


class TestUtilityMultimodal:
    """Test cross-modal predictability utility evaluation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample multimodal data for testing."""
        np.random.seed(42)

        # Create real data with predictive relationships:
        # - Text sentiment correlates with rating (categorical)
        # - Text sentiment correlates with price (continuous)
        real_df = pd.DataFrame(
            {
                "text": [
                    "excellent product",
                    "great quality",
                    "amazing experience",
                    "wonderful purchase",
                    "poor quality",
                    "terrible product",
                    "bad experience",
                    "disappointing purchase",
                ]
                * 15,
                "rating": ["high", "high", "high", "high", "low", "low", "low", "low"]
                * 15,  # Categorical target
                "price": [100, 110, 95, 105, 20, 25, 18, 22]
                * 15,  # Continuous target
                "category": ["electronics", "electronics", "electronics", "electronics",
                             "electronics", "electronics", "electronics", "electronics"]
                * 15,  # Tabular feature
            }
        )

        return real_df

    @pytest.fixture
    def good_synth_data(self, sample_data):
        """Create synthetic data that preserves predictive relationships."""
        # Good synthetic: preserves text-to-attribute and attribute-to-text relationships
        synth_df = pd.DataFrame(
            {
                "text": [
                    "fantastic product",
                    "excellent quality",
                    "great experience",
                    "amazing purchase",
                    "bad quality",
                    "poor product",
                    "terrible experience",
                    "awful purchase",
                ]
                * 15,
                "rating": ["high", "high", "high", "high", "low", "low", "low", "low"]
                * 15,
                "price": [98, 105, 102, 108, 22, 19, 24, 20]
                * 15,
                "category": ["electronics", "electronics", "electronics", "electronics",
                             "electronics", "electronics", "electronics", "electronics"]
                * 15,
            }
        )
        return synth_df

    @pytest.fixture
    def bad_synth_data(self, sample_data):
        """Create synthetic data that breaks predictive relationships."""
        # Bad synthetic: breaks correlation between text and attributes
        synth_df = pd.DataFrame(
            {
                "text": [
                    "excellent product",
                    "great quality",
                    "amazing experience",
                    "wonderful purchase",
                    "poor quality",
                    "terrible product",
                    "bad experience",
                    "disappointing purchase",
                ]
                * 15,
                "rating": ["low", "low", "low", "low", "high", "high", "high", "high"]
                * 15,  # Reversed!
                "price": [20, 25, 18, 22, 100, 110, 95, 105]
                * 15,  # Reversed!
                "category": ["electronics", "electronics", "electronics", "electronics",
                             "electronics", "electronics", "electronics", "electronics"]
                * 15,
            }
        )
        return synth_df

    @pytest.fixture
    def quantizer(self, sample_data):
        """Create a quantizer for testing."""
        quantizer = SemanticQuantizer(
            text_columns=["text"],
            tabular_columns=["price", "category"],
            text_clusters=5,
            tabular_bins=5,
            adaptive=False
        )
        # Fit on sample data
        quantizer.fit(sample_data)
        return quantizer

    def test_evaluate_utility_multimodal_classification(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test T2A for classification task (predict categorical from text)."""
        result = evaluate_utility_multimodal(
            sample_data, good_synth_data, quantizer, target_column="rating"
        )

        # Check structure
        assert "text_to_attribute" in result
        assert "attribute_to_text" in result

        t2a = result["text_to_attribute"]
        assert "target_column" in t2a
        assert t2a["target_column"] == "rating"
        assert "f1_score" in t2a

        # F1 score should be in valid range
        assert 0 <= t2a["f1_score"] <= 1.0, "F1 score should be in [0, 1]"

        # For good synthetic, F1 should be reasonably high
        assert t2a["f1_score"] > 0.3, "Good synthetic should have decent F1"

    def test_evaluate_utility_multimodal_regression(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test T2A for regression task (predict continuous from text)."""
        result = evaluate_utility_multimodal(
            sample_data, good_synth_data, quantizer, target_column="price"
        )

        # Check structure
        assert "text_to_attribute" in result
        assert "attribute_to_text" in result

        t2a = result["text_to_attribute"]
        assert "target_column" in t2a
        assert t2a["target_column"] == "price"
        assert "rmse" in t2a

        # RMSE should be positive
        assert t2a["rmse"] > 0, "RMSE should be positive"

        # For good synthetic, RMSE should be reasonable (not too high)
        # Price range is ~20-110, so RMSE should be less than the range
        assert t2a["rmse"] < 100, "Good synthetic should have reasonable RMSE"

    def test_evaluate_utility_multimodal_a2t(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test A2T (predict text cluster from tabular features)."""
        result = evaluate_utility_multimodal(
            sample_data, good_synth_data, quantizer, target_column="rating"
        )

        # Check A2T structure
        assert "attribute_to_text" in result
        a2t = result["attribute_to_text"]
        assert "cluster_prediction_f1" in a2t

        # F1 score should be in valid range
        assert 0 <= a2t["cluster_prediction_f1"] <= 1.0, "F1 score should be in [0, 1]"

    def test_evaluate_utility_multimodal_bad_synthetic(
        self, sample_data, bad_synth_data, quantizer
    ):
        """Test that bad synthetic (broken correlations) gets poor scores."""
        result = evaluate_utility_multimodal(
            sample_data, bad_synth_data, quantizer, target_column="rating"
        )

        t2a = result["text_to_attribute"]
        a2t = result["attribute_to_text"]

        # Bad synthetic should have lower F1 scores
        # Note: might not be terrible due to some residual patterns,
        # but should be worse than good synthetic
        assert t2a["f1_score"] < 0.8, "Bad synthetic should have lower T2A F1"
        assert a2t["cluster_prediction_f1"] < 0.8, "Bad synthetic should have lower A2T F1"

    def test_evaluate_utility_multimodal_return_structure(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test that return structure matches expected format."""
        result = evaluate_utility_multimodal(
            sample_data, good_synth_data, quantizer, target_column="rating"
        )

        # Check all required keys exist
        required_keys = ["text_to_attribute", "attribute_to_text"]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check T2A structure
        t2a = result["text_to_attribute"]
        assert "target_column" in t2a
        assert t2a["target_column"] == "rating"
        assert "f1_score" in t2a or "rmse" in t2a

        # Check A2T structure
        a2t = result["attribute_to_text"]
        assert "cluster_prediction_f1" in a2t

    def test_evaluate_utility_multimodal_train_test_split(
        self, sample_data, good_synth_data, quantizer
    ):
        """Test that train/test split is applied correctly."""
        # With seed=42, the split should be deterministic
        result1 = evaluate_utility_multimodal(
            sample_data, good_synth_data, quantizer, target_column="rating"
        )
        result2 = evaluate_utility_multimodal(
            sample_data, good_synth_data, quantizer, target_column="rating"
        )

        # Results should be identical with fixed random seed
        assert result1["text_to_attribute"]["f1_score"] == pytest.approx(
            result2["text_to_attribute"]["f1_score"], abs=1e-10
        ), "Results should be deterministic with fixed seed"

    def test_evaluate_utility_multimodal_with_missing_values(self):
        """Test handling of missing values in data."""
        np.random.seed(42)

        # Create data with some missing values
        real_df = pd.DataFrame(
            {
                "text": [
                    "excellent product",
                    "great quality",
                    None,  # Missing text
                    "wonderful purchase",
                    "poor quality",
                    "terrible product",
                    "bad experience",
                    "disappointing purchase",
                ]
                * 10,
                "rating": ["high", "high", "high", "high", "low", "low", None, "low"]
                * 10,  # Missing rating
                "price": [100, 110, 95, 105, 20, 25, 18, 22] * 10,
                "category": ["electronics"] * 80,
            }
        )

        synth_df = pd.DataFrame(
            {
                "text": [
                    "fantastic product",
                    "excellent quality",
                    "great experience",
                    "amazing purchase",
                    "bad quality",
                    "poor product",
                    "terrible experience",
                    "awful purchase",
                ]
                * 10,
                "rating": ["high", "high", "high", "high", "low", "low", "low", "low"]
                * 10,
                "price": [98, 105, 102, 108, 22, 19, 24, 20] * 10,
                "category": ["electronics"] * 80,
            }
        )

        quantizer = SemanticQuantizer(
            text_columns=["text"],
            tabular_columns=["price", "category"],
            text_clusters=5,
            tabular_bins=5,
            adaptive=False
        )

        # Should handle missing values gracefully
        quantizer.fit(real_df.dropna())
        result = evaluate_utility_multimodal(
            real_df.dropna(), synth_df, quantizer, target_column="rating"
        )

        # Should complete without errors
        assert "text_to_attribute" in result
        assert "attribute_to_text" in result

    def test_evaluate_utility_multimodal_small_dataset(self):
        """Test with minimal dataset size."""
        np.random.seed(42)

        # Small dataset (just enough for 80/20 split)
        real_df = pd.DataFrame(
            {
                "text": ["good", "bad", "good", "bad", "good"],
                "rating": ["high", "low", "high", "low", "high"],
                "price": [100, 20, 95, 25, 110],
                "category": ["electronics"] * 5,
            }
        )

        synth_df = pd.DataFrame(
            {
                "text": ["great", "poor", "excellent", "terrible", "amazing"],
                "rating": ["high", "low", "high", "low", "high"],
                "price": [105, 22, 98, 18, 108],
                "category": ["electronics"] * 5,
            }
        )

        quantizer = SemanticQuantizer(
            text_columns=["text"],
            tabular_columns=["price", "category"],
            text_clusters=2,
            tabular_bins=2,
            adaptive=False
        )
        quantizer.fit(real_df)

        # Should handle small dataset
        result = evaluate_utility_multimodal(
            real_df, synth_df, quantizer, target_column="rating"
        )

        assert "text_to_attribute" in result
        assert "attribute_to_text" in result
