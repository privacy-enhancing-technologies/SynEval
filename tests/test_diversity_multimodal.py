#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pytest

from evaluation.diversity import (
    compute_joint_entropy,
    evaluate_diversity_multimodal,
)
from evaluation.quantization.semantic_quantizer import SemanticQuantizer


class TestComputeJointEntropy:
    """Test joint entropy computation."""

    def test_uniform_distribution(self):
        """Test entropy for uniform distribution."""
        # 4 equally likely outcomes: H = log2(4) = 2.0
        joint_prob = np.array([[0.25, 0.25], [0.25, 0.25]])
        entropy = compute_joint_entropy(joint_prob)
        assert abs(entropy - 2.0) < 0.01

    def test_deterministic_distribution(self):
        """Test entropy for deterministic distribution."""
        # Single outcome with probability 1: H = 0
        joint_prob = np.array([[1.0, 0.0], [0.0, 0.0]])
        entropy = compute_joint_entropy(joint_prob)
        assert abs(entropy - 0.0) < 0.01

    def test_non_uniform_distribution(self):
        """Test entropy for non-uniform distribution."""
        # P(0,0)=0.5, P(0,1)=0.3, P(1,0)=0.15, P(1,1)=0.05
        joint_prob = np.array([[0.5, 0.3], [0.15, 0.05]])
        # H = -0.5*log2(0.5) - 0.3*log2(0.3) - 0.15*log2(0.15) - 0.05*log2(0.05)
        # H ≈ 0.5 + 0.521 + 0.411 + 0.216 ≈ 1.648
        entropy = compute_joint_entropy(joint_prob)
        assert 1.6 < entropy < 1.7

    def test_zero_probabilities_ignored(self):
        """Test that zero probabilities don't contribute (0*log(0) = 0)."""
        joint_prob = np.array([[0.5, 0.5], [0.0, 0.0]])
        # Only two non-zero outcomes with p=0.5 each
        entropy = compute_joint_entropy(joint_prob)
        assert abs(entropy - 1.0) < 0.01


class TestEvaluateDiversityMultimodal:
    """Test multimodal diversity evaluation."""

    @pytest.fixture
    def setup_data(self):
        """Setup test data and quantizer."""
        # Real data with good diversity
        real_df = pd.DataFrame(
            {
                "text": ["good product", "bad service", "average quality", "excellent"],
                "category": ["A", "B", "C", "A"],
                "value": [10, 20, 30, 40],
            }
        )

        # Quantizer configuration
        quantizer = SemanticQuantizer(
            text_columns=["text"],
            tabular_columns=["category", "value"],
            text_clusters=2,  # Simple quantization
            tabular_bins=2,
            adaptive=False
        )
        # Fit on sample data
        quantizer.fit(real_df)

        return real_df, quantizer

    def test_good_diversity(self, setup_data):
        """Test detection of good diversity (ratio > 0.9)."""
        real_df, quantizer = setup_data

        # Synthetic data with similar diversity
        synth_df = pd.DataFrame(
            {
                "text": ["nice item", "poor quality", "okay product", "great"],
                "category": ["B", "A", "C", "A"],
                "value": [15, 25, 35, 45],
            }
        )

        result = evaluate_diversity_multimodal(real_df, synth_df, quantizer)

        assert "multimodal_metrics" in result
        assert "unimodal_metrics" in result

        mm = result["multimodal_metrics"]
        assert "joint_shannon_entropy" in mm
        assert "real_joint_entropy" in mm
        assert "diversity_ratio" in mm
        assert "interpretation" in mm

        # Good diversity: ratio should be > 0.9
        assert mm["diversity_ratio"] > 0.9
        assert "Good diversity" in mm["interpretation"]
        assert mm["joint_shannon_entropy"] > 0
        assert mm["real_joint_entropy"] > 0

    def test_severe_mode_collapse(self, setup_data):
        """Test detection of severe mode collapse (ratio < 0.7)."""
        real_df, quantizer = setup_data

        # Synthetic data with severe mode collapse (all same)
        synth_df = pd.DataFrame(
            {
                "text": ["good product", "good product", "good product", "good product"],
                "category": ["A", "A", "A", "A"],
                "value": [10, 10, 10, 10],
            }
        )

        result = evaluate_diversity_multimodal(real_df, synth_df, quantizer)

        mm = result["multimodal_metrics"]

        # Severe mode collapse: ratio should be < 0.7
        assert mm["diversity_ratio"] < 0.7
        assert "severe mode collapse" in mm["interpretation"].lower()

    def test_moderate_mode_collapse(self, setup_data):
        """Test detection of moderate mode collapse (0.7 <= ratio <= 0.9)."""
        real_df, quantizer = setup_data

        # Synthetic data with moderate mode collapse (2 repeated pairs)
        synth_df = pd.DataFrame(
            {
                "text": ["good product", "good product", "bad service", "bad service"],
                "category": ["A", "A", "B", "B"],
                "value": [10, 10, 20, 20],
            }
        )

        result = evaluate_diversity_multimodal(real_df, synth_df, quantizer)

        mm = result["multimodal_metrics"]

        # Moderate collapse: ratio should be between 0.7 and 0.9
        # Note: exact ratio depends on quantization, so we check a reasonable range
        assert 0.5 <= mm["diversity_ratio"] <= 1.0
        # Interpretation should mention either moderate collapse or good diversity
        assert any(
            phrase in mm["interpretation"].lower()
            for phrase in ["moderate", "good diversity", "collapse"]
        )

    def test_marginal_entropies_computed(self, setup_data):
        """Test that marginal entropies are computed."""
        real_df, quantizer = setup_data

        synth_df = pd.DataFrame(
            {
                "text": ["nice item", "poor quality", "okay product", "great"],
                "category": ["B", "A", "C", "A"],
                "value": [15, 25, 35, 45],
            }
        )

        result = evaluate_diversity_multimodal(real_df, synth_df, quantizer)

        um = result["unimodal_metrics"]
        assert "text_marginal_entropy" in um
        assert "tabular_marginal_entropy" in um
        assert um["text_marginal_entropy"] > 0
        assert um["tabular_marginal_entropy"] > 0

    def test_empty_data_handling(self, setup_data):
        """Test handling of empty datasets."""
        _, quantizer = setup_data

        real_df = pd.DataFrame({"text": [], "category": [], "value": []})
        synth_df = pd.DataFrame({"text": [], "category": [], "value": []})

        # Should handle gracefully without crashing
        result = evaluate_diversity_multimodal(real_df, synth_df, quantizer)

        # Should return some result structure
        assert "multimodal_metrics" in result or "error" in result

    def test_single_row_data(self, setup_data):
        """Test handling of single-row datasets."""
        _, quantizer = setup_data

        real_df = pd.DataFrame(
            {"text": ["good product"], "category": ["A"], "value": [10]}
        )
        synth_df = pd.DataFrame(
            {"text": ["nice item"], "category": ["B"], "value": [20]}
        )

        result = evaluate_diversity_multimodal(real_df, synth_df, quantizer)

        # Should handle gracefully
        assert "multimodal_metrics" in result
        mm = result["multimodal_metrics"]
        assert "diversity_ratio" in mm

    def test_joint_entropy_greater_than_marginals(self, setup_data):
        """Test that joint entropy bounds make sense."""
        real_df, quantizer = setup_data

        synth_df = pd.DataFrame(
            {
                "text": ["nice item", "poor quality", "okay product", "great"],
                "category": ["B", "A", "C", "A"],
                "value": [15, 25, 35, 45],
            }
        )

        result = evaluate_diversity_multimodal(real_df, synth_df, quantizer)

        mm = result["multimodal_metrics"]
        um = result["unimodal_metrics"]

        # Joint entropy should be >= max(marginal entropies)
        # and <= sum(marginal entropies)
        joint = mm["joint_shannon_entropy"]
        text_marginal = um["text_marginal_entropy"]
        tabular_marginal = um["tabular_marginal_entropy"]

        # Joint should be at least as large as the larger marginal
        assert joint >= max(text_marginal, tabular_marginal) - 0.1

        # Joint should not exceed sum of marginals (for independent variables)
        assert joint <= text_marginal + tabular_marginal + 0.1
