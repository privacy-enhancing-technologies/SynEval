"""
Simple metrics validation tests for SynEval framework
Addresses the feedback about verifying metric outputs on small synthetic datasets
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diversity import DiversityEvaluator
from fidelity import FidelityEvaluator
from utility import UtilityEvaluator


class TestMetricsValidation:
    """Test actual metric outputs on small synthetic datasets"""

    def setup_method(self):
        """Setup test data for each test method"""
        # Create small synthetic dataset for testing
        self.original_data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "rating": [4, 5, 3, 4, 5, 2, 3, 4, 5, 1],
                "price": [10.5, 15.2, 8.7, 12.3, 9.8, 7.5, 11.2, 13.8, 16.1, 6.9],
                "text": [
                    "Great product",
                    "Excellent quality",
                    "Good value",
                    "Amazing",
                    "Perfect",
                    "Not bad",
                    "Okay",
                    "Good",
                    "Excellent",
                    "Poor",
                ],
            }
        )

        # Create slightly different synthetic data
        self.synthetic_data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "rating": [4, 5, 3, 4, 5, 2, 3, 4, 5, 1],
                "price": [10.2, 15.5, 8.9, 12.1, 9.6, 7.8, 11.0, 13.5, 15.8, 7.2],
                "text": [
                    "Great product",
                    "Excellent quality",
                    "Good value",
                    "Amazing",
                    "Perfect",
                    "Not bad",
                    "Okay",
                    "Good",
                    "Excellent",
                    "Poor",
                ],
            }
        )

        self.metadata = {
            "columns": {
                "id": {"sdtype": "numerical", "is_primary_key": True},
                "rating": {"sdtype": "categorical", "values": [1, 2, 3, 4, 5]},
                "price": {"sdtype": "numerical"},
                "text": {"sdtype": "text"},
            },
            "text_columns": ["text"],
        }

    def test_fidelity_metrics_output(self):
        """Test that FidelityEvaluator produces expected metric outputs"""
        evaluator = FidelityEvaluator(
            self.synthetic_data, self.original_data, self.metadata
        )
        result = evaluator.evaluate()

        # Verify result structure
        assert isinstance(result, dict)
        assert "diagnostic" in result
        assert "quality" in result
        assert "text" in result
        assert "numerical_statistics" in result

        # Verify diagnostic metrics
        diagnostic = result["diagnostic"]
        assert "Data Validity" in diagnostic
        assert "Data Structure" in diagnostic
        assert "Overall" in diagnostic
        assert isinstance(diagnostic["Data Validity"], (int, float))
        assert isinstance(diagnostic["Data Structure"], (int, float))
        assert isinstance(diagnostic["Overall"]["score"], (int, float))

        # Verify quality metrics
        quality = result["quality"]
        assert "Column Shapes" in quality
        assert "Column Pair Trends" in quality
        assert "Overall" in quality
        assert isinstance(quality["Column Shapes"], (int, float))
        assert isinstance(quality["Column Pair Trends"], (int, float))
        assert isinstance(quality["Overall"]["score"], (int, float))

        # Verify text analysis metrics
        text_analysis = result["text"]
        assert "text" in text_analysis
        text_metrics = text_analysis["text"]
        assert "length_stats" in text_metrics
        assert "word_count_stats" in text_metrics
        assert "keyword_analysis" in text_metrics
        assert "sentiment_analysis" in text_metrics

        # Verify numerical statistics
        numerical_stats = result["numerical_statistics"]
        assert "id" in numerical_stats
        assert "price" in numerical_stats

        print("✅ FidelityEvaluator produces comprehensive metric outputs")

    def test_utility_metrics_output(self):
        """Test that UtilityEvaluator produces expected metric outputs"""
        classification_metadata = self.metadata.copy()
        classification_metadata["utility"] = {
            "input_columns": ["text"],
            "output_columns": ["rating"],
            "task_type": "classification",
        }

        evaluator = UtilityEvaluator(
            self.synthetic_data,
            self.original_data,
            classification_metadata,
            input_columns=["text"],
            output_columns=["rating"],
            task_type="classification",
        )

        result = evaluator.evaluate()

        # Verify result structure (based on actual API)
        assert isinstance(result, dict)

        # Check if we have the expected keys (adapt to actual output)
        if "tstr_accuracy" in result:
            tstr_result = result["tstr_accuracy"]
            assert "input_columns" in tstr_result
            assert "output_columns" in tstr_result
            assert "task_type" in tstr_result
            assert tstr_result["task_type"] == "classification"
        else:
            # Fallback to expected structure
            assert "task_type" in result
            assert "performance_metrics" in result

        print("✅ UtilityEvaluator produces metric outputs")

    def test_diversity_metrics_output(self):
        """Test that DiversityEvaluator produces expected metric outputs"""
        evaluator = DiversityEvaluator(
            self.synthetic_data, self.original_data, self.metadata
        )
        result = evaluator.evaluate()

        # Verify result structure (based on actual API)
        assert isinstance(result, dict)
        assert "tabular_diversity" in result
        assert "text_diversity" in result

        # Verify tabular diversity metrics (based on actual output)
        tabular_diversity = result["tabular_diversity"]
        assert "coverage" in tabular_diversity
        assert "categorical_metrics" in tabular_diversity
        assert "numerical_metrics" in tabular_diversity
        assert "entropy_metrics" in tabular_diversity

        # Verify text diversity metrics
        text_diversity = result["text_diversity"]
        # Handle case where flair is not available (text_diversity will be skipped)
        if "skipped" in text_diversity:
            # If flair is not available, text_diversity will be skipped
            assert text_diversity.get("skipped") is True
            assert "reason" in text_diversity
            print("⚠️ Text diversity skipped (flair not available)")
        else:
            # If flair is available, should have real and synthetic keys
            assert "real" in text_diversity
            assert "synthetic" in text_diversity

        print("✅ DiversityEvaluator produces comprehensive metric outputs")

    def test_metrics_consistency(self):
        """Test that metrics are consistent across multiple runs"""
        evaluator = FidelityEvaluator(
            self.synthetic_data, self.original_data, self.metadata
        )

        # Run evaluation multiple times
        result1 = evaluator.evaluate()
        result2 = evaluator.evaluate()

        # Results should be identical (deterministic)
        assert result1 == result2, "Fidelity evaluation should be deterministic"

        print("✅ Metrics are consistent across multiple runs")

    def test_metric_value_ranges(self):
        """Test that metric values are within expected ranges"""
        evaluator = FidelityEvaluator(
            self.synthetic_data, self.original_data, self.metadata
        )
        result = evaluator.evaluate()

        # Check diagnostic scores (0-1 range)
        diagnostic = result["diagnostic"]
        for metric_name, score in diagnostic.items():
            if metric_name != "Overall":
                assert 0 <= score <= 1, f"{metric_name} should be between 0 and 1"

        # Check quality scores (0-1 range)
        quality = result["quality"]
        for metric_name, score in quality.items():
            if metric_name != "Overall":
                assert 0 <= score <= 1, f"{metric_name} should be between 0 and 1"

        print("✅ Metric values are within expected ranges")

    def test_performance_benchmarks(self):
        """Test that evaluation completes within reasonable time"""
        import time

        start_time = time.time()
        evaluator = FidelityEvaluator(
            self.synthetic_data, self.original_data, self.metadata
        )
        result = evaluator.evaluate()
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 60, f"Evaluation took too long: {execution_time:.2f}s"

        print(f"✅ Evaluation completed in {execution_time:.2f}s")
