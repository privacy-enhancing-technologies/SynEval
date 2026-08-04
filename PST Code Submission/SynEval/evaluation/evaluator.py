"""
Multimodal Evaluator - Unified Interface for All Metrics

This evaluator combines all four evaluation dimensions:
1. Fidelity: Jensen-Shannon Divergence (JSD)
2. Utility: Text-to-Attribute (T2A) and Attribute-to-Text (A2T)
3. Diversity: Joint Entropy
4. Privacy: Distance to Closest Record (DCR)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from evaluation.fidelity import evaluate_fidelity_multimodal
from evaluation.utility import evaluate_utility_multimodal
from evaluation.diversity import evaluate_diversity_multimodal
from evaluation.privacy import evaluate_privacy_multimodal
from evaluation.quantization.semantic_quantizer import SemanticQuantizer


class MultimodalEvaluator:
    """
    Unified evaluator for multimodal synthetic data.

    Combines fidelity, utility, diversity, and privacy metrics into a
    single interface for comprehensive quality assessment.
    """

    def __init__(
        self,
        text_columns: List[str],
        tabular_columns: List[str],
        adaptive: bool = True,
        text_clusters: Optional[int] = None,
        tabular_bins: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize the evaluator.

        Args:
            text_columns: List of text column names
            tabular_columns: List of tabular column names
            adaptive: Use adaptive quantization (default: True)
            text_clusters: Number of text clusters (None = adaptive)
            tabular_bins: Number of tabular bins (None = adaptive)
            random_seed: Random seed for reproducibility
        """
        self.text_columns = text_columns
        self.tabular_columns = tabular_columns
        self.adaptive = adaptive
        self.text_clusters = text_clusters
        self.tabular_bins = tabular_bins
        self.random_seed = random_seed

        # State
        self.is_fitted = False
        self.real_df = None
        self.quantizer = None

    def fit(self, real_df: pd.DataFrame):
        """
        Fit the evaluator on real data.

        Args:
            real_df: Real dataset to use as reference
        """
        self.real_df = real_df.copy()

        # Initialize quantizer
        self.quantizer = SemanticQuantizer(
            text_columns=self.text_columns,
            tabular_columns=self.tabular_columns,
            adaptive=self.adaptive,
            text_clusters=self.text_clusters,
            tabular_bins=self.tabular_bins,
            random_seed=self.random_seed
        )

        # Fit quantizer on real data
        self.quantizer.fit(real_df)

        self.is_fitted = True

    def evaluate(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate synthetic data on all metrics.

        Args:
            synthetic_df: Synthetic dataset to evaluate

        Returns:
            dict: All metric values
        """
        if not self.is_fitted:
            raise RuntimeError("Evaluator not fitted. Call fit() first.")

        metrics = {}

        try:
            # 1. Fidelity: JSD over joint distribution
            fid = evaluate_fidelity_multimodal(self.real_df, synthetic_df, self.quantizer)
            metrics['jsd'] = fid.get('joint_spectral_divergence', np.nan)

        except Exception as e:
            print(f"Warning: JSD computation failed: {e}")
            metrics['jsd'] = np.nan

        try:
            # 2 & 3. Utility: T2A and A2T
            target_col = self.tabular_columns[0]
            util = evaluate_utility_multimodal(
                self.real_df, synthetic_df, self.quantizer, target_column=target_col
            )
            t2a = util.get('text_to_attribute', {})
            a2t = util.get('attribute_to_text', {})
            metrics['t2a_accuracy'] = t2a.get('f1_score', t2a.get('rmse', np.nan))
            metrics['a2t_accuracy'] = a2t.get('cluster_prediction_f1', np.nan)

        except Exception as e:
            print(f"Warning: Utility computation failed: {e}")
            metrics['t2a_accuracy'] = np.nan
            metrics['a2t_accuracy'] = np.nan

        try:
            # 4. Diversity: Joint Shannon Entropy
            div = evaluate_diversity_multimodal(self.real_df, synthetic_df, self.quantizer)
            mm = div.get('multimodal_metrics', {})
            metrics['joint_entropy'] = mm.get('joint_shannon_entropy', np.nan)

        except Exception as e:
            print(f"Warning: Entropy computation failed: {e}")
            metrics['joint_entropy'] = np.nan

        try:
            # 5. Privacy: Semantic DCR
            priv = evaluate_privacy_multimodal(self.real_df, synthetic_df, self.quantizer)
            metrics['dcr_mean'] = priv.get('semantic_dcr_mean', np.nan)
            metrics['dcr_min'] = priv.get('semantic_dcr_min', np.nan)
            metrics['dcr_5th_percentile'] = priv.get('semantic_dcr_median', np.nan)

        except Exception as e:
            print(f"Warning: DCR computation failed: {e}")
            metrics['dcr_mean'] = np.nan
            metrics['dcr_min'] = np.nan
            metrics['dcr_5th_percentile'] = np.nan

        return metrics

    def evaluate_all(self, synthetic_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Evaluate multiple synthetic datasets.

        Args:
            synthetic_datasets: Dict mapping method names to synthetic DataFrames

        Returns:
            DataFrame with evaluation results for all methods
        """
        results = []

        for method_name, synth_df in synthetic_datasets.items():
            print(f"Evaluating {method_name}...")
            metrics = self.evaluate(synth_df)
            metrics['method'] = method_name
            results.append(metrics)

        return pd.DataFrame(results)
