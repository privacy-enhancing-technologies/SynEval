"""Joint probability space construction from quantized data."""
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


class JointSpace:
    """
    Construct joint probability space from text clusters and tabular bins.

    Builds 2D contingency table representing P(text_cluster, tabular_bin).
    """

    def __init__(self):
        """Initialize joint space constructor."""
        self.contingency_table = None
        self.joint_prob = None
        self.marginal_text = None
        self.marginal_tabular = None

    def build_contingency_table(self, text_clusters: np.ndarray,
                                tabular_bins: np.ndarray) -> np.ndarray:
        """
        Construct 2D contingency table.

        Args:
            text_clusters: Text cluster assignments (n_samples,)
            tabular_bins: Tabular bin assignments (n_samples,)

        Returns:
            Contingency table (n_text_clusters, n_tabular_bins)
        """
        self.contingency_table = pd.crosstab(text_clusters, tabular_bins).values
        return self.contingency_table

    def compute_joint_probabilities(self) -> np.ndarray:
        """
        Normalize contingency table to joint probability distribution.

        Returns:
            Joint probability matrix P(text, tabular)
        """
        if self.contingency_table is None:
            raise ValueError("Must call build_contingency_table() first")

        # Normalize to probabilities
        self.joint_prob = self.contingency_table / self.contingency_table.sum()

        # Compute marginals
        self.marginal_text = self.joint_prob.sum(axis=1)
        self.marginal_tabular = self.joint_prob.sum(axis=0)

        return self.joint_prob

    def get_mutual_information(self) -> float:
        """
        Compute mutual information I(Text; Tabular).

        Returns:
            Mutual information in bits
        """
        if self.contingency_table is None:
            raise ValueError("Must call build_contingency_table() first")

        return mutual_info_score(None, None, contingency=self.contingency_table)
