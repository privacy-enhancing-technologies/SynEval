"""Tabular feature binning for discrete joint space."""
import pandas as pd
import numpy as np
from typing import List


class TabularBinner:
    """
    Bin tabular features into discrete states.

    - Continuous features: quantile binning (equal-frequency)
    - Categorical features: preserved as-is
    - Multiple features: combined into joint state identifier
    """

    def __init__(self, n_bins: int = None):
        """
        Initialize tabular binner.

        Args:
            n_bins: Number of bins for continuous features
        """
        self.n_bins = n_bins
        self.bin_edges = {}
        self.categorical_columns = []
        self.continuous_columns = []
        self.tabular_columns = []

    def fit(self, df: pd.DataFrame, tabular_columns: List[str], n_bins: int = None):
        """
        Fit binning on real data.

        Args:
            df: DataFrame with tabular features
            tabular_columns: List of column names to bin
            n_bins: Number of bins (overrides __init__ value)
        """
        if n_bins is not None:
            self.n_bins = n_bins

        if self.n_bins is None:
            raise ValueError("n_bins must be specified")

        self.tabular_columns = tabular_columns

        # Detect column types
        for col in tabular_columns:
            if df[col].dtype in ['object', 'category']:
                # String/categorical types
                self.categorical_columns.append(col)
            else:
                # Numeric types - treat as continuous for binning
                self.continuous_columns.append(col)

                # Compute quantile bin edges
                self.bin_edges[col] = np.percentile(
                    df[col].dropna(),
                    np.linspace(0, 100, self.n_bins + 1)
                )

        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Transform tabular features to discrete bins.

        Args:
            df: DataFrame with tabular features

        Returns:
            Series with binned values or joint state identifiers
        """
        binned_df = df.copy()

        # Bin continuous columns
        for col in self.continuous_columns:
            binned_df[col] = pd.cut(
                df[col],
                bins=self.bin_edges[col],
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )

        # Single column case
        if len(self.tabular_columns) == 1:
            return binned_df[self.tabular_columns[0]]

        # Multiple columns: create joint state identifier
        return binned_df[self.tabular_columns].apply(
            lambda row: '_'.join(row.astype(str)),
            axis=1
        )
