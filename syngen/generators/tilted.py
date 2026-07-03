"""
Tilted Data Generator - Adversarial Baseline

This generator intentionally destroys cross-modal correlations by randomly shuffling
the pairing between text and tabular data. It serves as a "worst-case" benchmark to
validate that evaluation metrics can detect broken correlations.

Purpose:
    - Negative control for experiments
    - Validates that metrics can detect broken correlations
    - Establishes lower bound for quality assessment

Why It's Useful:
    - Proves evaluation metrics are sensitive to correlation quality
    - Shows what happens when text-tabular alignment is destroyed
    - Provides baseline that proper generators must outperform

When to Use:
    - As adversarial baseline in comparative experiments
    - To validate new evaluation metrics
    - To demonstrate the importance of cross-modal correlations

Example Results:
    Real: {sector: "Agriculture", use: "buy a dairy cow"} ✅ Correlated
    Tilted: {sector: "Agriculture", use: "repair motorcycle"} ❌ Broken

    Real: {specialty: "Cardiology", transcription: "...heart...cardiac..."} ✅ Correlated
    Tilted: {specialty: "Cardiology", transcription: "...prostate...bladder..."} ❌ Broken
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from generators.base import BaseGenerator


class TiltedGenerator(BaseGenerator):
    """
    Adversarial baseline generator that destroys cross-modal correlations.

    This generator keeps both text and tabular samples from real data, but
    randomly shuffles their pairing to intentionally break correlations.
    No training is required - it simply permutes the associations.

    Shuffling Strategies:
        - "random": Completely random shuffle (default)
        - "stratified": Shuffle within class labels (partially preserve distribution)
        - "adversarial": Intentionally pair opposites (e.g., positive text with negative labels)
    """

    def __init__(self,
                 shuffle_strategy: str = "random",
                 random_state: int = 42):
        """
        Initialize the Tilted generator.

        Args:
            shuffle_strategy: Strategy for shuffling:
                - "random": Completely random shuffle
                - "stratified": Shuffle within class labels
                - "adversarial": Pair opposites (requires target column)
            random_state: Random seed for reproducibility
        """
        self.shuffle_strategy = shuffle_strategy
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # These will be set during fit()
        self.real_df = None
        self.text_columns = None
        self.tabular_columns = None
        self.fitted = False

        # Validate strategy
        valid_strategies = ["random", "stratified", "adversarial"]
        if shuffle_strategy not in valid_strategies:
            raise ValueError(
                f"shuffle_strategy must be one of {valid_strategies}, "
                f"got '{shuffle_strategy}'"
            )

    def fit(self,
            real_df: pd.DataFrame,
            text_columns: List[str],
            tabular_columns: List[str],
            target_column: Optional[str] = None):
        """
        Fit generator on real data (just stores it, no actual training).

        Args:
            real_df: Real dataset to sample from
            text_columns: List of text column names
            tabular_columns: List of tabular column names
            target_column: Optional target column for stratified/adversarial shuffling

        Returns:
            Self for method chaining
        """
        # Validate inputs
        if not isinstance(real_df, pd.DataFrame):
            raise TypeError("real_df must be a pandas DataFrame")

        if len(real_df) == 0:
            raise ValueError("real_df cannot be empty")

        # Check all columns exist
        all_columns = text_columns + tabular_columns
        missing = set(all_columns) - set(real_df.columns)
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        # Check text columns have string data
        for col in text_columns:
            if not pd.api.types.is_string_dtype(real_df[col]) and not pd.api.types.is_object_dtype(real_df[col]):
                raise ValueError(f"Text column '{col}' must contain string data")

        # Store the data
        self.real_df = real_df.copy()
        self.text_columns = text_columns
        self.tabular_columns = tabular_columns
        self.target_column = target_column

        # Validate target column for stratified/adversarial strategies
        if self.shuffle_strategy in ["stratified", "adversarial"]:
            if target_column is None:
                raise ValueError(
                    f"shuffle_strategy '{self.shuffle_strategy}' requires target_column"
                )
            if target_column not in real_df.columns:
                raise ValueError(f"target_column '{target_column}' not found in DataFrame")

        self.fitted = True
        return self

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data by shuffling text-tabular pairings.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with shuffled pairings (broken correlations)
        """
        if not self.fitted:
            raise RuntimeError("Generator must be fitted before generating data")

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        # Sample with replacement to allow any n_samples
        sample_indices = self.rng.choice(
            len(self.real_df),
            size=n_samples,
            replace=True
        )

        # Get text and tabular data
        text_data = self.real_df.iloc[sample_indices][self.text_columns].reset_index(drop=True)
        tabular_data = self.real_df.iloc[sample_indices][self.tabular_columns].reset_index(drop=True)

        # Also get target column if needed
        target_data = None
        if self.target_column is not None:
            target_data = self.real_df.iloc[sample_indices][[self.target_column]].reset_index(drop=True)

        # Apply shuffling strategy
        if self.shuffle_strategy == "random":
            # Completely random shuffle
            shuffle_indices = self.rng.permutation(n_samples)
            tabular_data = tabular_data.iloc[shuffle_indices].reset_index(drop=True)
            if target_data is not None:
                target_data = target_data.iloc[shuffle_indices].reset_index(drop=True)

        elif self.shuffle_strategy == "stratified":
            # Shuffle within each class to partially preserve distribution
            target_values = self.real_df.iloc[sample_indices][self.target_column].reset_index(drop=True)

            # Shuffle within each class
            new_order = []
            for label in target_values.unique():
                label_mask = target_values == label
                label_indices = np.where(label_mask)[0]
                shuffled = self.rng.permutation(label_indices)
                new_order.extend(shuffled)

            tabular_data = tabular_data.iloc[new_order].reset_index(drop=True)
            if target_data is not None:
                target_data = target_data.iloc[new_order].reset_index(drop=True)

        elif self.shuffle_strategy == "adversarial":
            # Pair opposites (e.g., positive text with negative labels)
            target_values = self.real_df.iloc[sample_indices][self.target_column].reset_index(drop=True)
            unique_labels = sorted(target_values.unique())

            if len(unique_labels) == 2:
                # Binary: swap labels
                new_order = []
                for label in unique_labels:
                    label_indices = np.where(target_values == label)[0]
                    opposite_label = [l for l in unique_labels if l != label][0]
                    opposite_indices = np.where(target_values == opposite_label)[0]

                    # Match each sample from one class with random from opposite
                    for idx in label_indices:
                        if len(opposite_indices) > 0:
                            match_idx = self.rng.choice(opposite_indices)
                            new_order.append(match_idx)
                        else:
                            # Fallback to random if no opposites available
                            new_order.append(self.rng.choice(n_samples))

                tabular_data = tabular_data.iloc[new_order].reset_index(drop=True)
                if target_data is not None:
                    target_data = target_data.iloc[new_order].reset_index(drop=True)
            else:
                # Multi-class: use reverse order or random
                # (adversarial strategy is less defined for multi-class)
                reverse_map = dict(zip(unique_labels, reversed(unique_labels)))
                new_order = []
                for label in target_values:
                    target_label = reverse_map[label]
                    target_indices = np.where(target_values == target_label)[0]
                    if len(target_indices) > 0:
                        new_order.append(self.rng.choice(target_indices))
                    else:
                        new_order.append(self.rng.choice(n_samples))

                tabular_data = tabular_data.iloc[new_order].reset_index(drop=True)
                if target_data is not None:
                    target_data = target_data.iloc[new_order].reset_index(drop=True)

        # Combine text, shuffled tabular data, and target (if present)
        if target_data is not None:
            synthetic_df = pd.concat([text_data, tabular_data, target_data], axis=1)
        else:
            synthetic_df = pd.concat([text_data, tabular_data], axis=1)

        return synthetic_df

    def __repr__(self):
        """String representation."""
        return (
            f"TiltedGenerator(shuffle_strategy='{self.shuffle_strategy}', "
            f"random_state={self.random_state}, fitted={self.fitted})"
        )
