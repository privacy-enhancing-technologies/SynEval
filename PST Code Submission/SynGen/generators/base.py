"""
Base generator interface for synthetic data generation.

This module defines the abstract base class that all synthetic data generators
must implement, ensuring a consistent interface across different generation methods.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
import pickle


class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators."""

    @abstractmethod
    def fit(self, real_df: pd.DataFrame, text_columns: List[str], tabular_columns: List[str]):
        """
        Fit generator on real data.

        Args:
            real_df: Real dataset to learn from
            text_columns: List of text column names
            tabular_columns: List of tabular column names
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate n_samples synthetic records.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data containing both text and tabular columns
        """
        pass

    def save(self, path: str):
        """
        Save trained generator to file.

        Args:
            path: File path to save the generator (typically .pkl extension)
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """
        Load trained generator from file.

        Args:
            path: File path to load the generator from

        Returns:
            Self with loaded state
        """
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
        return self
