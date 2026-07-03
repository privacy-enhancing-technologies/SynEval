"""Unified Semantic Quantization API."""
import pandas as pd
import numpy as np
from typing import List
from .text_clusterer import TextClusterer
from .tabular_binner import TabularBinner
from .joint_space import JointSpace
from .adaptive_params import select_adaptive_params


class SemanticQuantizer:
    """
    Unified API for Semantic Quantization.

    Combines text clustering and tabular binning to project
    multimodal data into discrete joint probability space.
    """

    def __init__(self,
                 text_columns: List[str],
                 tabular_columns: List[str],
                 text_model: str = "all-MiniLM-L6-v2",
                 text_clusters: int = None,
                 tabular_bins: int = None,
                 adaptive: bool = True,
                 cache_dir: str = ".cache/embeddings/"):
        """
        Initialize semantic quantizer.

        Args:
            text_columns: List of text column names
            tabular_columns: List of tabular column names
            text_model: Sentence-BERT model name
            text_clusters: Number of text clusters (None = adaptive)
            tabular_bins: Number of tabular bins (None = adaptive)
            adaptive: Use adaptive parameter selection
            cache_dir: Directory for caching embeddings
        """
        self.text_columns = text_columns
        self.tabular_columns = tabular_columns
        self.adaptive = adaptive

        self.text_clusterer = TextClusterer(
            model_name=text_model,
            n_clusters=text_clusters,
            cache_dir=cache_dir
        )
        self.tabular_binner = TabularBinner(n_bins=tabular_bins)
        self.joint_space = JointSpace()

    def fit(self, real_df: pd.DataFrame):
        """
        Fit quantizer on real data.

        Args:
            real_df: DataFrame with text and tabular columns
        """
        n_samples = len(real_df)

        # Determine K and B
        if self.adaptive:
            K, B = select_adaptive_params(n_samples)
            if self.text_clusterer.n_clusters is None:
                self.text_clusterer.n_clusters = K
            if self.tabular_binner.n_bins is None:
                self.tabular_binner.n_bins = B
        else:
            # Use specified or default values
            if self.text_clusterer.n_clusters is None:
                self.text_clusterer.n_clusters = 10
            if self.tabular_binner.n_bins is None:
                self.tabular_binner.n_bins = 10

        # Fit components
        texts = real_df[self.text_columns[0]].tolist()
        self.text_clusterer.fit(texts)
        self.tabular_binner.fit(real_df, self.tabular_columns)

        return self

    def transform(self, df: pd.DataFrame) -> dict:
        """
        Transform dataset to quantized space.

        Args:
            df: DataFrame with text and tabular columns

        Returns:
            Dict with text_clusters, tabular_bins, and original df
        """
        texts = df[self.text_columns[0]].tolist()
        text_clusters = self.text_clusterer.transform(texts)
        tabular_bins = self.tabular_binner.transform(df)

        return {
            "text_clusters": text_clusters,
            "tabular_bins": tabular_bins,
            "df": df
        }

    def get_joint_distribution(self, quantized_data: dict) -> np.ndarray:
        """
        Build contingency table and compute joint probabilities.

        Args:
            quantized_data: Output from transform()

        Returns:
            Joint probability matrix P(text, tabular)
        """
        self.joint_space.build_contingency_table(
            quantized_data["text_clusters"],
            quantized_data["tabular_bins"]
        )
        return self.joint_space.compute_joint_probabilities()
