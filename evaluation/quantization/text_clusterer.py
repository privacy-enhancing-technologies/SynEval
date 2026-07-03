"""Text clustering using Sentence-BERT embeddings and K-Means."""
import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List


class TextClusterer:
    """
    Cluster text into discrete semantic groups.

    Uses Sentence-BERT to encode texts to dense embeddings,
    then K-Means to partition into K clusters.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 n_clusters: int = None,
                 cache_dir: str = ".cache/embeddings/"):
        """
        Initialize text clusterer.

        Args:
            model_name: Sentence-BERT model name
            n_clusters: Number of clusters (None = will be set during fit)
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.cache_dir = cache_dir
        self.model = None
        self.kmeans = None

        # Create cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, texts: List[str]) -> str:
        """Generate cache file path for texts."""
        text_hash = hashlib.md5("".join(texts).encode()).hexdigest()
        cache_key = f"{self.model_name}_{text_hash}"
        return os.path.join(self.cache_dir, f"{cache_key}.npy")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for texts (with caching).

        Args:
            texts: List of text strings

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        # Check cache
        if self.cache_dir:
            cache_path = self._get_cache_path(texts)
            if os.path.exists(cache_path):
                return np.load(cache_path)

        # Compute embeddings
        if self.model is None:
            # Use local_files_only=True to avoid network access (eBay proxy issues)
            self.model = SentenceTransformer(self.model_name, local_files_only=True)

        embeddings = self.model.encode(texts, show_progress_bar=False)

        # Save to cache
        if self.cache_dir:
            np.save(cache_path, embeddings)

        return embeddings

    def fit(self, texts: List[str], n_clusters: int = None):
        """
        Fit K-Means on text embeddings.

        Args:
            texts: List of text strings (real data only)
            n_clusters: Number of clusters (overrides __init__ value)
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        if self.n_clusters is None:
            raise ValueError("n_clusters must be specified")

        # Get embeddings
        embeddings = self.get_embeddings(texts)

        # Fit K-Means with better initialization
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(embeddings)

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to cluster assignments.

        Args:
            texts: List of text strings

        Returns:
            Cluster assignments array of shape (n_texts,)
        """
        if self.kmeans is None:
            raise ValueError("Must call fit() before transform()")

        embeddings = self.get_embeddings(texts)
        return self.kmeans.predict(embeddings)
