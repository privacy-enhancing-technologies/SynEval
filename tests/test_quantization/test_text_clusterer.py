import pytest
import numpy as np
import os
from evaluation.quantization.text_clusterer import TextClusterer
from tests.fixtures.test_data import get_sample_texts


def test_text_clusterer_initialization():
    """Should initialize with default parameters."""
    clusterer = TextClusterer()
    assert clusterer.model_name == "all-MiniLM-L6-v2"
    assert clusterer.n_clusters is None
    assert clusterer.kmeans is None


def test_text_clusterer_fit():
    """Should fit K-Means on text embeddings."""
    texts = get_sample_texts()
    clusterer = TextClusterer(n_clusters=3)

    clusterer.fit(texts)

    assert clusterer.kmeans is not None
    assert clusterer.kmeans.n_clusters == 3


def test_text_clusterer_transform():
    """Should transform texts to cluster assignments."""
    texts = get_sample_texts()
    clusterer = TextClusterer(n_clusters=3)
    clusterer.fit(texts)

    clusters = clusterer.transform(texts)

    assert len(clusters) == len(texts)
    assert clusters.min() >= 0
    assert clusters.max() < 3


def test_text_clusterer_semantic_clustering():
    """Similar texts should cluster together."""
    texts = get_sample_texts()
    clusterer = TextClusterer(n_clusters=3)
    clusterer.fit(texts)

    clusters = clusterer.transform(texts)

    # Luxury texts (0-2) should have same cluster
    assert clusters[0] == clusters[1] == clusters[2]

    # Budget texts (3-5) should have same cluster
    assert clusters[3] == clusters[4] == clusters[5]

    # Family texts (6-8) should have same cluster
    assert clusters[6] == clusters[7] == clusters[8]


def test_text_clusterer_get_embeddings():
    """Should return raw embeddings."""
    texts = get_sample_texts()
    clusterer = TextClusterer()

    embeddings = clusterer.get_embeddings(texts)

    assert embeddings.shape == (len(texts), 384)  # MiniLM dimension


def test_text_clusterer_embedding_cache(tmp_path):
    """Should cache embeddings to disk."""
    texts = get_sample_texts()
    cache_dir = str(tmp_path / "cache")
    clusterer = TextClusterer(cache_dir=cache_dir)

    # First call - should cache
    emb1 = clusterer.get_embeddings(texts)
    assert os.path.exists(cache_dir)

    # Second call - should load from cache
    emb2 = clusterer.get_embeddings(texts)
    np.testing.assert_array_equal(emb1, emb2)
