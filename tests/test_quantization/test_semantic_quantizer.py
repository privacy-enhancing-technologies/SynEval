"""Tests for SemanticQuantizer unified API."""
import pytest
import pandas as pd
from evaluation.quantization.semantic_quantizer import SemanticQuantizer
from tests.fixtures.test_data import get_sample_multimodal_df


def test_semantic_quantizer_initialization():
    """Should initialize with specified parameters."""
    quantizer = SemanticQuantizer(
        text_columns=['description'],
        tabular_columns=['price', 'property_type'],
        text_model='all-MiniLM-L6-v2',
        text_clusters=5,
        tabular_bins=3
    )

    assert quantizer.text_columns == ['description']
    assert quantizer.tabular_columns == ['price', 'property_type']
    assert quantizer.text_clusterer.model_name == 'all-MiniLM-L6-v2'


def test_semantic_quantizer_fit():
    """Should fit all components on real data."""
    df = get_sample_multimodal_df()
    quantizer = SemanticQuantizer(
        text_columns=['description'],
        tabular_columns=['price'],
        text_clusters=3,
        tabular_bins=3
    )

    quantizer.fit(df)

    assert quantizer.text_clusterer.kmeans is not None
    assert 'price' in quantizer.tabular_binner.bin_edges


def test_semantic_quantizer_transform():
    """Should transform dataset to quantized space."""
    df = get_sample_multimodal_df()
    quantizer = SemanticQuantizer(
        text_columns=['description'],
        tabular_columns=['price'],
        text_clusters=3,
        tabular_bins=3
    )
    quantizer.fit(df)

    quantized = quantizer.transform(df)

    assert 'text_clusters' in quantized
    assert 'tabular_bins' in quantized
    assert 'df' in quantized
    assert len(quantized['text_clusters']) == len(df)


def test_semantic_quantizer_get_joint_distribution():
    """Should compute joint probability distribution."""
    df = get_sample_multimodal_df()
    quantizer = SemanticQuantizer(
        text_columns=['description'],
        tabular_columns=['price'],
        text_clusters=3,
        tabular_bins=3
    )
    quantizer.fit(df)
    quantized = quantizer.transform(df)

    joint_prob = quantizer.get_joint_distribution(quantized)

    assert joint_prob.shape == (3, 3)
    assert abs(joint_prob.sum() - 1.0) < 1e-6


def test_semantic_quantizer_adaptive_params():
    """Should use adaptive parameters when enabled."""
    df = get_sample_multimodal_df()
    quantizer = SemanticQuantizer(
        text_columns=['description'],
        tabular_columns=['price'],
        adaptive=True
    )

    quantizer.fit(df)

    # Small dataset (9 samples) should get small K and B
    assert quantizer.text_clusterer.n_clusters <= 5
    assert quantizer.tabular_binner.n_bins <= 5


def test_semantic_quantizer_preserves_correlation():
    """Quantization should preserve text-tabular correlation."""
    df = get_sample_multimodal_df()
    quantizer = SemanticQuantizer(
        text_columns=['description'],
        tabular_columns=['price'],
        text_clusters=3,
        tabular_bins=3
    )
    quantizer.fit(df)
    quantized = quantizer.transform(df)

    # Build joint space and check MI
    quantizer.joint_space.build_contingency_table(
        quantized['text_clusters'],
        quantized['tabular_bins']
    )
    mi = quantizer.joint_space.get_mutual_information()

    # Should have non-zero MI (text and price are correlated)
    assert mi > 0.5
