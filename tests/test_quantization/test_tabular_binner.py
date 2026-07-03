import pytest
import pandas as pd
import numpy as np
from evaluation.quantization.tabular_binner import TabularBinner
from tests.fixtures.test_data import get_sample_tabular_data


def test_tabular_binner_initialization():
    """Should initialize with default parameters."""
    binner = TabularBinner()
    assert binner.n_bins is None
    assert binner.bin_edges == {}
    assert binner.categorical_columns == []
    assert binner.continuous_columns == []


def test_tabular_binner_fit_detects_column_types():
    """Should auto-detect continuous vs categorical columns."""
    df = get_sample_tabular_data()
    binner = TabularBinner(n_bins=3)

    binner.fit(df, ['price', 'accommodates', 'property_type'])

    assert 'price' in binner.continuous_columns
    assert 'accommodates' in binner.continuous_columns
    assert 'property_type' in binner.categorical_columns


def test_tabular_binner_fit_creates_bin_edges():
    """Should create quantile bin edges for continuous columns."""
    df = get_sample_tabular_data()
    binner = TabularBinner(n_bins=3)

    binner.fit(df, ['price', 'accommodates'])

    assert 'price' in binner.bin_edges
    assert 'accommodates' in binner.bin_edges
    # n_bins=3 creates 4 edges (3 bins)
    assert len(binner.bin_edges['price']) == 4


def test_tabular_binner_transform_single_column():
    """Should bin single continuous column."""
    df = get_sample_tabular_data()
    binner = TabularBinner(n_bins=3)
    binner.fit(df, ['price'])

    binned = binner.transform(df)

    assert len(binned) == len(df)
    assert binned.dtype in [np.int64, np.float64, object]
    # Should have 3 unique bins (0, 1, 2)
    assert binned.nunique() == 3


def test_tabular_binner_transform_preserves_categorical():
    """Should preserve categorical columns as-is."""
    df = get_sample_tabular_data()
    binner = TabularBinner(n_bins=3)
    binner.fit(df, ['property_type'])

    binned = binner.transform(df)

    # Should have same categories as original
    assert set(binned.unique()) == set(df['property_type'].unique())


def test_tabular_binner_transform_multicolumn_creates_joint_state():
    """Should combine multiple columns into joint state identifier."""
    df = get_sample_tabular_data()
    binner = TabularBinner(n_bins=3)
    binner.fit(df, ['price', 'property_type'])

    binned = binner.transform(df)

    # Should create string identifiers like "0_Apartment"
    assert binned.dtype == object
    assert '_' in binned.iloc[0]


def test_tabular_binner_bins_correlate_with_data():
    """Binning should group similar values together."""
    df = get_sample_tabular_data()
    binner = TabularBinner(n_bins=3)
    binner.fit(df, ['price'])

    binned = binner.transform(df)

    # High prices (450-500) should get same bin
    high_price_bins = binned[df['price'] > 400]
    assert high_price_bins.nunique() == 1

    # Low prices (50-60) should get same bin
    low_price_bins = binned[df['price'] < 100]
    assert low_price_bins.nunique() == 1
