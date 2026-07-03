"""
Tests for BaseGenerator abstract base class.
"""
import pytest
import pandas as pd
import tempfile
import os
from abc import ABC
from generators.base import BaseGenerator


class MockGenerator(BaseGenerator):
    """Concrete implementation of BaseGenerator for testing."""

    def __init__(self):
        self.is_fitted = False
        self.text_cols = []
        self.tabular_cols = []

    def fit(self, real_df: pd.DataFrame, text_columns: list, tabular_columns: list):
        """Mock fit implementation."""
        self.is_fitted = True
        self.text_cols = text_columns
        self.tabular_cols = tabular_columns

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Mock generate implementation."""
        if not self.is_fitted:
            raise ValueError("Generator must be fitted before generating")

        # Create mock data
        data = {}
        for col in self.text_cols:
            data[col] = [f"Generated text {i}" for i in range(n_samples)]
        for col in self.tabular_cols:
            data[col] = list(range(n_samples))

        return pd.DataFrame(data)


def test_base_generator_is_abstract():
    """Test that BaseGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        BaseGenerator()

    assert "Can't instantiate abstract class" in str(exc_info.value)


def test_base_generator_interface():
    """Test that BaseGenerator defines required abstract methods."""
    # Check that BaseGenerator is an ABC
    assert issubclass(BaseGenerator, ABC)

    # Check that required methods exist
    assert hasattr(BaseGenerator, 'fit')
    assert hasattr(BaseGenerator, 'generate')
    assert hasattr(BaseGenerator, 'save')
    assert hasattr(BaseGenerator, 'load')


def test_base_generator_save_load():
    """Test save/load functionality with mock generator."""
    # Create and fit a mock generator
    gen = MockGenerator()
    real_df = pd.DataFrame({
        'text_col': ['sample text 1', 'sample text 2'],
        'num_col': [1, 2]
    })
    gen.fit(real_df, text_columns=['text_col'], tabular_columns=['num_col'])

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        temp_path = f.name

    try:
        gen.save(temp_path)

        # Verify file was created
        assert os.path.exists(temp_path)

        # Load into new generator
        loaded_gen = MockGenerator()
        loaded_gen.load(temp_path)

        # Verify state was preserved
        assert loaded_gen.is_fitted == gen.is_fitted
        assert loaded_gen.text_cols == gen.text_cols
        assert loaded_gen.tabular_cols == gen.tabular_cols

        # Verify loaded generator can generate
        synthetic_df = loaded_gen.generate(5)
        assert len(synthetic_df) == 5
        assert 'text_col' in synthetic_df.columns
        assert 'num_col' in synthetic_df.columns

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_mock_generator_fit():
    """Test that MockGenerator fit method works correctly."""
    gen = MockGenerator()
    real_df = pd.DataFrame({
        'review': ['Great product!', 'Not good'],
        'rating': [5, 2],
        'price': [29.99, 19.99]
    })

    gen.fit(real_df, text_columns=['review'], tabular_columns=['rating', 'price'])

    assert gen.is_fitted is True
    assert gen.text_cols == ['review']
    assert gen.tabular_cols == ['rating', 'price']


def test_mock_generator_generate():
    """Test that MockGenerator generate method works correctly."""
    gen = MockGenerator()
    real_df = pd.DataFrame({
        'text': ['sample'],
        'num': [1]
    })

    gen.fit(real_df, text_columns=['text'], tabular_columns=['num'])

    # Generate samples
    synthetic_df = gen.generate(3)

    assert len(synthetic_df) == 3
    assert 'text' in synthetic_df.columns
    assert 'num' in synthetic_df.columns
    assert all(synthetic_df['text'].str.startswith('Generated text'))


def test_mock_generator_generate_before_fit():
    """Test that generate raises error if called before fit."""
    gen = MockGenerator()

    with pytest.raises(ValueError) as exc_info:
        gen.generate(5)

    assert "must be fitted before generating" in str(exc_info.value)
