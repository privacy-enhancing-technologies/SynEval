"""End-to-end integration tests for Tilted generator (no heavy dependencies)."""
import pytest
import os
import tempfile
from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generators.tilted import TiltedGenerator


@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    data_path = Path(__file__).parent.parent / 'fixtures' / 'sample_data.csv'
    return pd.read_csv(data_path)


@pytest.fixture
def text_columns():
    """Return text column names."""
    return ['use']


@pytest.fixture
def tabular_columns():
    """Return tabular column names."""
    return ['sector', 'loan_amount']


class TestTiltedE2E:
    """End-to-end tests for Tilted generator."""

    def test_tilted_fit_generate(self, sample_data, text_columns, tabular_columns):
        """Test full Tilted pipeline."""
        generator = TiltedGenerator(shuffle_strategy='random', random_state=42)

        # Fit
        generator.fit(sample_data, text_columns, tabular_columns)

        # Generate
        n_samples = 20
        synthetic = generator.generate(n_samples)

        # Validate
        assert len(synthetic) == n_samples
        assert all(col in synthetic.columns for col in text_columns)
        assert all(col in synthetic.columns for col in tabular_columns)

    def test_tilted_reproducibility(self, sample_data, text_columns, tabular_columns):
        """Test that same seed gives same results."""
        generator1 = TiltedGenerator(shuffle_strategy='random', random_state=42)
        generator1.fit(sample_data, text_columns, tabular_columns)
        synthetic1 = generator1.generate(10)

        generator2 = TiltedGenerator(shuffle_strategy='random', random_state=42)
        generator2.fit(sample_data, text_columns, tabular_columns)
        synthetic2 = generator2.generate(10)

        # Should be identical
        pd.testing.assert_frame_equal(synthetic1, synthetic2)

    def test_tilted_different_strategies(self, sample_data, text_columns, tabular_columns):
        """Test different shuffle strategies."""
        strategies = ['random']  # Only test strategies that don't require target column

        for strategy in strategies:
            generator = TiltedGenerator(shuffle_strategy=strategy, random_state=42)
            generator.fit(sample_data, text_columns, tabular_columns)
            synthetic = generator.generate(10)

            assert len(synthetic) == 10
            assert all(col in synthetic.columns for col in text_columns + tabular_columns)


class TestDataQuality:
    """Test quality of generated data."""

    def test_no_null_values_tilted(self, sample_data, text_columns, tabular_columns):
        """Test that generated data has no null values."""
        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data, text_columns, tabular_columns)
        synthetic = generator.generate(20)

        # No nulls in any column
        assert not synthetic.isnull().any().any()

    def test_numeric_ranges_tilted(self, sample_data, text_columns, tabular_columns):
        """Test that numeric values are in reasonable ranges."""
        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data, text_columns, tabular_columns)
        synthetic = generator.generate(50)

        # loan_amount should be positive
        assert (synthetic['loan_amount'] > 0).all()

        # Values should be within reasonable range of original data
        original_min = sample_data['loan_amount'].min()
        original_max = sample_data['loan_amount'].max()

        # For Tilted (which shuffles), values should be from original data
        assert synthetic['loan_amount'].min() >= original_min
        assert synthetic['loan_amount'].max() <= original_max

    def test_categorical_values_tilted(self, sample_data, text_columns, tabular_columns):
        """Test that categorical values come from original data."""
        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data, text_columns, tabular_columns)
        synthetic = generator.generate(50)

        # All sectors should be from original data
        original_sectors = set(sample_data['sector'].unique())
        synthetic_sectors = set(synthetic['sector'].unique())

        assert synthetic_sectors.issubset(original_sectors)

    def test_text_field_non_empty(self, sample_data, text_columns, tabular_columns):
        """Test that text fields are not empty."""
        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data, text_columns, tabular_columns)
        synthetic = generator.generate(20)

        # Text field should not be empty
        assert (synthetic['use'].str.len() > 0).all()


class TestSaveLoad:
    """Test generator save/load functionality."""

    def test_tilted_save_load(self, sample_data, text_columns, tabular_columns):
        """Test saving and loading Tilted generator."""
        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data, text_columns, tabular_columns)

        # Generate before save
        synthetic_before = generator.generate(10)

        # Save
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            generator.save(temp_path)

            # Load into new generator
            loaded_generator = TiltedGenerator()
            loaded_generator.load(temp_path)

            # Generate after load
            synthetic_after = loaded_generator.generate(10)

            # Should have same structure
            assert list(synthetic_after.columns) == list(synthetic_before.columns)
            assert len(synthetic_after) == len(synthetic_before)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_generate_more_than_original(self, sample_data, text_columns, tabular_columns):
        """Test generating more samples than original dataset."""
        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data, text_columns, tabular_columns)

        # Generate more samples than in training data
        n_samples = len(sample_data) * 2
        synthetic = generator.generate(n_samples)

        assert len(synthetic) == n_samples

    def test_single_sample_generation(self, sample_data, text_columns, tabular_columns):
        """Test generating a single sample."""
        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data, text_columns, tabular_columns)

        synthetic = generator.generate(1)

        assert len(synthetic) == 1
        assert all(col in synthetic.columns for col in text_columns + tabular_columns)

    def test_invalid_column_name(self, sample_data):
        """Test error when column doesn't exist."""
        generator = TiltedGenerator(random_state=42)

        with pytest.raises((KeyError, ValueError)):
            generator.fit(sample_data, ['use'], ['nonexistent_column'])


class TestMultipleTextColumns:
    """Test with multiple text columns."""

    def test_multiple_text_columns(self, sample_data):
        """Test with multiple text columns."""
        # Add another text column for testing
        sample_data_copy = sample_data.copy()
        sample_data_copy['description'] = 'Test description for ' + sample_data_copy['sector']

        generator = TiltedGenerator(random_state=42)
        generator.fit(sample_data_copy, ['use', 'description'], ['sector', 'loan_amount'])

        synthetic = generator.generate(10)

        assert len(synthetic) == 10
        assert 'use' in synthetic.columns
        assert 'description' in synthetic.columns
        assert 'sector' in synthetic.columns
        assert 'loan_amount' in synthetic.columns
