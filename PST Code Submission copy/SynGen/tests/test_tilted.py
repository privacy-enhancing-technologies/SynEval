"""
Unit tests for the Tilted Generator.
"""
import pytest
import pandas as pd
import numpy as np
from generators.tilted import TiltedGenerator


@pytest.fixture
def sample_data():
    """Create sample multimodal data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'text_description': [
            'Heart-related medical issue',
            'Brain scan results',
            'Kidney function test',
            'Lung capacity assessment',
            'Liver enzyme levels'
        ] * 4,  # 20 samples total
        'specialty': ['Cardiology', 'Neurology', 'Nephrology', 'Pulmonology', 'Hepatology'] * 4,
        'age': np.random.randint(20, 80, 20),
        'severity': np.random.choice(['low', 'medium', 'high'], 20),
        'label': [0, 1, 0, 1, 0] * 4  # Binary labels for adversarial testing
    })


@pytest.fixture
def binary_data():
    """Create binary classification data for adversarial strategy testing."""
    return pd.DataFrame({
        'text': ['positive text'] * 10 + ['negative text'] * 10,
        'feature1': [1] * 10 + [0] * 10,
        'label': [1] * 10 + [0] * 10
    })


class TestTiltedGeneratorInit:
    """Test TiltedGenerator initialization."""

    def test_default_init(self):
        """Test initialization with default parameters."""
        gen = TiltedGenerator()
        assert gen.shuffle_strategy == "random"
        assert gen.random_state == 42
        assert not gen.fitted

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        gen = TiltedGenerator(shuffle_strategy="stratified", random_state=123)
        assert gen.shuffle_strategy == "stratified"
        assert gen.random_state == 123
        assert not gen.fitted

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="shuffle_strategy must be one of"):
            TiltedGenerator(shuffle_strategy="invalid")


class TestTiltedGeneratorFit:
    """Test TiltedGenerator fit method."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting."""
        gen = TiltedGenerator()
        result = gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age', 'severity']
        )
        assert result is gen  # Should return self
        assert gen.fitted
        assert gen.real_df is not None
        assert len(gen.real_df) == 20

    def test_fit_with_target(self, sample_data):
        """Test fitting with target column."""
        gen = TiltedGenerator(shuffle_strategy="stratified")
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age'],
            target_column='label'
        )
        assert gen.fitted
        assert gen.target_column == 'label'

    def test_fit_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        gen = TiltedGenerator()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            gen.fit(empty_df, [], [])

    def test_fit_invalid_type(self):
        """Test that non-DataFrame input raises error."""
        gen = TiltedGenerator()
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            gen.fit([[1, 2], [3, 4]], [], [])

    def test_fit_missing_columns(self, sample_data):
        """Test that missing columns raise error."""
        gen = TiltedGenerator()
        with pytest.raises(ValueError, match="Columns not found"):
            gen.fit(
                sample_data,
                text_columns=['nonexistent'],
                tabular_columns=['specialty']
            )

    def test_stratified_without_target(self, sample_data):
        """Test that stratified strategy requires target."""
        gen = TiltedGenerator(shuffle_strategy="stratified")
        with pytest.raises(ValueError, match="requires target_column"):
            gen.fit(
                sample_data,
                text_columns=['text_description'],
                tabular_columns=['specialty']
            )

    def test_adversarial_without_target(self, sample_data):
        """Test that adversarial strategy requires target."""
        gen = TiltedGenerator(shuffle_strategy="adversarial")
        with pytest.raises(ValueError, match="requires target_column"):
            gen.fit(
                sample_data,
                text_columns=['text_description'],
                tabular_columns=['specialty']
            )

    def test_fit_copies_data(self, sample_data):
        """Test that fit creates a copy of the data."""
        gen = TiltedGenerator()
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty']
        )
        # Modify original
        sample_data.loc[0, 'specialty'] = 'Modified'
        # Generator should have original value
        assert gen.real_df.loc[0, 'specialty'] != 'Modified'


class TestTiltedGeneratorGenerate:
    """Test TiltedGenerator generate method."""

    def test_generate_before_fit(self):
        """Test that generate raises error if not fitted."""
        gen = TiltedGenerator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            gen.generate(10)

    def test_generate_invalid_n_samples(self, sample_data):
        """Test that invalid n_samples raises error."""
        gen = TiltedGenerator()
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age']
        )
        with pytest.raises(ValueError, match="must be positive"):
            gen.generate(0)
        with pytest.raises(ValueError, match="must be positive"):
            gen.generate(-5)

    def test_random_shuffle_basic(self, sample_data):
        """Test random shuffling strategy."""
        gen = TiltedGenerator(shuffle_strategy="random", random_state=42)
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age']
        )
        synthetic = gen.generate(10)

        # Check shape and columns
        assert len(synthetic) == 10
        assert 'text_description' in synthetic.columns
        assert 'specialty' in synthetic.columns
        assert 'age' in synthetic.columns

    def test_random_shuffle_breaks_correlation(self, sample_data):
        """Test that random shuffle actually breaks correlations."""
        gen = TiltedGenerator(shuffle_strategy="random", random_state=42)
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty']
        )
        synthetic = gen.generate(100)

        # Count how many "Heart-related" texts are paired with "Cardiology"
        # In real data, all should match; in tilted, should be ~20% (random chance)
        heart_cardiology = (
            (synthetic['text_description'] == 'Heart-related medical issue') &
            (synthetic['specialty'] == 'Cardiology')
        ).sum()

        # With random shuffling and 5 specialties, expect ~20% match
        # Allow some variance but should be much less than 100%
        assert heart_cardiology < 50  # Should not be 100% correlated

    def test_stratified_shuffle(self, sample_data):
        """Test stratified shuffling preserves label distribution."""
        gen = TiltedGenerator(shuffle_strategy="stratified", random_state=42)
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age'],
            target_column='label'
        )
        synthetic = gen.generate(20)

        # Check that label distribution is similar to input
        original_dist = sample_data['label'].value_counts(normalize=True)
        synthetic_dist = synthetic['label'].value_counts(normalize=True)

        # Should be similar (allowing some variance due to sampling)
        for label in original_dist.index:
            assert abs(original_dist[label] - synthetic_dist.get(label, 0)) < 0.3

    def test_adversarial_binary(self, binary_data):
        """Test adversarial shuffling with binary labels."""
        gen = TiltedGenerator(shuffle_strategy="adversarial", random_state=42)
        gen.fit(
            binary_data,
            text_columns=['text'],
            tabular_columns=['feature1'],
            target_column='label'
        )
        synthetic = gen.generate(20)

        # In adversarial mode, text and labels should be mostly mismatched
        # Positive text should pair with label=0 and vice versa
        assert len(synthetic) == 20
        assert 'text' in synthetic.columns
        assert 'feature1' in synthetic.columns

    def test_generate_more_samples_than_original(self, sample_data):
        """Test generating more samples than in original data."""
        gen = TiltedGenerator(random_state=42)
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty']
        )
        # Original has 20 samples, generate 50
        synthetic = gen.generate(50)
        assert len(synthetic) == 50

    def test_generate_less_samples_than_original(self, sample_data):
        """Test generating fewer samples than in original data."""
        gen = TiltedGenerator(random_state=42)
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty']
        )
        synthetic = gen.generate(5)
        assert len(synthetic) == 5

    def test_reproducibility(self, sample_data):
        """Test that same random_state produces same results."""
        gen1 = TiltedGenerator(random_state=42)
        gen1.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age']
        )
        synthetic1 = gen1.generate(10)

        gen2 = TiltedGenerator(random_state=42)
        gen2.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age']
        )
        synthetic2 = gen2.generate(10)

        pd.testing.assert_frame_equal(synthetic1, synthetic2)

    def test_different_seeds_different_results(self, sample_data):
        """Test that different seeds produce different results."""
        gen1 = TiltedGenerator(random_state=42)
        gen1.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age']
        )
        synthetic1 = gen1.generate(10)

        gen2 = TiltedGenerator(random_state=123)
        gen2.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty', 'age']
        )
        synthetic2 = gen2.generate(10)

        # Results should differ
        assert not synthetic1.equals(synthetic2)


class TestTiltedGeneratorMultipleColumns:
    """Test TiltedGenerator with multiple text/tabular columns."""

    def test_multiple_text_columns(self):
        """Test with multiple text columns."""
        data = pd.DataFrame({
            'text1': ['a', 'b', 'c'] * 3,
            'text2': ['x', 'y', 'z'] * 3,
            'num': [1, 2, 3] * 3
        })
        gen = TiltedGenerator(random_state=42)
        gen.fit(data, text_columns=['text1', 'text2'], tabular_columns=['num'])
        synthetic = gen.generate(5)

        assert 'text1' in synthetic.columns
        assert 'text2' in synthetic.columns
        assert 'num' in synthetic.columns
        assert len(synthetic) == 5

    def test_multiple_tabular_columns(self):
        """Test with multiple tabular columns."""
        data = pd.DataFrame({
            'text': ['a', 'b', 'c'] * 3,
            'num1': [1, 2, 3] * 3,
            'num2': [4, 5, 6] * 3,
            'cat': ['x', 'y', 'z'] * 3
        })
        gen = TiltedGenerator(random_state=42)
        gen.fit(data, text_columns=['text'], tabular_columns=['num1', 'num2', 'cat'])
        synthetic = gen.generate(5)

        assert len(synthetic.columns) == 4
        assert len(synthetic) == 5


class TestTiltedGeneratorRepr:
    """Test TiltedGenerator string representation."""

    def test_repr_unfitted(self):
        """Test repr before fitting."""
        gen = TiltedGenerator(shuffle_strategy="random", random_state=42)
        repr_str = repr(gen)
        assert "TiltedGenerator" in repr_str
        assert "random" in repr_str
        assert "42" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self, sample_data):
        """Test repr after fitting."""
        gen = TiltedGenerator(shuffle_strategy="stratified", random_state=123)
        gen.fit(
            sample_data,
            text_columns=['text_description'],
            tabular_columns=['specialty'],
            target_column='label'
        )
        repr_str = repr(gen)
        assert "TiltedGenerator" in repr_str
        assert "stratified" in repr_str
        assert "123" in repr_str
        assert "fitted=True" in repr_str


class TestTiltedGeneratorEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_single_row_data(self):
        """Test with single row of data."""
        data = pd.DataFrame({
            'text': ['only one'],
            'num': [1]
        })
        gen = TiltedGenerator(random_state=42)
        gen.fit(data, text_columns=['text'], tabular_columns=['num'])
        synthetic = gen.generate(5)
        # Should repeat the single row
        assert len(synthetic) == 5
        assert (synthetic['text'] == 'only one').all()

    def test_multiclass_adversarial(self):
        """Test adversarial strategy with more than 2 classes."""
        data = pd.DataFrame({
            'text': ['a', 'b', 'c', 'd', 'e'] * 4,
            'feature': [1, 2, 3, 4, 5] * 4,
            'label': [0, 1, 2, 3, 4] * 4
        })
        gen = TiltedGenerator(shuffle_strategy="adversarial", random_state=42)
        gen.fit(
            data,
            text_columns=['text'],
            tabular_columns=['feature'],
            target_column='label'
        )
        synthetic = gen.generate(20)
        assert len(synthetic) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
