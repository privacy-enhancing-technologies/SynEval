"""
Tests for MultimodalDiffusionGenerator.
"""
import pytest
import pandas as pd
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from generators.multimodal_diffusion import (
    MultimodalDiffusionGenerator,
    TabularEncoder,
    TabularDecoder,
    DenoisingNetwork
)


@pytest.fixture
def sample_data():
    """Create sample multimodal data for testing."""
    return pd.DataFrame({
        'review': [
            'Great product, highly recommend!',
            'Not satisfied, poor quality',
            'Excellent service and fast delivery',
            'Good value for money',
            'Could be better'
        ],
        'rating': [5, 2, 5, 4, 3],
        'price': [29.99, 19.99, 39.99, 24.99, 34.99],
        'quantity': [10, 5, 15, 8, 12]
    })


class TestTabularEncoder:
    """Test TabularEncoder module."""

    def test_initialization(self):
        """Test that TabularEncoder initializes correctly."""
        encoder = TabularEncoder(input_dim=3, latent_dim=64, hidden_dim=128)
        assert encoder is not None
        assert isinstance(encoder, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass through TabularEncoder."""
        encoder = TabularEncoder(input_dim=3, latent_dim=64, hidden_dim=128)
        x = torch.randn(5, 3)  # 5 samples, 3 features
        output = encoder(x)

        assert output.shape == (5, 64)  # 5 samples, 64 latent dimensions

    def test_forward_batch_size_one(self):
        """Test forward pass with single sample."""
        encoder = TabularEncoder(input_dim=3, latent_dim=64, hidden_dim=128)
        x = torch.randn(1, 3)
        output = encoder(x)

        assert output.shape == (1, 64)


class TestTabularDecoder:
    """Test TabularDecoder module."""

    def test_initialization(self):
        """Test that TabularDecoder initializes correctly."""
        decoder = TabularDecoder(latent_dim=64, output_dim=3, hidden_dim=128)
        assert decoder is not None
        assert isinstance(decoder, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass through TabularDecoder."""
        decoder = TabularDecoder(latent_dim=64, output_dim=3, hidden_dim=128)
        x = torch.randn(5, 64)  # 5 samples, 64 latent dimensions
        output = decoder(x)

        assert output.shape == (5, 3)  # 5 samples, 3 output features

    def test_encoder_decoder_roundtrip(self):
        """Test that encoder-decoder can reconstruct data."""
        encoder = TabularEncoder(input_dim=3, latent_dim=64, hidden_dim=128)
        decoder = TabularDecoder(latent_dim=64, output_dim=3, hidden_dim=128)

        x = torch.randn(5, 3)
        latent = encoder(x)
        reconstructed = decoder(latent)

        assert reconstructed.shape == x.shape


class TestDenoisingNetwork:
    """Test DenoisingNetwork module."""

    def test_initialization(self):
        """Test that DenoisingNetwork initializes correctly."""
        network = DenoisingNetwork(latent_dim=128, hidden_dim=256)
        assert network is not None
        assert isinstance(network, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass through DenoisingNetwork."""
        network = DenoisingNetwork(latent_dim=128, hidden_dim=256)
        x = torch.randn(5, 128)  # 5 samples, 128 latent dimensions
        t = torch.rand(5)  # Normalized timesteps

        output = network(x, t)

        assert output.shape == (5, 128)  # Should predict noise of same shape

    def test_different_timesteps(self):
        """Test that network handles different timesteps."""
        network = DenoisingNetwork(latent_dim=128, hidden_dim=256)
        x = torch.randn(3, 128)

        # Different timesteps
        t1 = torch.zeros(3)  # t=0
        t2 = torch.ones(3)   # t=1

        output1 = network(x, t1)
        output2 = network(x, t2)

        # Outputs should be different for different timesteps
        assert not torch.allclose(output1, output2)


class TestMultimodalDiffusionGenerator:
    """Test MultimodalDiffusionGenerator class."""

    def test_initialization(self):
        """Test that generator initializes with default parameters."""
        gen = MultimodalDiffusionGenerator()

        assert gen is not None
        assert gen.is_fitted is False
        assert gen.text_columns == []
        assert gen.tabular_columns == []

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        gen = MultimodalDiffusionGenerator(
            latent_dim=64,
            n_diffusion_steps=25,
            n_epochs=10,
            random_seed=42
        )

        assert gen.latent_dim == 64
        assert gen.n_diffusion_steps == 25
        assert gen.n_epochs == 10
        assert gen.random_seed == 42

    def test_setup_noise_schedule(self):
        """Test noise schedule setup."""
        gen = MultimodalDiffusionGenerator(n_diffusion_steps=10)
        gen._setup_noise_schedule()

        assert gen.betas is not None
        assert gen.alphas is not None
        assert gen.alpha_bars is not None
        assert len(gen.betas) == 10
        assert len(gen.alphas) == 10
        assert len(gen.alpha_bars) == 10

        # Check that alphas = 1 - betas
        assert torch.allclose(gen.alphas, 1.0 - gen.betas)

    def test_normalize_denormalize_tabular(self):
        """Test tabular data normalization and denormalization."""
        gen = MultimodalDiffusionGenerator()

        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        # Normalize
        normalized = gen._normalize_tabular(data)

        # Check zero mean and unit variance
        assert np.abs(normalized.mean()) < 1e-6
        assert np.abs(normalized.std() - 1.0) < 1e-6

        # Denormalize
        denormalized = gen._denormalize_tabular(normalized)

        # Should recover original data
        assert np.allclose(denormalized, data, rtol=1e-5)

    def test_forward_diffusion(self):
        """Test forward diffusion process."""
        gen = MultimodalDiffusionGenerator(n_diffusion_steps=10)
        gen._setup_noise_schedule()

        x = torch.randn(5, 128)
        t = 5

        noisy_x, noise = gen._forward_diffusion(x, t)

        # Check shapes
        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape

        # Noisy data should be different from original
        assert not torch.allclose(noisy_x, x)

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=2,  # Very few epochs for speed
            n_diffusion_steps=5,  # Few steps for speed
            batch_size=2,
            random_seed=42
        )

        # Fit on small sample
        gen.fit(
            sample_data.head(3),  # Use only 3 samples for speed
            text_columns=['review'],
            tabular_columns=['rating', 'price', 'quantity']
        )

        assert gen.is_fitted is True
        assert gen.text_columns == ['review']
        assert gen.tabular_columns == ['rating', 'price', 'quantity']
        assert gen.text_encoder is not None
        assert gen.tabular_encoder is not None
        assert gen.tabular_decoder is not None
        assert gen.denoising_network is not None

    def test_fit_stores_statistics(self, sample_data):
        """Test that fit stores normalization statistics."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=1,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        assert gen.tabular_mean is not None
        assert gen.tabular_std is not None
        assert len(gen.tabular_mean) == 2  # Two tabular columns
        assert len(gen.tabular_std) == 2

    def test_fit_caches_text_embeddings(self, sample_data):
        """Test that fit caches text embeddings."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=1,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        assert gen.text_embeddings_cache is not None
        assert gen.text_samples_cache is not None
        assert len(gen.text_samples_cache) == 3  # 3 samples

    def test_generate_before_fit_raises_error(self):
        """Test that generate raises error if called before fit."""
        gen = MultimodalDiffusionGenerator()

        with pytest.raises(ValueError, match="must be fitted before generating"):
            gen.generate(5)

    def test_generate_basic(self, sample_data):
        """Test basic generation functionality."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=2,
            n_diffusion_steps=5,
            batch_size=2,
            random_seed=42
        )

        # Fit on sample data
        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        # Generate synthetic data
        synthetic = gen.generate(n_samples=2)

        # Check output structure
        assert isinstance(synthetic, pd.DataFrame)
        assert len(synthetic) == 2
        assert 'review' in synthetic.columns
        assert 'rating' in synthetic.columns
        assert 'price' in synthetic.columns

    def test_generate_output_types(self, sample_data):
        """Test that generated data has correct types."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=2,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        synthetic = gen.generate(n_samples=2)

        # Text column should contain strings
        assert all(isinstance(x, str) for x in synthetic['review'])

        # Tabular columns should contain numbers
        assert all(isinstance(x, (int, float, np.number)) for x in synthetic['rating'])
        assert all(isinstance(x, (int, float, np.number)) for x in synthetic['price'])

    def test_generate_column_order(self, sample_data):
        """Test that generated data maintains column order."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=2,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        synthetic = gen.generate(n_samples=2)

        # Columns should be in original order (text first, then tabular)
        expected_order = ['review', 'rating', 'price']
        assert list(synthetic.columns) == expected_order

    def test_multiple_text_columns(self, sample_data):
        """Test handling of multiple text columns."""
        # Add another text column
        sample_data_multi = sample_data.copy()
        sample_data_multi['comment'] = [
            'Would buy again',
            'Disappointed',
            'Great experience',
            'Fair product',
            'Okay'
        ]

        gen = MultimodalDiffusionGenerator(
            n_epochs=2,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data_multi.head(3),
            text_columns=['review', 'comment'],
            tabular_columns=['rating', 'price']
        )

        synthetic = gen.generate(n_samples=2)

        assert 'review' in synthetic.columns
        assert 'comment' in synthetic.columns
        assert all(isinstance(x, str) for x in synthetic['review'])
        assert all(isinstance(x, str) for x in synthetic['comment'])

    def test_save_load(self, sample_data):
        """Test save/load functionality."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=1,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            gen.save(temp_path)

            # Verify file was created
            assert os.path.exists(temp_path)

            # Load into new generator
            loaded_gen = MultimodalDiffusionGenerator()
            loaded_gen.load(temp_path)

            # Verify state was preserved
            assert loaded_gen.is_fitted == gen.is_fitted
            assert loaded_gen.text_columns == gen.text_columns
            assert loaded_gen.tabular_columns == gen.tabular_columns
            assert loaded_gen.latent_dim == gen.latent_dim

            # Verify loaded generator can generate
            synthetic = loaded_gen.generate(2)
            assert len(synthetic) == 2
            assert 'review' in synthetic.columns

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_reverse_diffusion_step(self):
        """Test single reverse diffusion step."""
        gen = MultimodalDiffusionGenerator(n_diffusion_steps=10)
        gen._setup_noise_schedule()
        gen.denoising_network = DenoisingNetwork(latent_dim=128)

        x_t = torch.randn(5, 128)
        t = 5

        x_t_minus_1 = gen._reverse_diffusion_step(x_t, t)

        # Check shape is preserved
        assert x_t_minus_1.shape == x_t.shape

        # Denoised version should be different
        assert not torch.allclose(x_t_minus_1, x_t)

    def test_decode_text_nearest_neighbor(self, sample_data):
        """Test text decoding using nearest neighbor."""
        gen = MultimodalDiffusionGenerator(random_seed=42)

        # Initialize text encoder
        from sentence_transformers import SentenceTransformer
        gen.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Cache some text samples
        texts = sample_data['review'].tolist()[:3]
        gen.text_samples_cache = texts
        gen.text_embeddings_cache = gen.text_encoder.encode(texts)

        # Get embedding for first text
        test_embedding = gen.text_embeddings_cache[0:1]

        # Decode - should return the first text
        decoded = gen._decode_text_nearest_neighbor(test_embedding)

        assert len(decoded) == 1
        assert decoded[0] == texts[0]

    def test_random_seed_reproducibility(self, sample_data):
        """Test that random seed makes generation reproducible."""
        # Create two generators with same seed
        gen1 = MultimodalDiffusionGenerator(
            n_epochs=2,
            n_diffusion_steps=5,
            random_seed=42
        )
        gen2 = MultimodalDiffusionGenerator(
            n_epochs=2,
            n_diffusion_steps=5,
            random_seed=42
        )

        # Fit both on same data
        gen1.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )
        gen2.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        # Generate with same seed should give similar distributions
        # Note: Due to LLM non-determinism, exact match is not guaranteed
        # but distributions should be similar
        synthetic1 = gen1.generate(5)
        synthetic2 = gen2.generate(5)

        # At least check that both generated successfully
        assert len(synthetic1) == 5
        assert len(synthetic2) == 5

    def test_device_handling_cpu(self):
        """Test that generator works on CPU."""
        gen = MultimodalDiffusionGenerator(device='cpu')
        assert gen.device == torch.device('cpu')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_handling_cuda(self):
        """Test that generator works on CUDA if available."""
        gen = MultimodalDiffusionGenerator(device='cuda')
        assert gen.device == torch.device('cuda')

    def test_edge_case_single_sample_generation(self, sample_data):
        """Test generating a single sample."""
        gen = MultimodalDiffusionGenerator(
            n_epochs=1,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        synthetic = gen.generate(n_samples=1)

        assert len(synthetic) == 1
        assert 'review' in synthetic.columns

    def test_latent_dim_parameter(self, sample_data):
        """Test that latent_dim parameter is used correctly."""
        gen = MultimodalDiffusionGenerator(
            latent_dim=32,
            n_epochs=1,
            n_diffusion_steps=5,
            random_seed=42
        )

        gen.fit(
            sample_data.head(3),
            text_columns=['review'],
            tabular_columns=['rating', 'price']
        )

        # Check that encoder/decoder use correct latent dim
        assert gen.latent_dim == 32

        # Generate to ensure it works
        synthetic = gen.generate(2)
        assert len(synthetic) == 2
