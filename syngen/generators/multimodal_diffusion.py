"""
Multimodal Diffusion Generator

This generator uses a diffusion model to jointly generate tabular and text data
in a shared latent space, preserving cross-modal correlations through the joint
denoising process.

Method:
1. Encode text using sentence-transformers (pre-trained BERT/RoBERTa)
2. Encode tabular data using a simple MLP
3. Create joint latent representations
4. Train a diffusion denoising network on these joint latents
5. Generate by sampling noise and iteratively denoising
6. Decode latents back to text and tabular data

This approach maintains cross-modal correlations by jointly modeling both
modalities in a shared diffusion process, unlike independent generation.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
from generators.base import BaseGenerator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings


class TabularEncoder(nn.Module):
    """MLP encoder for tabular features."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        """
        Initialize tabular encoder.

        Args:
            input_dim: Number of input tabular features
            latent_dim: Size of latent representation
            hidden_dim: Size of hidden layer
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tabular features to latent space."""
        return self.network(x)


class TabularDecoder(nn.Module):
    """MLP decoder for tabular features."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initialize tabular decoder.

        Args:
            latent_dim: Size of latent representation
            output_dim: Number of output tabular features
            hidden_dim: Size of hidden layer
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to tabular features."""
        return self.network(x)


class DenoisingNetwork(nn.Module):
    """Neural network for denoising in diffusion process."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        """
        Initialize denoising network.

        Args:
            latent_dim: Size of latent representation
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main denoising network
        self.network = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise at timestep t.

        Args:
            x: Noisy latent representation
            t: Timestep (normalized to [0, 1])

        Returns:
            Predicted noise
        """
        # Embed time
        t_emb = self.time_mlp(t.unsqueeze(-1))

        # Concatenate noisy input with time embedding
        x_t = torch.cat([x, t_emb], dim=-1)

        # Predict noise
        return self.network(x_t)


class MultimodalDiffusionGenerator(BaseGenerator):
    """
    Generator that uses diffusion models for joint tabular and text generation.

    This approach preserves cross-modal correlations by jointly modeling both
    modalities in a shared latent space with a diffusion process.
    """

    def __init__(
        self,
        text_encoder_model: str = 'all-MiniLM-L6-v2',
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_diffusion_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 32,
        device: str = None,
        random_seed: int = None
    ):
        """
        Initialize the Multimodal Diffusion Generator.

        Args:
            text_encoder_model: Sentence-transformers model for text encoding
            latent_dim: Dimensionality of joint latent space
            hidden_dim: Hidden dimension for neural networks
            n_diffusion_steps: Number of diffusion steps for generation
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value
            learning_rate: Learning rate for training
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
            random_seed: Random seed for reproducibility
        """
        self.text_encoder_model = text_encoder_model
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # State variables
        self.is_fitted = False
        self.text_columns = []
        self.tabular_columns = []

        # Models (initialized in fit)
        self.text_encoder = None
        self.tabular_encoder = None
        self.tabular_decoder = None
        self.denoising_network = None

        # Data statistics for normalization
        self.tabular_mean = None
        self.tabular_std = None

        # Text embedding cache for nearest neighbor decoding
        self.text_embeddings_cache = None
        self.text_samples_cache = None

        # Noise schedule
        self.betas = None
        self.alphas = None
        self.alpha_bars = None

        # Set random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

    def _setup_noise_schedule(self):
        """Setup linear noise schedule for diffusion."""
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.n_diffusion_steps,
            device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _normalize_tabular(self, tabular_data: np.ndarray) -> np.ndarray:
        """Normalize tabular data to zero mean and unit variance."""
        if self.tabular_mean is None:
            self.tabular_mean = tabular_data.mean(axis=0)
            self.tabular_std = tabular_data.std(axis=0) + 1e-8

        return (tabular_data - self.tabular_mean) / self.tabular_std

    def _denormalize_tabular(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize tabular data back to original scale."""
        return normalized_data * self.tabular_std + self.tabular_mean

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text using sentence-transformers.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings as numpy array
        """
        embeddings = self.text_encoder.encode(texts, show_progress_bar=False)
        return embeddings

    def _decode_text_nearest_neighbor(self, text_embeddings: np.ndarray) -> List[str]:
        """
        Decode text embeddings using nearest neighbor search.

        Args:
            text_embeddings: Text embeddings to decode

        Returns:
            List of decoded text strings
        """
        # Compute cosine similarity with cached embeddings
        # Normalize vectors
        text_emb_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
        cache_emb_norm = self.text_embeddings_cache / (np.linalg.norm(self.text_embeddings_cache, axis=1, keepdims=True) + 1e-8)

        # Compute similarities
        similarities = text_emb_norm @ cache_emb_norm.T

        # Find nearest neighbors
        nearest_indices = similarities.argmax(axis=1)

        # Return corresponding text samples
        decoded_texts = []
        for idx in nearest_indices:
            # Get the corresponding text for each text column
            decoded_texts.append(self.text_samples_cache[idx])

        return decoded_texts

    def _forward_diffusion(self, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to data at timestep t.

        Args:
            x: Clean data
            t: Timestep

        Returns:
            Tuple of (noisy_data, noise)
        """
        noise = torch.randn_like(x, device=self.device)
        alpha_bar = self.alpha_bars[t]

        # x_t = sqrt(alpha_bar) * x + sqrt(1 - alpha_bar) * noise
        noisy_x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

        return noisy_x, noise

    def _reverse_diffusion_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single reverse diffusion step: denoise data at timestep t.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep

        Returns:
            Denoised data at timestep t-1
        """
        # Normalize timestep to [0, 1]
        t_normalized = torch.tensor([t / self.n_diffusion_steps], device=self.device).repeat(x_t.shape[0])

        # Predict noise
        predicted_noise = self.denoising_network(x_t, t_normalized)

        # Get noise schedule values
        beta = self.betas[t]
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]

        # Compute mean of x_{t-1}
        coef1 = 1 / torch.sqrt(alpha)
        coef2 = beta / torch.sqrt(1 - alpha_bar)
        mean = coef1 * (x_t - coef2 * predicted_noise)

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_t, device=self.device)
            alpha_bar_prev = self.alpha_bars[t - 1]
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    def fit(self, real_df: pd.DataFrame, text_columns: List[str], tabular_columns: List[str]):
        """
        Fit the diffusion generator on real data.

        Args:
            real_df: Real dataset to learn from
            text_columns: List of text column names
            tabular_columns: List of tabular column names
        """
        self.text_columns = text_columns
        self.tabular_columns = tabular_columns

        # Initialize text encoder
        self.text_encoder = SentenceTransformer(self.text_encoder_model)
        # Use get_embedding_dimension (new API) or fall back to get_sentence_embedding_dimension
        if hasattr(self.text_encoder, 'get_embedding_dimension'):
            text_embedding_dim = self.text_encoder.get_embedding_dimension()
        else:
            text_embedding_dim = self.text_encoder.get_sentence_embedding_dimension()

        # Encode all text data
        all_texts = []
        for col in text_columns:
            all_texts.extend(real_df[col].tolist())

        text_embeddings = self._encode_text(all_texts)

        # Reshape text embeddings by row (concatenate columns)
        n_samples = len(real_df)
        n_text_cols = len(text_columns)
        text_emb_per_sample = text_embeddings.reshape(n_text_cols, n_samples, -1).transpose(1, 0, 2)
        text_emb_flat = text_emb_per_sample.reshape(n_samples, -1)

        # Cache text embeddings for nearest neighbor decoding
        self.text_embeddings_cache = text_embeddings
        self.text_samples_cache = all_texts

        # Get tabular data
        tabular_data = real_df[tabular_columns].values.astype(np.float32)

        # Normalize tabular data
        normalized_tabular = self._normalize_tabular(tabular_data)

        # Initialize tabular encoder/decoder
        tabular_dim = len(tabular_columns)
        self.tabular_encoder = TabularEncoder(tabular_dim, self.latent_dim, self.hidden_dim).to(self.device)
        self.tabular_decoder = TabularDecoder(self.latent_dim, tabular_dim, self.hidden_dim).to(self.device)

        # Encode tabular to latent space
        tabular_tensor = torch.FloatTensor(normalized_tabular).to(self.device)
        with torch.no_grad():
            tabular_latents = self.tabular_encoder(tabular_tensor).cpu().numpy()

        # Project text embeddings to latent space (simple linear projection)
        text_emb_dim = text_emb_flat.shape[1]
        text_projection = nn.Linear(text_emb_dim, self.latent_dim).to(self.device)
        text_tensor = torch.FloatTensor(text_emb_flat).to(self.device)
        with torch.no_grad():
            text_latents = text_projection(text_tensor).cpu().numpy()

        # Create joint latent representation (concatenate)
        joint_latents = np.concatenate([text_latents, tabular_latents], axis=1)
        joint_latent_dim = joint_latents.shape[1]

        # Initialize denoising network
        self.denoising_network = DenoisingNetwork(joint_latent_dim, self.hidden_dim).to(self.device)

        # Setup noise schedule
        self._setup_noise_schedule()

        # Training loop
        optimizer = optim.Adam(
            list(self.tabular_encoder.parameters()) +
            list(self.tabular_decoder.parameters()) +
            list(text_projection.parameters()) +
            list(self.denoising_network.parameters()),
            lr=self.learning_rate
        )

        joint_latents_tensor = torch.FloatTensor(joint_latents).to(self.device)

        self.tabular_encoder.train()
        self.tabular_decoder.train()
        text_projection.train()
        self.denoising_network.train()

        pbar = tqdm(range(self.n_epochs), desc="Training diffusion model")
        for epoch in pbar:
            # Sample random timesteps
            t = torch.randint(0, self.n_diffusion_steps, (len(joint_latents_tensor),), device=self.device)

            # Forward diffusion
            losses = []
            for i in range(0, len(joint_latents_tensor), self.batch_size):
                batch_idx = slice(i, min(i + self.batch_size, len(joint_latents_tensor)))
                batch_x = joint_latents_tensor[batch_idx]
                batch_t = t[batch_idx]

                # Add noise
                batch_losses = []
                for j in range(len(batch_x)):
                    noisy_x, noise = self._forward_diffusion(batch_x[j:j+1], batch_t[j].item())

                    # Predict noise
                    t_normalized = torch.tensor([batch_t[j].item() / self.n_diffusion_steps], device=self.device)
                    predicted_noise = self.denoising_network(noisy_x, t_normalized)

                    # Compute loss
                    loss = nn.functional.mse_loss(predicted_noise, noise)
                    batch_losses.append(loss)

                if batch_losses:
                    batch_loss = torch.stack(batch_losses).mean()

                    # Backpropagation
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    losses.append(batch_loss.item())

            avg_loss = np.mean(losses) if losses else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Store text projection for generation
        self.text_projection = text_projection

        # Set models to eval mode
        self.tabular_encoder.eval()
        self.tabular_decoder.eval()
        self.text_projection.eval()
        self.denoising_network.eval()

        self.is_fitted = True

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples using reverse diffusion.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data maintaining cross-modal correlations
        """
        if not self.is_fitted:
            raise ValueError("Generator must be fitted before generating samples")

        # Start from random noise
        joint_latent_dim = self.latent_dim * 2  # text + tabular
        x_t = torch.randn(n_samples, joint_latent_dim, device=self.device)

        # Reverse diffusion process
        self.denoising_network.eval()
        with torch.no_grad():
            for t in tqdm(reversed(range(self.n_diffusion_steps)), desc="Generating samples", total=self.n_diffusion_steps):
                x_t = self._reverse_diffusion_step(x_t, t)

        # Split joint latents into text and tabular
        text_latents = x_t[:, :self.latent_dim]
        tabular_latents = x_t[:, self.latent_dim:]

        # Decode tabular features
        with torch.no_grad():
            tabular_output = self.tabular_decoder(tabular_latents).cpu().numpy()

        # Denormalize tabular data
        tabular_synthetic = self._denormalize_tabular(tabular_output)

        # Decode text features
        # Project latents back to text embedding space
        text_emb_dim = self.text_embeddings_cache.shape[1]
        n_text_cols = len(self.text_columns)

        # Use inverse projection (pseudo-inverse)
        with torch.no_grad():
            # Simple approach: use nearest neighbor in latent space
            # Map back to embedding space using learned projection
            text_projection_weight = self.text_projection.weight.data.cpu().numpy()
            text_projection_bias = self.text_projection.bias.data.cpu().numpy()

            # Pseudo-inverse to go from latent to embedding
            text_embeddings_reconstructed = (text_latents.cpu().numpy() - text_projection_bias) @ np.linalg.pinv(text_projection_weight.T)

        # Reshape to per-column embeddings
        text_embeddings_per_col = text_embeddings_reconstructed.reshape(n_samples, n_text_cols, -1)

        # Decode each text column using nearest neighbor
        text_synthetic = {col: [] for col in self.text_columns}

        for i in range(n_samples):
            for j, col in enumerate(self.text_columns):
                # Get embedding for this column
                col_embedding = text_embeddings_per_col[i, j:j+1, :]

                # Decode using nearest neighbor
                decoded_text = self._decode_text_nearest_neighbor(col_embedding)[0]
                text_synthetic[col].append(decoded_text)

        # Combine into DataFrame
        result = pd.DataFrame(tabular_synthetic, columns=self.tabular_columns)
        for col in self.text_columns:
            result[col] = text_synthetic[col]

        # Reorder columns to match original order
        all_columns = self.text_columns + self.tabular_columns
        result = result[all_columns]

        return result

    def __getstate__(self):
        """Custom pickling to handle PyTorch models."""
        state = self.__dict__.copy()
        # Convert PyTorch models to CPU and state dicts
        if self.tabular_encoder is not None:
            state['tabular_encoder'] = self.tabular_encoder.cpu().state_dict()
        if self.tabular_decoder is not None:
            state['tabular_decoder'] = self.tabular_decoder.cpu().state_dict()
        if self.denoising_network is not None:
            state['denoising_network'] = self.denoising_network.cpu().state_dict()
        if hasattr(self, 'text_projection') and self.text_projection is not None:
            state['text_projection'] = self.text_projection.cpu().state_dict()

        # Remove sentence-transformers model (will be reloaded)
        state['text_encoder'] = None

        return state

    def __setstate__(self, state):
        """Custom unpickling to restore PyTorch models."""
        self.__dict__.update(state)

        # Reload sentence-transformers model
        if self.is_fitted:
            self.text_encoder = SentenceTransformer(self.text_encoder_model)

            # Reconstruct PyTorch models
            if isinstance(self.tabular_encoder, dict):
                tabular_dim = len(self.tabular_columns)
                self.tabular_encoder = TabularEncoder(tabular_dim, self.latent_dim, self.hidden_dim).to(self.device)
                self.tabular_encoder.load_state_dict(state['tabular_encoder'])

            if isinstance(self.tabular_decoder, dict):
                tabular_dim = len(self.tabular_columns)
                self.tabular_decoder = TabularDecoder(self.latent_dim, tabular_dim, self.hidden_dim).to(self.device)
                self.tabular_decoder.load_state_dict(state['tabular_decoder'])

            if isinstance(self.denoising_network, dict):
                joint_latent_dim = self.latent_dim * 2
                self.denoising_network = DenoisingNetwork(joint_latent_dim, self.hidden_dim).to(self.device)
                self.denoising_network.load_state_dict(state['denoising_network'])

            if 'text_projection' in state and isinstance(state['text_projection'], dict):
                text_emb_dim = self.text_embeddings_cache.shape[1]
                n_text_cols = len(self.text_columns)
                self.text_projection = nn.Linear(text_emb_dim * n_text_cols, self.latent_dim).to(self.device)
                self.text_projection.load_state_dict(state['text_projection'])
