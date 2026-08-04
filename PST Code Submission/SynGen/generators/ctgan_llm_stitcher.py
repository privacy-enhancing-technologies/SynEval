"""
CTGAN + LLM Stitcher Generator

This generator demonstrates the Stitching Fallacy by generating tabular and text
data independently, then randomly pairing them. This approach should fail multimodal
evaluation metrics that check for correlation between text and tabular features.

Method:
1. Train CTGAN on tabular columns only
2. Extract few-shot examples from text columns
3. Generate tabular data with CTGAN
4. Generate text data with LLM (few-shot prompting)
5. Randomly shuffle and pair them (this breaks correlations)
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from generators.base import BaseGenerator

# Import CTGAN from SDV
from sdv.single_table import CTGANSynthesizer

# Import LLM clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class CTGANLLMStitcher(BaseGenerator):
    """
    Generator that combines CTGAN (tabular) and LLM (text) with random pairing.

    This demonstrates the Stitching Fallacy - a common mistake in multimodal
    synthetic data generation where modalities are generated independently
    and then randomly combined, destroying cross-modal correlations.
    """

    def __init__(
        self,
        provider: str = 'openai',
        model: str = 'gpt-4o-mini',
        n_few_shot: int = 3,
        random_seed: int = None
    ):
        """
        Initialize the CTGAN + LLM Stitcher.

        Args:
            provider: LLM provider ('openai' or 'anthropic')
            model: Model name (default: 'gpt-4o-mini' for OpenAI)
            n_few_shot: Number of few-shot examples to use (default: 3)
            random_seed: Random seed for reproducibility
        """
        if provider not in ['openai', 'anthropic']:
            raise ValueError(f"provider must be 'openai' or 'anthropic', got '{provider}'")

        self.provider = provider
        self.model = model
        self.n_few_shot = n_few_shot
        self.random_seed = random_seed

        # State variables
        self.is_fitted = False
        self.text_columns = []
        self.tabular_columns = []
        self.few_shot_examples: Dict[str, List[str]] = {}
        self.ctgan_model = None
        self.llm_client = None

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

    def _initialize_llm_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == 'openai':
            if OpenAI is None:
                raise ImportError("openai package not installed. Install with: pip install openai")

            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Please set it with your OpenAI API key."
                )

            self.llm_client = OpenAI(api_key=api_key)

        elif self.provider == 'anthropic':
            if Anthropic is None:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")

            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Please set it with your Anthropic API key."
                )

            self.llm_client = Anthropic(api_key=api_key)

    def _extract_few_shot_examples(self, real_df: pd.DataFrame):
        """
        Extract few-shot examples from text columns.

        Args:
            real_df: Real dataset containing text columns
        """
        self.few_shot_examples = {}

        for col in self.text_columns:
            # Sample up to n_few_shot examples from the column
            n_samples = min(self.n_few_shot, len(real_df))
            examples = real_df[col].sample(n=n_samples, random_state=self.random_seed).tolist()
            self.few_shot_examples[col] = examples

    def _build_few_shot_prompt(self, column_name: str) -> str:
        """
        Build few-shot prompt for text generation.

        Args:
            column_name: Name of the text column to generate

        Returns:
            Formatted few-shot prompt
        """
        examples = self.few_shot_examples[column_name]

        prompt = f"Generate a {column_name} similar to these examples:\n\n"

        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}: {example}\n"

        prompt += f"\nNow generate a new {column_name} (return ONLY the generated text, no preamble):"

        return prompt

    def _generate_text_with_llm(self, column_name: str) -> str:
        """
        Generate a single text sample using LLM.

        Args:
            column_name: Name of the text column

        Returns:
            Generated text string
        """
        prompt = self._build_few_shot_prompt(column_name)

        if self.provider == 'openai':
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates synthetic text data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()

        elif self.provider == 'anthropic':
            response = self.llm_client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.8,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()

    def fit(self, real_df: pd.DataFrame, text_columns: List[str], tabular_columns: List[str]):
        """
        Fit the generator on real data.

        Args:
            real_df: Real dataset to learn from
            text_columns: List of text column names
            tabular_columns: List of tabular column names
        """
        self.text_columns = text_columns
        self.tabular_columns = tabular_columns

        # Initialize LLM client
        self._initialize_llm_client()

        # Extract few-shot examples for text generation
        self._extract_few_shot_examples(real_df)

        # Train CTGAN on tabular columns only
        tabular_df = real_df[tabular_columns]

        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(tabular_df)

        self.ctgan_model = CTGANSynthesizer(metadata)
        self.ctgan_model.fit(tabular_df)

        self.is_fitted = True

    def __getstate__(self):
        """Custom pickling to handle non-serializable LLM client and CTGAN model."""
        state = self.__dict__.copy()
        # Remove the LLM client (it can be recreated on load)
        # We keep it as None so it can be reinitialized later
        state['llm_client'] = None
        # Store CTGAN model separately if it's not None
        # CTGANSynthesizer is picklable, but in tests it might be a mock which is not
        try:
            # Try to pickle test if ctgan_model is picklable
            import pickle
            import io
            if state['ctgan_model'] is not None:
                buffer = io.BytesIO()
                pickle.dump(state['ctgan_model'], buffer)
        except Exception:
            # If it's not picklable (e.g., a mock), set to None
            # Real models should be picklable
            state['ctgan_model'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore LLM client."""
        self.__dict__.update(state)
        # Don't reinitialize automatically - let it be lazy initialized when needed
        # This avoids issues with missing API keys during unpickling

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples using the Stitching approach.

        This method demonstrates the Stitching Fallacy:
        1. Generate tabular data with CTGAN
        2. Generate text data with LLM (independently)
        3. Randomly pair them (destroying correlations)

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data (text and tabular randomly paired)
        """
        if not self.is_fitted:
            raise ValueError("Generator must be fitted before generating samples")

        # Reinitialize LLM client if needed (e.g., after unpickling)
        if self.llm_client is None:
            self._initialize_llm_client()

        # Step 1: Generate tabular data with CTGAN
        tabular_synthetic = self.ctgan_model.sample(n_samples)

        # Step 2: Generate text data with LLM (independently)
        text_synthetic = {}

        for col in self.text_columns:
            text_samples = []
            for _ in range(n_samples):
                text = self._generate_text_with_llm(col)
                text_samples.append(text)
            text_synthetic[col] = text_samples

        # Step 3: Random pairing (THE STITCHING FALLACY)
        # We shuffle the text samples relative to tabular samples
        # This breaks any correlation between text and tabular features
        for col in self.text_columns:
            # Create a random permutation
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed)
                perm = rng.permutation(n_samples)
            else:
                perm = np.random.permutation(n_samples)

            # Shuffle the text samples
            text_synthetic[col] = [text_synthetic[col][i] for i in perm]

        # Combine into final DataFrame
        result = tabular_synthetic.copy()
        for col, texts in text_synthetic.items():
            result[col] = texts

        # Reorder columns to match original order
        all_columns = self.text_columns + self.tabular_columns
        result = result[all_columns]

        return result
