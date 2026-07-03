"""
Prompt-Conditioned LLM Generator

This generator uses LLM with tabular data embedded in prompts to generate
semantically consistent text that preserves cross-modal correlations.

Method:
1. Train CTGAN on tabular columns
2. Generate tabular data
3. For each row, build a structured prompt incorporating the tabular values
4. Generate text conditioned on those values using LLM
5. Combine into multimodal dataset with preserved correlations

This approach maintains cross-modal consistency unlike the Stitching Fallacy.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from generators.base import BaseGenerator
import time

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

# Import tqdm for progress tracking
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class PromptLLMGenerator(BaseGenerator):
    """
    Generator that uses LLM with tabular data in prompts to generate text.

    This maintains cross-modal correlations by conditioning text generation
    on specific tabular attribute values, rather than generating independently.
    """

    def __init__(
        self,
        provider: str = 'openai',
        model: str = 'gpt-4o-mini',
        temperature: float = 0.8,
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        random_seed: int = None
    ):
        """
        Initialize the Prompt-Conditioned LLM Generator.

        Args:
            provider: LLM provider ('openai' or 'anthropic')
            model: Model name (default: 'gpt-4o-mini' for OpenAI)
            temperature: Temperature for text generation (default: 0.8)
            batch_size: Number of samples to process between progress updates (default: 10)
            max_retries: Maximum number of retries for API calls (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
            random_seed: Random seed for reproducibility
        """
        if provider not in ['openai', 'anthropic']:
            raise ValueError(f"provider must be 'openai' or 'anthropic', got '{provider}'")

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.random_seed = random_seed

        # State variables
        self.is_fitted = False
        self.text_columns = []
        self.tabular_columns = []
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

    def _build_prompt(self, row: pd.Series, text_column: str) -> str:
        """
        Build a structured prompt incorporating tabular column values.

        Args:
            row: A pandas Series containing tabular values
            text_column: Name of the text column to generate

        Returns:
            Formatted prompt string
        """
        # Build the conditioning context from tabular values
        context_lines = []
        for col in self.tabular_columns:
            value = row[col]
            # Format numeric values nicely
            if isinstance(value, (int, np.integer)):
                formatted_value = str(int(value))
            elif isinstance(value, (float, np.floating)):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            # Create human-readable column names (replace underscores with spaces, title case)
            readable_col = col.replace('_', ' ').title()
            context_lines.append(f"- {readable_col}: {formatted_value}")

        context = "\n".join(context_lines)

        # Build the complete prompt
        prompt = f"""Generate a realistic {text_column.replace('_', ' ')} based on the following context:

{context}

Write only the {text_column.replace('_', ' ')} text (1-2 sentences). Do not include any preamble or explanation."""

        return prompt

    def _generate_text_with_retry(self, prompt: str) -> str:
        """
        Generate text using LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Generated text string

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                if self.provider == 'openai':
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates realistic synthetic text data based on provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=self.temperature
                    )
                    return response.choices[0].message.content.strip()

                elif self.provider == 'anthropic':
                    response = self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=500,
                        temperature=self.temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.content[0].text.strip()

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # Last attempt failed, raise the exception
                    raise last_exception

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

        # Train CTGAN on tabular columns
        tabular_df = real_df[tabular_columns]

        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(tabular_df)

        self.ctgan_model = CTGANSynthesizer(metadata)
        self.ctgan_model.fit(tabular_df)

        self.is_fitted = True

    def __getstate__(self):
        """Custom pickling to handle non-serializable LLM client."""
        state = self.__dict__.copy()
        # Remove the LLM client (it can be recreated on load)
        state['llm_client'] = None
        # Handle CTGAN model pickling
        try:
            import pickle
            import io
            if state['ctgan_model'] is not None:
                buffer = io.BytesIO()
                pickle.dump(state['ctgan_model'], buffer)
        except Exception:
            # If it's not picklable (e.g., a mock), set to None
            state['ctgan_model'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore LLM client."""
        self.__dict__.update(state)
        # LLM client will be reinitialized lazily when needed

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples with prompt-conditioned text generation.

        This method preserves cross-modal correlations by:
        1. Generating tabular data
        2. For each row, conditioning text generation on the tabular values

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data maintaining cross-modal correlations
        """
        if not self.is_fitted:
            raise ValueError("Generator must be fitted before generating samples")

        # Reinitialize LLM client if needed (e.g., after unpickling)
        if self.llm_client is None:
            self._initialize_llm_client()

        # Step 1: Generate tabular data with CTGAN
        tabular_synthetic = self.ctgan_model.sample(n_samples)

        # Step 2: Generate text conditioned on tabular values
        text_synthetic = {col: [] for col in self.text_columns}

        # Use tqdm for progress tracking if available
        if tqdm is not None:
            iterator = tqdm(range(n_samples), desc="Generating text samples")
        else:
            iterator = range(n_samples)

        for i in iterator:
            row = tabular_synthetic.iloc[i]

            # Generate text for each text column, conditioned on tabular values
            for text_col in self.text_columns:
                prompt = self._build_prompt(row, text_col)
                text = self._generate_text_with_retry(prompt)
                text_synthetic[text_col].append(text)

        # Combine into final DataFrame
        result = tabular_synthetic.copy()
        for col, texts in text_synthetic.items():
            result[col] = texts

        # Reorder columns to match original order
        all_columns = self.text_columns + self.tabular_columns
        result = result[all_columns]

        return result
