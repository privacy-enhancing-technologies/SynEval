"""
Tests for PromptLLMGenerator.

This generator uses LLM with tabular data in prompts to generate text
that maintains cross-modal correlations.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from generators.prompt_llm import PromptLLMGenerator


@pytest.fixture
def sample_data():
    """Create sample dataset with text and tabular columns."""
    return pd.DataFrame({
        'loan_purpose': [
            'To buy seeds and fertilizer for planting season',
            'Purchase livestock for dairy business',
            'Expand inventory for retail shop',
            'Buy sewing machine for tailoring business',
            'Invest in fishing equipment and nets'
        ],
        'sector': ['Agriculture', 'Agriculture', 'Retail', 'Services', 'Agriculture'],
        'loan_amount': [500, 1200, 800, 350, 900],
        'term_in_months': [12, 24, 18, 6, 15]
    })


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "To purchase agricultural equipment for farming operations."
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "To invest in business expansion and inventory."
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_prompt_llm_initialization():
    """Test that PromptLLMGenerator can be initialized."""
    gen = PromptLLMGenerator(provider='openai', model='gpt-4o-mini')
    assert gen.provider == 'openai'
    assert gen.model == 'gpt-4o-mini'
    assert gen.temperature == 0.8
    assert gen.batch_size == 10
    assert gen.is_fitted is False


def test_prompt_llm_default_initialization():
    """Test default initialization."""
    gen = PromptLLMGenerator()
    assert gen.provider == 'openai'
    assert gen.model == 'gpt-4o-mini'
    assert gen.temperature == 0.8


def test_prompt_llm_custom_parameters():
    """Test initialization with custom parameters."""
    gen = PromptLLMGenerator(
        provider='anthropic',
        model='claude-3-haiku-20240307',
        temperature=0.5,
        batch_size=20,
        max_retries=5,
        retry_delay=2.0
    )
    assert gen.provider == 'anthropic'
    assert gen.model == 'claude-3-haiku-20240307'
    assert gen.temperature == 0.5
    assert gen.batch_size == 20
    assert gen.max_retries == 5
    assert gen.retry_delay == 2.0


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_fit(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test fitting the generator on sample data."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Set API key
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai', model='gpt-4o-mini')
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

    # Verify state
    assert gen.is_fitted is True
    assert gen.text_columns == ['loan_purpose']
    assert gen.tabular_columns == ['sector', 'loan_amount', 'term_in_months']

    # Verify CTGAN was fitted on tabular columns only
    mock_ctgan_instance.fit.assert_called_once()
    fitted_df = mock_ctgan_instance.fit.call_args[0][0]
    assert set(fitted_df.columns) == {'sector', 'loan_amount', 'term_in_months'}


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_build_prompt(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test that prompts are correctly constructed with tabular values."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Fit generator
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai')
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

    # Create a test row
    test_row = pd.Series({
        'sector': 'Agriculture',
        'loan_amount': 500,
        'term_in_months': 14
    })

    # Build prompt
    prompt = gen._build_prompt(test_row, 'loan_purpose')

    # Verify prompt contains the tabular values
    assert 'Agriculture' in prompt
    assert '500' in prompt
    assert '14' in prompt
    assert 'loan purpose' in prompt.lower() or 'loan_purpose' in prompt.lower()

    # Verify it's asking for text generation
    assert any(word in prompt.lower() for word in ['generate', 'write', 'create'])


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_generate(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test generating synthetic samples."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Mock CTGAN generate
    mock_tabular_df = pd.DataFrame({
        'sector': ['Agriculture', 'Retail', 'Services'],
        'loan_amount': [600, 750, 400],
        'term_in_months': [12, 18, 9]
    })
    mock_ctgan_instance.sample.return_value = mock_tabular_df

    # Fit and generate
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai', model='gpt-4o-mini')
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

        synthetic_df = gen.generate(3)

    # Verify output
    assert len(synthetic_df) == 3
    assert set(synthetic_df.columns) == {'loan_purpose', 'sector', 'loan_amount', 'term_in_months'}
    assert all(isinstance(text, str) for text in synthetic_df['loan_purpose'])

    # Verify CTGAN sample was called
    mock_ctgan_instance.sample.assert_called_once_with(3)

    # Verify LLM was called for each sample (3 samples * 1 text column = 3 calls)
    assert mock_openai_client.chat.completions.create.call_count == 3


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_conditioned_on_tabular(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """
    Test that text generation is conditioned on tabular values.

    This is the key test: we verify that the prompts include the tabular values
    from each specific row, maintaining cross-modal correlation.
    """
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Mock CTGAN to return specific tabular data
    mock_tabular_df = pd.DataFrame({
        'sector': ['Agriculture', 'Retail'],
        'loan_amount': [500, 1000],
        'term_in_months': [12, 24]
    })
    mock_ctgan_instance.sample.return_value = mock_tabular_df

    # Track prompts sent to the LLM
    prompts_sent = []

    def capture_prompt(*args, **kwargs):
        messages = kwargs.get('messages', [])
        for msg in messages:
            if msg['role'] == 'user':
                prompts_sent.append(msg['content'])

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Generated text"
        return response

    mock_openai_client.chat.completions.create.side_effect = capture_prompt

    # Fit and generate
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai')
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

        synthetic_df = gen.generate(2)

    # Verify we got 2 prompts
    assert len(prompts_sent) == 2

    # First prompt should contain values from first row
    assert 'Agriculture' in prompts_sent[0]
    assert '500' in prompts_sent[0]
    assert '12' in prompts_sent[0]

    # Second prompt should contain values from second row
    assert 'Retail' in prompts_sent[1]
    assert '1000' in prompts_sent[1]
    assert '24' in prompts_sent[1]


@patch('generators.prompt_llm.Anthropic')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_anthropic(mock_ctgan, mock_anthropic_class, sample_data, mock_anthropic_client):
    """Test using Anthropic provider."""
    # Setup mocks
    mock_anthropic_class.return_value = mock_anthropic_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Mock CTGAN generate
    mock_tabular_df = pd.DataFrame({
        'sector': ['Agriculture', 'Services'],
        'loan_amount': [800, 450],
        'term_in_months': [15, 10]
    })
    mock_ctgan_instance.sample.return_value = mock_tabular_df

    # Fit and generate
    with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='anthropic', model='claude-3-haiku-20240307')
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

        synthetic_df = gen.generate(2)

    # Verify output
    assert len(synthetic_df) == 2
    assert 'loan_purpose' in synthetic_df.columns

    # Verify Anthropic API was called
    assert mock_anthropic_client.messages.create.call_count == 2


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_retry_logic(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test retry logic on API failures."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Mock CTGAN generate
    mock_tabular_df = pd.DataFrame({
        'sector': ['Agriculture'],
        'loan_amount': [500],
        'term_in_months': [12]
    })
    mock_ctgan_instance.sample.return_value = mock_tabular_df

    # Make API fail twice, then succeed
    call_count = 0

    def api_with_failures(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("API rate limit exceeded")
        else:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Success after retry"
            return response

    mock_openai_client.chat.completions.create.side_effect = api_with_failures

    # Fit and generate (should retry and eventually succeed)
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai', max_retries=3, retry_delay=0.01)
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

        synthetic_df = gen.generate(1)

    # Should have succeeded after retries
    assert len(synthetic_df) == 1
    assert synthetic_df['loan_purpose'].iloc[0] == "Success after retry"
    assert call_count == 3


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_retry_exhausted(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test that exceptions are raised when retries are exhausted."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Mock CTGAN generate
    mock_tabular_df = pd.DataFrame({
        'sector': ['Agriculture'],
        'loan_amount': [500],
        'term_in_months': [12]
    })
    mock_ctgan_instance.sample.return_value = mock_tabular_df

    # Make API always fail
    mock_openai_client.chat.completions.create.side_effect = Exception("Persistent API error")

    # Fit and try to generate (should raise exception after retries)
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai', max_retries=2, retry_delay=0.01)
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

        with pytest.raises(Exception) as exc_info:
            gen.generate(1)

        assert "Persistent API error" in str(exc_info.value)


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_save_load(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test save and load functionality."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Fit generator
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai', model='gpt-4o-mini', temperature=0.7)
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            gen.save(temp_path)
            assert os.path.exists(temp_path)

            # Load into new generator
            loaded_gen = PromptLLMGenerator()
            loaded_gen.load(temp_path)

            # Verify state was preserved
            assert loaded_gen.is_fitted == gen.is_fitted
            assert loaded_gen.text_columns == gen.text_columns
            assert loaded_gen.tabular_columns == gen.tabular_columns
            assert loaded_gen.provider == gen.provider
            assert loaded_gen.model == gen.model
            assert loaded_gen.temperature == gen.temperature

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def test_prompt_llm_missing_api_key():
    """Test that missing API key raises clear error."""
    with patch.dict(os.environ, {}, clear=True):
        gen = PromptLLMGenerator(provider='openai')

        with pytest.raises(ValueError) as exc_info:
            gen.fit(
                pd.DataFrame({'text': ['test'], 'num': [1]}),
                text_columns=['text'],
                tabular_columns=['num']
            )

        assert "OPENAI_API_KEY" in str(exc_info.value)


def test_prompt_llm_invalid_provider():
    """Test that invalid provider raises error."""
    with pytest.raises(ValueError) as exc_info:
        PromptLLMGenerator(provider='invalid_provider')

    assert "provider must be" in str(exc_info.value).lower()


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_generate_before_fit(mock_ctgan, mock_openai_class, mock_openai_client):
    """Test that generate raises error if called before fit."""
    mock_openai_class.return_value = mock_openai_client

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai')

        with pytest.raises(ValueError) as exc_info:
            gen.generate(5)

        assert "must be fitted" in str(exc_info.value).lower()


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_multiple_text_columns(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test handling multiple text columns."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Create data with multiple text columns
    multi_text_data = pd.DataFrame({
        'loan_purpose': ['Buy seeds', 'Purchase livestock'],
        'business_description': ['Small farm', 'Dairy business'],
        'sector': ['Agriculture', 'Agriculture'],
        'loan_amount': [500, 1200]
    })

    # Mock CTGAN generate
    mock_tabular_df = pd.DataFrame({
        'sector': ['Retail', 'Services'],
        'loan_amount': [800, 400]
    })
    mock_ctgan_instance.sample.return_value = mock_tabular_df

    # Fit and generate
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai')
        gen.fit(
            multi_text_data,
            text_columns=['loan_purpose', 'business_description'],
            tabular_columns=['sector', 'loan_amount']
        )

        synthetic_df = gen.generate(2)

    # Verify output has both text columns
    assert 'loan_purpose' in synthetic_df.columns
    assert 'business_description' in synthetic_df.columns
    assert 'sector' in synthetic_df.columns
    assert 'loan_amount' in synthetic_df.columns

    # Verify LLM was called for both text columns (2 samples * 2 text columns = 4 calls)
    assert mock_openai_client.chat.completions.create.call_count == 4


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
def test_prompt_llm_numeric_formatting(mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test that numeric values are properly formatted in prompts."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Fit generator
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai')
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

    # Test with integer
    test_row_int = pd.Series({
        'sector': 'Agriculture',
        'loan_amount': 500,
        'term_in_months': 12
    })
    prompt_int = gen._build_prompt(test_row_int, 'loan_purpose')
    assert '500' in prompt_int
    assert '12' in prompt_int

    # Test with float
    test_row_float = pd.Series({
        'sector': 'Retail',
        'loan_amount': 1234.56,
        'term_in_months': 18
    })
    prompt_float = gen._build_prompt(test_row_float, 'loan_purpose')
    assert '1234.56' in prompt_float


@patch('generators.prompt_llm.OpenAI')
@patch('generators.prompt_llm.GaussianCopulaSynthesizer')
@patch('generators.prompt_llm.tqdm')
def test_prompt_llm_progress_tracking(mock_tqdm, mock_ctgan, mock_openai_class, sample_data, mock_openai_client):
    """Test that progress tracking is enabled when tqdm is available."""
    # Setup mocks
    mock_openai_class.return_value = mock_openai_client
    mock_ctgan_instance = MagicMock()
    mock_ctgan.return_value = mock_ctgan_instance

    # Mock tqdm
    mock_progress_bar = MagicMock()
    mock_tqdm.return_value = mock_progress_bar
    mock_progress_bar.__iter__ = lambda self: iter(range(3))

    # Mock CTGAN generate
    mock_tabular_df = pd.DataFrame({
        'sector': ['Agriculture', 'Retail', 'Services'],
        'loan_amount': [500, 800, 400],
        'term_in_months': [12, 18, 9]
    })
    mock_ctgan_instance.sample.return_value = mock_tabular_df

    # Fit and generate
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        gen = PromptLLMGenerator(provider='openai')
        gen.fit(
            sample_data,
            text_columns=['loan_purpose'],
            tabular_columns=['sector', 'loan_amount', 'term_in_months']
        )

        synthetic_df = gen.generate(3)

    # Verify tqdm was used
    mock_tqdm.assert_called_once()
    call_args = mock_tqdm.call_args
    assert call_args[1]['desc'] == "Generating text samples"
