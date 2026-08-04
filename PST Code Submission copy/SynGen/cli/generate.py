"""Main CLI entry point for SynGen synthetic data generation."""
import click
import logging
from pathlib import Path
from typing import Optional, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use lazy imports to avoid dependency issues
def _import_generator(name):
    """Lazy import generators to avoid unnecessary dependencies."""
    if name == 'CTGANLLMStitcher':
        from generators.ctgan_llm_stitcher import CTGANLLMStitcher
        return CTGANLLMStitcher
    elif name == 'PromptLLMGenerator':
        from generators.prompt_llm import PromptLLMGenerator
        return PromptLLMGenerator
    elif name == 'MultimodalDiffusionGenerator':
        from generators.multimodal_diffusion import MultimodalDiffusionGenerator
        return MultimodalDiffusionGenerator
    elif name == 'TiltedGenerator':
        from generators.tilted import TiltedGenerator
        return TiltedGenerator
    else:
        raise ValueError(f"Unknown generator: {name}")

from cli.utils import (
    setup_logging,
    validate_columns,
    load_data,
    save_data,
    print_statistics
)
from config.loader import load_config

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """SynGen - Synthetic Multimodal Data Generation Framework."""
    pass


@cli.command(name='ctgan-llm')
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='Path to input CSV file')
@click.option('--output', '-o', 'output_path', required=True, type=click.Path(),
              help='Path to output CSV file')
@click.option('--text-column', '-t', 'text_columns', multiple=True, required=True,
              help='Text column name (can specify multiple)')
@click.option('--tabular-column', '-c', 'tabular_columns', multiple=True, required=True,
              help='Tabular column name (can specify multiple)')
@click.option('--n-samples', '-n', type=int, required=True,
              help='Number of synthetic samples to generate')
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), default='openai',
              help='LLM provider (default: openai)')
@click.option('--model', type=str, default='gpt-4o-mini',
              help='LLM model name (default: gpt-4o-mini)')
@click.option('--n-few-shot', type=int, default=3,
              help='Number of few-shot examples for LLM (default: 3)')
@click.option('--random-seed', type=int, default=None,
              help='Random seed for reproducibility')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True),
              help='Path to configuration file (overrides other options)')
def ctgan_llm_command(input_path, output_path, text_columns, tabular_columns, n_samples,
                     provider, model, n_few_shot, random_seed, verbose, config):
    """Generate synthetic data using CTGAN + LLM Stitcher method.

    This method uses CTGAN for tabular data and an LLM to generate conditioned text.
    """
    setup_logging(verbose)
    logger.info("Starting CTGAN + LLM Stitcher generation")

    # Load config if provided
    if config:
        config_dict = load_config(config)
        # Override with config values
        provider = config_dict.get('provider', provider)
        model = config_dict.get('model', model)
        n_few_shot = config_dict.get('n_few_shot', n_few_shot)
        random_seed = config_dict.get('random_seed', random_seed)

    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_data(input_path)
        logger.info(f"Loaded {len(df)} rows")

        # Validate columns
        text_cols = list(text_columns)
        tab_cols = list(tabular_columns)
        validate_columns(df, text_cols, tab_cols)

        # Initialize generator
        logger.info(f"Initializing CTGAN+LLM with provider={provider}, model={model}")
        CTGANLLMStitcher = _import_generator('CTGANLLMStitcher')
        generator = CTGANLLMStitcher(
            provider=provider,
            model=model,
            n_few_shot=n_few_shot,
            random_seed=random_seed
        )

        # Fit generator
        logger.info("Fitting generator on real data...")
        generator.fit(df, text_cols, tab_cols)

        # Generate synthetic data
        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_df = generator.generate(n_samples)

        # Save output
        logger.info(f"Saving synthetic data to {output_path}")
        save_data(synthetic_df, output_path)

        # Print statistics
        print_statistics(df, synthetic_df, text_cols, tab_cols)

        logger.info("Generation complete!")

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=verbose)
        raise click.ClickException(str(e))


@cli.command(name='prompt-llm')
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='Path to input CSV file')
@click.option('--output', '-o', 'output_path', required=True, type=click.Path(),
              help='Path to output CSV file')
@click.option('--text-column', '-t', 'text_columns', multiple=True, required=True,
              help='Text column name (can specify multiple)')
@click.option('--tabular-column', '-c', 'tabular_columns', multiple=True, required=True,
              help='Tabular column name (can specify multiple)')
@click.option('--n-samples', '-n', type=int, required=True,
              help='Number of synthetic samples to generate')
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), default='openai',
              help='LLM provider (default: openai)')
@click.option('--model', type=str, default='gpt-4o-mini',
              help='LLM model name (default: gpt-4o-mini)')
@click.option('--temperature', type=float, default=0.8,
              help='LLM temperature (default: 0.8)')
@click.option('--batch-size', type=int, default=10,
              help='Batch size for generation (default: 10)')
@click.option('--max-retries', type=int, default=3,
              help='Maximum retries for API calls (default: 3)')
@click.option('--random-seed', type=int, default=None,
              help='Random seed for reproducibility')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True),
              help='Path to configuration file (overrides other options)')
def prompt_llm_command(input_path, output_path, text_columns, tabular_columns, n_samples,
                      provider, model, temperature, batch_size, max_retries, random_seed, verbose, config):
    """Generate synthetic data using Prompt-Conditioned LLM method.

    This method uses an LLM to generate both text and tabular data jointly.
    """
    setup_logging(verbose)
    logger.info("Starting Prompt-Conditioned LLM generation")

    # Load config if provided
    if config:
        config_dict = load_config(config)
        provider = config_dict.get('provider', provider)
        model = config_dict.get('model', model)
        temperature = config_dict.get('temperature', temperature)
        batch_size = config_dict.get('batch_size', batch_size)
        max_retries = config_dict.get('max_retries', max_retries)
        random_seed = config_dict.get('random_seed', random_seed)

    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_data(input_path)
        logger.info(f"Loaded {len(df)} rows")

        # Validate columns
        text_cols = list(text_columns)
        tab_cols = list(tabular_columns)
        validate_columns(df, text_cols, tab_cols)

        # Initialize generator
        logger.info(f"Initializing PromptLLM with provider={provider}, model={model}")
        PromptLLMGenerator = _import_generator('PromptLLMGenerator')
        generator = PromptLLMGenerator(
            provider=provider,
            model=model,
            temperature=temperature,
            batch_size=batch_size,
            max_retries=max_retries,
            random_seed=random_seed
        )

        # Fit generator
        logger.info("Fitting generator on real data...")
        generator.fit(df, text_cols, tab_cols)

        # Generate synthetic data
        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_df = generator.generate(n_samples)

        # Save output
        logger.info(f"Saving synthetic data to {output_path}")
        save_data(synthetic_df, output_path)

        # Print statistics
        print_statistics(df, synthetic_df, text_cols, tab_cols)

        logger.info("Generation complete!")

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=verbose)
        raise click.ClickException(str(e))


@cli.command(name='diffusion')
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='Path to input CSV file')
@click.option('--output', '-o', 'output_path', required=True, type=click.Path(),
              help='Path to output CSV file')
@click.option('--text-column', '-t', 'text_columns', multiple=True, required=True,
              help='Text column name (can specify multiple)')
@click.option('--tabular-column', '-c', 'tabular_columns', multiple=True, required=True,
              help='Tabular column name (can specify multiple)')
@click.option('--n-samples', '-n', type=int, required=True,
              help='Number of synthetic samples to generate')
@click.option('--text-encoder', type=str, default='all-MiniLM-L6-v2',
              help='Sentence transformer model (default: all-MiniLM-L6-v2)')
@click.option('--latent-dim', type=int, default=128,
              help='Latent dimension (default: 128)')
@click.option('--hidden-dim', type=int, default=256,
              help='Hidden dimension (default: 256)')
@click.option('--n-diffusion-steps', type=int, default=50,
              help='Number of diffusion steps (default: 50)')
@click.option('--epochs', type=int, default=100,
              help='Number of training epochs (default: 100)')
@click.option('--batch-size', type=int, default=32,
              help='Batch size (default: 32)')
@click.option('--learning-rate', type=float, default=1e-3,
              help='Learning rate (default: 0.001)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True),
              help='Path to configuration file (overrides other options)')
def diffusion_command(input_path, output_path, text_columns, tabular_columns, n_samples,
                     text_encoder, latent_dim, hidden_dim, n_diffusion_steps, epochs,
                     batch_size, learning_rate, verbose, config):
    """Generate synthetic data using Multimodal Diffusion method.

    This method uses a diffusion model to generate jointly consistent text and tabular data.
    """
    setup_logging(verbose)
    logger.info("Starting Multimodal Diffusion generation")

    # Load config if provided
    if config:
        config_dict = load_config(config)
        text_encoder = config_dict.get('text_encoder_model', text_encoder)
        latent_dim = config_dict.get('latent_dim', latent_dim)
        hidden_dim = config_dict.get('hidden_dim', hidden_dim)
        n_diffusion_steps = config_dict.get('n_diffusion_steps', n_diffusion_steps)
        epochs = config_dict.get('n_epochs', epochs)
        batch_size = config_dict.get('batch_size', batch_size)
        learning_rate = config_dict.get('learning_rate', learning_rate)

    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_data(input_path)
        logger.info(f"Loaded {len(df)} rows")

        # Validate columns
        text_cols = list(text_columns)
        tab_cols = list(tabular_columns)
        validate_columns(df, text_cols, tab_cols)

        # Initialize generator
        logger.info(f"Initializing Diffusion model with latent_dim={latent_dim}")
        MultimodalDiffusionGenerator = _import_generator('MultimodalDiffusionGenerator')
        generator = MultimodalDiffusionGenerator(
            text_encoder_model=text_encoder,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_diffusion_steps=n_diffusion_steps,
            learning_rate=learning_rate,
            n_epochs=epochs,
            batch_size=batch_size
        )

        # Fit generator
        logger.info(f"Training diffusion model for {epochs} epochs...")
        generator.fit(df, text_cols, tab_cols)

        # Generate synthetic data
        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_df = generator.generate(n_samples)

        # Save output
        logger.info(f"Saving synthetic data to {output_path}")
        save_data(synthetic_df, output_path)

        # Print statistics
        print_statistics(df, synthetic_df, text_cols, tab_cols)

        logger.info("Generation complete!")

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=verbose)
        raise click.ClickException(str(e))


@cli.command(name='tilted')
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='Path to input CSV file')
@click.option('--output', '-o', 'output_path', required=True, type=click.Path(),
              help='Path to output CSV file')
@click.option('--text-column', '-t', 'text_columns', multiple=True, required=True,
              help='Text column name (can specify multiple)')
@click.option('--tabular-column', '-c', 'tabular_columns', multiple=True, required=True,
              help='Tabular column name (can specify multiple)')
@click.option('--n-samples', '-n', type=int, required=True,
              help='Number of synthetic samples to generate')
@click.option('--shuffle-strategy', type=click.Choice(['random', 'stratified', 'adversarial']),
              default='random', help='Shuffle strategy (default: random)')
@click.option('--random-seed', type=int, default=42,
              help='Random seed (default: 42)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True),
              help='Path to configuration file (overrides other options)')
def tilted_command(input_path, output_path, text_columns, tabular_columns, n_samples,
                  shuffle_strategy, random_seed, verbose, config):
    """Generate synthetic data using Tilted (adversarial baseline) method.

    This method creates mismatched text-tabular pairs as a baseline for evaluation.
    """
    setup_logging(verbose)
    logger.info("Starting Tilted (adversarial baseline) generation")

    # Load config if provided
    if config:
        config_dict = load_config(config)
        shuffle_strategy = config_dict.get('shuffle_strategy', shuffle_strategy)
        random_seed = config_dict.get('random_state', random_seed)

    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_data(input_path)
        logger.info(f"Loaded {len(df)} rows")

        # Validate columns
        text_cols = list(text_columns)
        tab_cols = list(tabular_columns)
        validate_columns(df, text_cols, tab_cols)

        # Initialize generator
        logger.info(f"Initializing Tilted with strategy={shuffle_strategy}")
        TiltedGenerator = _import_generator('TiltedGenerator')
        generator = TiltedGenerator(
            shuffle_strategy=shuffle_strategy,
            random_state=random_seed
        )

        # Fit generator
        logger.info("Fitting generator on real data...")
        generator.fit(df, text_cols, tab_cols)

        # Generate synthetic data
        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_df = generator.generate(n_samples)

        # Save output
        logger.info(f"Saving synthetic data to {output_path}")
        save_data(synthetic_df, output_path)

        # Print statistics
        print_statistics(df, synthetic_df, text_cols, tab_cols)

        logger.info("Generation complete!")

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=verbose)
        raise click.ClickException(str(e))


def main():
    """Entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
