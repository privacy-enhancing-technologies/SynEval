"""Configuration file loading and validation."""
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from config.defaults import get_default_config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary of configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported or config is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load based on file extension
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {path.suffix}. "
            "Use .yaml, .yml, or .json"
        )

    # Validate config
    validate_config(config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check for required method field
    if 'method' not in config:
        raise ValueError("Config must specify 'method' field")

    method = config['method']
    valid_methods = ['ctgan-llm', 'prompt-llm', 'diffusion', 'tilted']

    if method not in valid_methods:
        raise ValueError(
            f"Invalid method: {method}. "
            f"Valid methods: {valid_methods}"
        )

    # Get default config to check for unknown fields
    defaults = get_default_config(method)

    # Warn about unknown fields (but don't error)
    unknown = set(config.keys()) - set(defaults.keys())
    if unknown:
        import warnings
        warnings.warn(
            f"Unknown configuration fields for {method}: {unknown}. "
            "These will be ignored."
        )

    # Validate specific field types
    if method in ['ctgan-llm', 'prompt-llm']:
        if 'provider' in config and config['provider'] not in ['openai', 'anthropic']:
            raise ValueError(
                f"Invalid provider: {config['provider']}. "
                "Must be 'openai' or 'anthropic'"
            )

        if 'model' in config and not isinstance(config['model'], str):
            raise ValueError("model must be a string")

    if method == 'tilted':
        if 'shuffle_strategy' in config:
            valid_strategies = ['random', 'stratified', 'adversarial']
            if config['shuffle_strategy'] not in valid_strategies:
                raise ValueError(
                    f"Invalid shuffle_strategy: {config['shuffle_strategy']}. "
                    f"Must be one of {valid_strategies}"
                )

    # Validate numeric fields
    numeric_fields = {
        'n_few_shot': int,
        'temperature': float,
        'batch_size': int,
        'max_retries': int,
        'latent_dim': int,
        'hidden_dim': int,
        'n_diffusion_steps': int,
        'n_epochs': int,
        'learning_rate': float,
        'random_seed': (int, type(None)),
        'random_state': int
    }

    for field, expected_type in numeric_fields.items():
        if field in config:
            if not isinstance(config[field], expected_type):
                raise ValueError(
                    f"{field} must be of type {expected_type.__name__}, "
                    f"got {type(config[field]).__name__}"
                )


def merge_config_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user config with defaults.

    Args:
        config: User configuration dictionary

    Returns:
        Complete configuration with defaults filled in
    """
    method = config['method']
    defaults = get_default_config(method)

    # Merge: user config overrides defaults
    merged = defaults.copy()
    merged.update(config)

    return merged
