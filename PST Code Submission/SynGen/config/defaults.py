"""Default configurations for each generator."""

CTGAN_LLM_DEFAULTS = {
    "method": "ctgan-llm",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "n_few_shot": 3,
    "random_seed": None
}

PROMPT_LLM_DEFAULTS = {
    "method": "prompt-llm",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.8,
    "batch_size": 10,
    "max_retries": 3,
    "retry_delay": 1.0,
    "random_seed": None
}

DIFFUSION_DEFAULTS = {
    "method": "diffusion",
    "text_encoder_model": "all-MiniLM-L6-v2",
    "latent_dim": 128,
    "hidden_dim": 256,
    "n_diffusion_steps": 50,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "learning_rate": 1e-3,
    "n_epochs": 100,
    "batch_size": 32
}

TILTED_DEFAULTS = {
    "method": "tilted",
    "shuffle_strategy": "random",
    "random_state": 42
}


def get_default_config(method: str) -> dict:
    """
    Get default configuration for a specific method.

    Args:
        method: Name of the generation method

    Returns:
        Dictionary of default configuration values

    Raises:
        ValueError: If method is not recognized
    """
    defaults = {
        "ctgan-llm": CTGAN_LLM_DEFAULTS,
        "prompt-llm": PROMPT_LLM_DEFAULTS,
        "diffusion": DIFFUSION_DEFAULTS,
        "tilted": TILTED_DEFAULTS
    }

    if method not in defaults:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid methods: {list(defaults.keys())}"
        )

    return defaults[method].copy()
