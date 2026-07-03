"""
Synthetic data generators for multimodal data generation.

Uses lazy imports to avoid unnecessary dependencies when only using specific generators.
"""
from generators.base import BaseGenerator

def __getattr__(name):
    """Lazy import generators to avoid loading unnecessary dependencies."""
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
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BaseGenerator',
    'CTGANLLMStitcher',
    'PromptLLMGenerator',
    'MultimodalDiffusionGenerator',
    'TiltedGenerator'
]
