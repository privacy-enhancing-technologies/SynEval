"""Configuration management for SynGen."""
from config.loader import load_config
from config.defaults import get_default_config

__all__ = ['load_config', 'get_default_config']
