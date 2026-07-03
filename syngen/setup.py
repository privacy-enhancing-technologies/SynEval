"""Setup script for SynGen package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="syngen",
    version="0.1.0",
    author="SynGen Team",
    description="Synthetic Multimodal Data Generation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "tqdm>=4.65.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "sdv>=1.0.0"  # For CTGAN
    ],
    entry_points={
        "console_scripts": [
            "syngen=cli.generate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
