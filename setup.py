#!/usr/bin/env python3
"""
Setup script for SynEval package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="syneval",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive evaluation framework for synthetic data quality assessment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yefyuan/SynEval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "syneval=syneval.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "syneval": ["*.txt", "*.md", "*.json"],
    },
    keywords="synthetic data, evaluation, machine learning, privacy, fidelity, utility, diversity",
    project_urls={
        "Bug Reports": "https://github.com/yefyuan/SynEval/issues",
        "Source": "https://github.com/yefyuan/SynEval",
        "Documentation": "https://github.com/yefyuan/SynEval#readme",
    },
) 