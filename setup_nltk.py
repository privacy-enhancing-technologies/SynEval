#!/usr/bin/env python3
"""
Simple NLTK setup script for SynEval
"""
import os
import ssl

import nltk


def main():
    print("🔧 Setting up NLTK data...")

    # Create NLTK data directory
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download required NLTK data packages
    required_packages = [
        "punkt",
        "averaged_perceptron_tagger",
        "maxent_ne_chunker",
        "words",
        "stopwords",
    ]
    print("📦 Downloading NLTK data packages...")

    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
            print(f"✅ {package} downloaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to download {package}: {e}")

    # Create plots directory
    os.makedirs("./plots", exist_ok=True)
    print("📁 Created plots directory")
    print("🎉 Setup completed!")


if __name__ == "__main__":
    main()
