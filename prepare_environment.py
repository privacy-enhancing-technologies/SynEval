#!/usr/bin/env python3

"""
SynEval Environment Preparation Script
Complete environment preparation for SynEval framework
"""

import os
import sys
import subprocess
import ssl

def install_requirements():
    """Install required Python packages from requirements.txt."""
    print("📦 Installing Python dependencies...")
    
    try:
        # First, upgrade pip to latest version
        print("  Upgrading pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        # Force upgrade critical packages that are causing conflicts
        print("  Upgrading critical packages (numpy, scipy, gensim)...")
        critical_packages = [
            "numpy>=1.26.0", 
            "scipy>=1.11.0,<1.16.0",  # Constrain scipy version for gensim compatibility
            "gensim>=4.3.1"  # Upgrade gensim to fix scipy compatibility
        ]
        for package in critical_packages:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "--upgrade", "--force-reinstall", package
                ])
                print(f"    ✅ {package} upgraded successfully")
            except subprocess.CalledProcessError as e:
                print(f"    ⚠️ Failed to upgrade {package}: {e}")
        
        # Install requirements with upgrade to resolve conflicts
        print("  Installing remaining dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        
        # Note about dependency conflicts
        print("\n💡 Note: You may see dependency conflict warnings above.")
        print("   This is normal in environments like Google Colab or when other packages are already installed.")
        print("   These conflicts won't affect SynEval functionality.")
        print("   If you want to avoid conflicts completely, consider using a fresh virtual environment.")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running: pip install --upgrade pip")
        return False

def download_nltk_data():
    """Download required NLTK data with proxy handling."""
    
    print("🔧 Setting up NLTK data...")
    
    # Import nltk after installing dependencies
    try:
        import nltk
    except ImportError:
        print("❌ NLTK not found. Please install dependencies first.")
        return False
    
    # Create NLTK data directory
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Required NLTK data packages (including punkt_tab)
    required_packages = [
        'punkt',
        'punkt_tab',  # Added this missing package
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'stopwords'
    ]
    
    print("📦 Downloading NLTK data packages...")
    
    for package in required_packages:
        try:
            print(f"  Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"  ✅ {package} downloaded successfully")
        except Exception as e:
            print(f"  ⚠️ Failed to download {package}: {e}")
            print(f"  💡 You can manually download it later or continue without it")
    
    print("\n🎉 NLTK data setup completed!")
    print(f"📁 Data location: {nltk_data_dir}")
    return True

def test_installation():
    """Test if the installation is working properly."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test basic NLTK functionality
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        
        # Test sentence tokenization
        text = "Hello world! This is a test sentence. How are you?"
        sentences = sent_tokenize(text)
        print(f"✅ Sentence tokenization: {sentences}")
        
        # Test word tokenization
        words = word_tokenize(text)
        print(f"✅ Word tokenization: {words[:5]}...")
        
        # Test stopwords
        stop_words = set(stopwords.words('english'))
        print(f"✅ Stopwords loaded: {len(stop_words)} words")
        
        # Test other key dependencies
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Core dependencies (pandas, numpy, matplotlib, seaborn) working")
        
        print("🎉 Installation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def create_plots_directory():
    """Create plots directory for output."""
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"📁 Created plots directory: {plots_dir}")

def main():
    """Main function to prepare SynEval environment."""
    print("🚀 SynEval Environment Preparation")
    print("=" * 40)
    
    # Install dependencies
    if not install_requirements():
        print("❌ Preparation failed at dependency installation step.")
        return 1
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Preparation failed at NLTK data download step.")
        return 1
    
    # Create necessary directories
    create_plots_directory()
    
    # Test installation
    success = test_installation()
    
    if success:
        print("\n✅ SynEval environment preparation completed successfully!")
        print("You can now run SynEval without any issues.")
        print("\n📝 Next steps:")
        print("  1. Run: python run.py --help")
        print("  2. Check the README.md for usage examples")
        print("  3. Example command:")
        print("     python run.py --synthetic data.csv --original data.csv --metadata metadata.json --dimension fidelity --plot")
    else:
        print("\n⚠️ Preparation completed with warnings.")
        print("Some features may not work properly.")
        print("💡 Try running the preparation script again or check the error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 