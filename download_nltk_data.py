#!/usr/bin/env python3

"""
NLTK Data Download Script
解决NLTK数据下载的代理认证问题
"""

import os
import sys
import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data with proxy handling."""
    
    print("🔧 Setting up NLTK data...")
    
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
    
    # Required NLTK data packages
    required_packages = [
        'punkt',
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

def test_nltk_installation():
    """Test if NLTK is working properly."""
    print("\n🧪 Testing NLTK installation...")
    
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
        
        print("🎉 NLTK is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ NLTK test failed: {e}")
        return False

def main():
    """Main function to setup NLTK data."""
    print("🚀 NLTK Data Setup for SynEval")
    print("=" * 40)
    
    # Download NLTK data
    download_nltk_data()
    
    # Test installation
    success = test_nltk_installation()
    
    if success:
        print("\n✅ NLTK setup completed successfully!")
        print("You can now run SynEval without NLTK data issues.")
    else:
        print("\n⚠️ NLTK setup completed with warnings.")
        print("Some features may not work properly.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())