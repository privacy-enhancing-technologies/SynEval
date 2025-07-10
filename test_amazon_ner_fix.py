#!/usr/bin/env python3
"""
Test script to clear cache and test Amazon Fashion NER analysis with device fixes.
"""

import os
import shutil
from pathlib import Path

def clear_cache():
    """Clear all cache files to force fresh model loading."""
    print("Clearing cache files...")
    
    # Clear model cache
    cache_files = [
        './cache/amazon_fashion_ner_fast_results.json',
        './cache/privacy_entities_detected.json',
        './cache/flair_ner_results.json'
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed: {cache_file}")
    
    # Clear reports directory
    reports_dir = Path('./reports')
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
        print("Removed reports directory")
    
    print("Cache cleared successfully!")

def test_amazon_ner():
    """Test the Amazon Fashion NER analysis."""
    print("\nTesting Amazon Fashion NER analysis...")
    
    try:
        from amazon_fashion_ner_analysis_fast import FastAmazonFashionNERAnalyzer
        
        # Initialize analyzer
        analyzer = FastAmazonFashionNERAnalyzer()
        
        # Test with a small sample
        print("Running analysis with 100 samples...")
        results = analyzer.analyze_dataset(
            csv_file='real_10k.csv',
            text_column='text',
            sample_size=100
        )
        
        print("✅ Analysis completed successfully!")
        print(f"Found {results['total_entities']} entities")
        print(f"Average entity density: {results['avg_entity_density']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("="*60)
    print("AMAZON FASHION NER DEVICE FIX TEST")
    print("="*60)
    
    # Clear cache
    clear_cache()
    
    # Test analysis
    success = test_amazon_ner()
    
    if success:
        print("\n" + "="*60)
        print("✅ DEVICE FIX TEST PASSED")
        print("="*60)
        print("The Amazon Fashion NER analysis is now working correctly!")
    else:
        print("\n" + "="*60)
        print("❌ DEVICE FIX TEST FAILED")
        print("="*60)
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 