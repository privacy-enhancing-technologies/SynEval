#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.metadata import SingleTableMetadata
import logging
import os
import hashlib
import json
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU acceleration setup
def setup_gpu_acceleration():
    """Setup GPU acceleration if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

# Initialize device as None - will be set when evaluator is created
DEVICE = None

def compute_dataset_fingerprint(data: pd.DataFrame, metadata: Dict) -> str:
    """
    Compute a unique fingerprint for the dataset to identify changes.
    
    Args:
        data: DataFrame to fingerprint
        metadata: Metadata dictionary
    
    Returns:
        str: SHA256 hash of dataset characteristics
    """
    # Create a fingerprint based on data characteristics
    fingerprint_data = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
        'metadata_columns': list(metadata.get('columns', {}).keys()),
        'sample_hash': hashlib.sha256(
            data.head(1000).to_string().encode()
        ).hexdigest()[:16],  # First 1000 rows hash
        'total_hash': hashlib.sha256(
            data.to_string().encode()
        ).hexdigest()[:16]   # Full dataset hash
    }
    
    # Convert to string and hash
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()

class SmartCache:
    """Smart cache system with dataset fingerprinting."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.fingerprint_file = self.cache_dir / "dataset_fingerprints.json"
        self.fingerprints = self._load_fingerprints()
    
    def _load_fingerprints(self) -> Dict:
        """Load dataset fingerprints."""
        if self.fingerprint_file.exists():
            try:
                with open(self.fingerprint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Error loading fingerprints: {str(e)}")
        return {}
    
    def _save_fingerprints(self):
        """Save dataset fingerprints."""
        try:
            with open(self.fingerprint_file, 'w') as f:
                json.dump(self.fingerprints, f, indent=2)
        except Exception as e:
            logging.warning(f"Error saving fingerprints: {str(e)}")
    
    def get_cache_path(self, cache_type: str, column: str = None) -> Path:
        """Get cache file path."""
        if column:
            return self.cache_dir / f"{cache_type}_{column}.pkl"
        return self.cache_dir / f"{cache_type}.pkl"
    
    def is_valid_cache(self, cache_type: str, dataset_fingerprint: str, column: str = None) -> bool:
        """Check if cache is valid for current dataset."""
        cache_key = f"{cache_type}_{column}" if column else cache_type
        return self.fingerprints.get(cache_key) == dataset_fingerprint
    
    def load_cache(self, cache_type: str, dataset_fingerprint: str, column: str = None) -> Optional[Dict]:
        """Load cache if valid."""
        if not self.is_valid_cache(cache_type, dataset_fingerprint, column):
            return None
        
        cache_path = self.get_cache_path(cache_type, column)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Error loading cache {cache_type}: {str(e)}")
        return None
    
    def save_cache(self, cache_type: str, dataset_fingerprint: str, data: Dict, column: str = None):
        """Save cache with fingerprint."""
        cache_path = self.get_cache_path(cache_type, column)
        cache_key = f"{cache_type}_{column}" if column else cache_type
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update fingerprint
            self.fingerprints[cache_key] = dataset_fingerprint
            self._save_fingerprints()
            
            logging.info(f"Saved cache for {cache_type}")
        except Exception as e:
            logging.warning(f"Error saving cache {cache_type}: {str(e)}")

class FidelityEvaluator:
    def __init__(self, 
                 synthetic_data: pd.DataFrame,
                 original_data: pd.DataFrame,
                 metadata: Dict,
                 selected_metrics: List[str] = None):
        """
        Initialize the fidelity evaluator.
        
        Args:
            synthetic_data: DataFrame containing synthetic data
            original_data: DataFrame containing original data
            metadata: Dictionary containing metadata about the data
        """
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.metadata = metadata
        self.text_columns = self._get_text_columns()
        self.logger = logging.getLogger(__name__)
        
        # Available metrics for selection
        self.available_metrics = [
            'diagnostic', 'quality', 'text', 'numerical_statistics'
        ]
        
        # Use all metrics if none specified
        self.selected_metrics = selected_metrics if selected_metrics else self.available_metrics
        
        # Initialize smart cache
        self.cache = SmartCache('./cache')
        
        # Compute dataset fingerprints
        self.original_fingerprint = compute_dataset_fingerprint(original_data, metadata)
        self.synthetic_fingerprint = compute_dataset_fingerprint(synthetic_data, metadata)
        
        # Setup GPU acceleration only when evaluator is created
        global DEVICE
        if DEVICE is None:
            DEVICE = setup_gpu_acceleration()
            if DEVICE == 'cuda':
                self.logger.info("GPU acceleration available - using CUDA")
            else:
                self.logger.info("GPU not available - using CPU")
        
    def _get_text_columns(self) -> List[str]:
        """Get list of text columns from metadata."""
        return [col for col, info in self.metadata['columns'].items() 
                if info['sdtype'] == 'text']
    
    def _prepare_sdv_metadata(self) -> SingleTableMetadata:
        """Convert dictionary metadata to SDV SingleTableMetadata format."""
        sdv_metadata = SingleTableMetadata()
        
        # Add columns with their types
        for col, info in self.metadata['columns'].items():
            # Skip _id column
            if col == '_id':
                continue
                
            sdtype = info['sdtype']
            if sdtype == 'text':
                sdv_metadata.add_column(col, sdtype='text')
            elif sdtype == 'categorical':
                sdv_metadata.add_column(col, sdtype='categorical')
            elif sdtype == 'numerical':
                sdv_metadata.add_column(col, sdtype='numerical')
            elif sdtype == 'datetime':
                sdv_metadata.add_column(col, sdtype='datetime')
            elif sdtype == 'boolean':
                sdv_metadata.add_column(col, sdtype='boolean')
        
        # Add primary key if specified
        if 'primary_key' in self.metadata:
            sdv_metadata.set_primary_key(self.metadata['primary_key'])
            
        return sdv_metadata
    
    def evaluate(self) -> Dict:
        """
        Evaluate the fidelity of synthetic data using multiple metrics.
        
        Returns:
            Dictionary containing fidelity metrics
        """
        results = {}

        # Drop _id if exists
        self.synthetic_data = self.synthetic_data.drop(columns=['_id'], errors='ignore')
        self.original_data = self.original_data.drop(columns=['_id'], errors='ignore')

        # Prepare metadata
        try:
            sdv_metadata = self._prepare_sdv_metadata()
        except Exception as e:
            logger.error(f"Error preparing SDV metadata: {str(e)}")
            sdv_metadata = None

        # Diagnostic
        if 'diagnostic' in self.selected_metrics and sdv_metadata:
            try:
                logger.info("Running SDV diagnostic evaluation...")
                diagnostic_report = run_diagnostic(
                    real_data=self.original_data,
                    synthetic_data=self.synthetic_data,
                    metadata=sdv_metadata
                )

                results['diagnostic'] = {
                    'Data Validity': round(diagnostic_report.get_details('Data Validity')["Score"].mean(), 4),
                    'Data Structure': round(diagnostic_report.get_details('Data Structure')["Score"].mean(), 4),
                    'Overall': {
                        'score': round(diagnostic_report.get_score(), 4)
                    }
                }

                logger.info(f"Diagnostic evaluation completed. Overall score: {results['diagnostic']['Overall']['score']}")
            except Exception as e:
                logger.warning(f"Failed to extract diagnostic scores: {e}")
                results['diagnostic'] = {
                    'Data Validity': None,
                    'Data Structure': None,
                    'Overall': {
                        'score': round(diagnostic_report.get_score(), 4) if 'diagnostic_report' in locals() else None
                    }
                }
        elif 'diagnostic' in self.selected_metrics:
            results['diagnostic'] = None

        # Quality
        if 'quality' in self.selected_metrics and sdv_metadata:
            try:
                logger.info("Running SDV quality evaluation...")
                quality_report = evaluate_quality(
                    real_data=self.original_data,
                    synthetic_data=self.synthetic_data,
                    metadata=sdv_metadata
                )

                results['quality'] = {
                    'Column Shapes': round(quality_report.get_details('Column Shapes')["Score"].mean(), 4),
                    'Column Pair Trends': round(quality_report.get_details('Column Pair Trends')["Score"].mean(), 4),
                    'Overall': {
                        'score': round(quality_report.get_score(), 4)
                    }
                }

                logger.info(f"Quality evaluation completed. Overall score: {results['quality']['Overall']['score']}")
            except Exception as e:
                logger.warning(f"Failed to extract quality scores: {e}")
                results['quality'] = {
                    'Column Shapes': None,
                    'Column Pair Trends': None,
                    'Overall': {
                        'score': round(quality_report.get_score(), 4) if 'quality_report' in locals() else None
                    }
                }
        elif 'quality' in self.selected_metrics:
            results['quality'] = None

        # Text
        if 'text' in self.selected_metrics and self.text_columns:
            results['text'] = self._evaluate_text(self.text_columns)

        # Numerical statistics
        if 'numerical_statistics' in self.selected_metrics:
            results['numerical_statistics'] = self._evaluate_numerical_statistics()

        return results

    def _evaluate_text(self, columns: list) -> Dict:
        """
        Evaluate text columns using basic statistics, keyword analysis, and sentiment analysis.
        """
        results = {}
        
        # Get text columns only
        text_cols = [col for col, info in self.metadata['columns'].items() 
                    if info['sdtype'] == 'text']
        
        for col in text_cols:
            col_results = {}
            
            # Basic text statistics
            original_lengths = self.original_data[col].str.len()
            synthetic_lengths = self.synthetic_data[col].str.len()
            
            col_results['length_stats'] = {
                'original_mean': original_lengths.mean(),
                'original_std': original_lengths.std(),
                'synthetic_mean': synthetic_lengths.mean(),
                'synthetic_std': synthetic_lengths.std()
            }
            
            # Word count statistics
            original_word_counts = self.original_data[col].str.split().str.len()
            synthetic_word_counts = self.synthetic_data[col].str.split().str.len()
            
            col_results['word_count_stats'] = {
                'original_mean': original_word_counts.mean(),
                'original_std': original_word_counts.std(),
                'synthetic_mean': synthetic_word_counts.mean(),
                'synthetic_std': synthetic_word_counts.std()
            }
            
            # Keyword analysis using TF-IDF with GPU acceleration
            try:
                vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
                original_tfidf = vectorizer.fit_transform(self.original_data[col].fillna(''))
                synthetic_tfidf = vectorizer.transform(self.synthetic_data[col].fillna(''))
                
                # Convert to GPU tensors if available for faster computation
                if DEVICE == 'cuda':
                    import torch
                    original_scores = torch.tensor(original_tfidf.mean(axis=0).A1, device='cuda')
                    synthetic_scores = torch.tensor(synthetic_tfidf.mean(axis=0).A1, device='cuda')
                    # Convert back to CPU for further processing
                    original_scores = original_scores.cpu().numpy()
                    synthetic_scores = synthetic_scores.cpu().numpy()
                else:
                    original_scores = original_tfidf.mean(axis=0).A1
                    synthetic_scores = synthetic_tfidf.mean(axis=0).A1
                
                # Get top keywords for both datasets
                original_keywords = vectorizer.get_feature_names_out()
                
                # Sort keywords by TF-IDF scores
                original_keyword_scores = sorted(zip(original_keywords, original_scores), 
                                              key=lambda x: x[1], reverse=True)
                synthetic_keyword_scores = sorted(zip(original_keywords, synthetic_scores), 
                                               key=lambda x: x[1], reverse=True)
                
                col_results['keyword_analysis'] = {
                    'original_top_keywords': dict(original_keyword_scores[:10]),
                    'synthetic_top_keywords': dict(synthetic_keyword_scores[:10])
                }
            except Exception as e:
                logger.error(f"Error in keyword analysis for {col}: {str(e)}")
                col_results['keyword_analysis'] = None
            
            # Sentiment analysis
            try:
                original_sentiments = [TextBlob(str(text)).sentiment.polarity 
                                    for text in self.original_data[col].fillna('')]
                synthetic_sentiments = [TextBlob(str(text)).sentiment.polarity 
                                      for text in self.synthetic_data[col].fillna('')]
                
                # Calculate sentiment percentages
                def get_sentiment_percentages(sentiments):
                    total = len(sentiments)
                    negative = sum(1 for s in sentiments if s < -0.1) / total * 100
                    neutral = sum(1 for s in sentiments if -0.1 <= s <= 0.1) / total * 100
                    positive = sum(1 for s in sentiments if s > 0.1) / total * 100
                    return {
                        'negative': negative,
                        'neutral': neutral,
                        'positive': positive
                    }
                
                col_results['sentiment_analysis'] = {
                    'original_mean': np.mean(original_sentiments),
                    'original_std': np.std(original_sentiments),
                    'synthetic_mean': np.mean(synthetic_sentiments),
                    'synthetic_std': np.std(synthetic_sentiments),
                    'original_sentiment_distribution': get_sentiment_percentages(original_sentiments),
                    'synthetic_sentiment_distribution': get_sentiment_percentages(synthetic_sentiments)
                }
            except Exception as e:
                logger.error(f"Error in sentiment analysis for {col}: {str(e)}")
                col_results['sentiment_analysis'] = None
            
            results[col] = col_results
            
        return results

    def _evaluate_numerical_statistics(self) -> Dict:
        """
        Evaluate numerical columns using comprehensive statistical analysis.
        
        Returns:
            Dictionary containing statistical comparison metrics for numerical columns
        """
        results = {}
        
        # Get numerical columns
        numerical_cols = [col for col, info in self.metadata['columns'].items() 
                         if info['sdtype'] == 'numerical']
        
        if not numerical_cols:
            return results
            
        for col in numerical_cols:
            try:
                # Get data for the column
                orig_data = self.original_data[col].dropna()
                syn_data = self.synthetic_data[col].dropna()
                
                if len(orig_data) == 0 or len(syn_data) == 0:
                    continue
                
                # Basic statistics
                orig_stats = {
                    'count': len(orig_data),
                    'mean': orig_data.mean(),
                    'median': orig_data.median(),
                    'std': orig_data.std(),
                    'min': orig_data.min(),
                    'max': orig_data.max(),
                    'q25': orig_data.quantile(0.25),
                    'q75': orig_data.quantile(0.75),
                    'skewness': orig_data.skew(),
                    'kurtosis': orig_data.kurtosis()
                }
                
                syn_stats = {
                    'count': len(syn_data),
                    'mean': syn_data.mean(),
                    'median': syn_data.median(),
                    'std': syn_data.std(),
                    'min': syn_data.min(),
                    'max': syn_data.max(),
                    'q25': syn_data.quantile(0.25),
                    'q75': syn_data.quantile(0.75),
                    'skewness': syn_data.skew(),
                    'kurtosis': syn_data.kurtosis()
                }
                
                # Calculate relative differences
                def calc_relative_diff(orig_val, syn_val):
                    if orig_val == 0:
                        return abs(syn_val) if syn_val != 0 else 0
                    return abs(syn_val - orig_val) / abs(orig_val)
                
                relative_diffs = {
                    'mean_diff': calc_relative_diff(orig_stats['mean'], syn_stats['mean']),
                    'median_diff': calc_relative_diff(orig_stats['median'], syn_stats['median']),
                    'std_diff': calc_relative_diff(orig_stats['std'], syn_stats['std']),
                    'skewness_diff': abs(orig_stats['skewness'] - syn_stats['skewness']),
                    'kurtosis_diff': abs(orig_stats['kurtosis'] - syn_stats['kurtosis'])
                }
                
                # Calculate range coverage
                orig_range = orig_stats['max'] - orig_stats['min']
                if orig_range > 0:
                    overlap_min = max(orig_stats['min'], syn_stats['min'])
                    overlap_max = min(orig_stats['max'], syn_stats['max'])
                    overlap_range = max(0, overlap_max - overlap_min)
                    range_coverage = overlap_range / orig_range
                else:
                    range_coverage = 0
                
                # Calculate distribution similarity using histogram comparison with GPU acceleration
                bins = min(50, len(orig_data.unique()), len(syn_data.unique()))
                if bins > 1:
                    if DEVICE == 'cuda':
                        import torch
                        # Convert to GPU tensors for faster computation
                        orig_tensor = torch.tensor(orig_data.values, device='cuda', dtype=torch.float32)
                        syn_tensor = torch.tensor(syn_data.values, device='cuda', dtype=torch.float32)
                        
                        # Calculate histograms on GPU
                        orig_hist = torch.histc(orig_tensor, bins=bins, min=orig_tensor.min(), max=orig_tensor.max())
                        syn_hist = torch.histc(syn_tensor, bins=bins, min=syn_tensor.min(), max=syn_tensor.max())
                        
                        # Normalize histograms
                        orig_hist = orig_hist / (orig_hist.sum() + 1e-10)
                        syn_hist = syn_hist / (syn_hist.sum() + 1e-10)
                        
                        # Calculate KL divergence on GPU
                        kl_div = torch.sum(orig_hist * torch.log((orig_hist + 1e-10) / (syn_hist + 1e-10)))
                        
                        # Calculate histogram intersection similarity on GPU
                        intersection = torch.minimum(orig_hist, syn_hist)
                        histogram_similarity = torch.sum(intersection)
                        
                        # Convert back to CPU
                        kl_div = kl_div.cpu().numpy()
                        histogram_similarity = histogram_similarity.cpu().numpy()
                    else:
                        # CPU fallback
                        orig_hist, _ = np.histogram(orig_data, bins=bins, density=True)
                        syn_hist, _ = np.histogram(syn_data, bins=bins, density=True)
                        
                        # Normalize histograms
                        orig_hist = orig_hist / (orig_hist.sum() + 1e-10)
                        syn_hist = syn_hist / (syn_hist.sum() + 1e-10)
                        
                        # Calculate KL divergence
                        kl_div = np.sum(orig_hist * np.log((orig_hist + 1e-10) / (syn_hist + 1e-10)))
                        
                        # Calculate histogram intersection similarity
                        intersection = np.minimum(orig_hist, syn_hist)
                        histogram_similarity = np.sum(intersection)
                else:
                    kl_div = float('inf')
                    histogram_similarity = 0
                
                # Calculate overall fidelity score for this column
                # Combine multiple metrics into a single score
                fidelity_components = [
                    1 - min(relative_diffs['mean_diff'], 1),  # Mean preservation
                    1 - min(relative_diffs['std_diff'], 1),   # Standard deviation preservation
                    1 - min(relative_diffs['skewness_diff'] / 2, 1),  # Skewness preservation
                    range_coverage,  # Range coverage
                    histogram_similarity  # Distribution similarity
                ]
                
                overall_fidelity = np.mean(fidelity_components)
                
                results[col] = {
                    'original_statistics': orig_stats,
                    'synthetic_statistics': syn_stats,
                    'relative_differences': relative_diffs,
                    'range_coverage': range_coverage,
                    'distribution_similarity': {
                        'kl_divergence': kl_div,
                        'histogram_similarity': histogram_similarity
                    },
                    'overall_fidelity_score': overall_fidelity,
                    'fidelity_interpretation': self._interpret_fidelity_score(overall_fidelity)
                }
                
            except Exception as e:
                logger.error(f"Error evaluating numerical statistics for column {col}: {str(e)}")
                results[col] = {
                    'error': str(e)
                }
        
        return results
    
    def _interpret_fidelity_score(self, score: float) -> str:
        """
        Interpret the fidelity score and provide a description.
        
        Args:
            score: Fidelity score between 0 and 1
            
        Returns:
            String interpretation of the score
        """
        if score >= 0.9:
            return "Excellent fidelity - synthetic data closely matches original data distribution"
        elif score >= 0.8:
            return "Good fidelity - synthetic data preserves most statistical properties"
        elif score >= 0.7:
            return "Fair fidelity - synthetic data preserves basic statistical properties"
        elif score >= 0.6:
            return "Poor fidelity - synthetic data shows significant deviation from original"
        else:
            return "Very poor fidelity - synthetic data does not preserve original data characteristics"
