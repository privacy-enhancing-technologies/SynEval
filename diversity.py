#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import math
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_distances
import networkx as nx
import logging
import json
import os
import hashlib
from pathlib import Path
from tqdm import tqdm
import pickle
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment varies
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from flair.models import TextClassifier  # type: ignore
    from flair.data import Sentence  # type: ignore
    FLAIR_AVAILABLE = True
    FLAIR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    TextClassifier = None  # type: ignore
    Sentence = None  # type: ignore
    FLAIR_AVAILABLE = False
    FLAIR_IMPORT_ERROR = exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
# Set up local NLTK data directory
import os
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add local directory to NLTK data path
nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK data to local directory
def download_nltk_data_if_needed(package_name):
    try:
        nltk.data.find(f'tokenizers/{package_name}' if package_name in ['punkt', 'punkt_tab'] else f'corpora/{package_name}')
    except LookupError:
        nltk.download(package_name, download_dir=nltk_data_dir, quiet=True)

download_nltk_data_if_needed('punkt')
download_nltk_data_if_needed('stopwords')
download_nltk_data_if_needed('punkt_tab')

def get_device() -> str:
    """
    Get the best available device label.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def ensure_tensor_device(tensor, device: str):
    """
    Ensure a tensor is on the specified device.
    """
    if tensor is None or not TORCH_AVAILABLE:
        return tensor
    target = torch.device(device)
    if tensor.device != target:
        return tensor.to(target)
    return tensor


def require_flair():
    """Raise a helpful error if Flair (and by extension torch) is unavailable."""
    if not FLAIR_AVAILABLE:
        raise ImportError(
            "Flair is required for this metric but is not installed or could not be imported. "
            f"Original error: {FLAIR_IMPORT_ERROR}"
        )
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Flair-based metrics but is not installed."
        )

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
                logger.warning(f"Error loading fingerprints: {str(e)}")
        return {}
    
    def _save_fingerprints(self):
        """Save dataset fingerprints."""
        try:
            with open(self.fingerprint_file, 'w') as f:
                json.dump(self.fingerprints, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving fingerprints: {str(e)}")
    
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
                logger.warning(f"Error loading cache {cache_type}: {str(e)}")
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
            
            logger.info(f"Saved cache for {cache_type}")
        except Exception as e:
            logger.warning(f"Error saving cache {cache_type}: {str(e)}")

class DiversityEvaluator:
    def __init__(self, 
                 synthetic_data: pd.DataFrame,
                 original_data: pd.DataFrame,
                 metadata: Dict,
                 cache_dir: str = "./cache",
                 selected_metrics: List[str] = None,
                 device: str = 'auto'):
        """
        Initialize the diversity evaluator.
        
        Args:
            synthetic_data: Synthetic dataset
            original_data: Original dataset
            metadata: Dictionary containing column types and other metadata
            cache_dir: Directory to store cached results
            selected_metrics: List of specific metrics to run
            device: Device to use for computation ('auto', 'cpu', 'cuda')
        """
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.metadata = metadata
        self.text_columns = self._get_text_columns()
        self.structured_columns = self._get_structured_columns()
        
        # Initialize smart cache
        self.cache = SmartCache(cache_dir)
        
        # Compute dataset fingerprints
        self.original_fingerprint = compute_dataset_fingerprint(original_data, metadata)
        self.synthetic_fingerprint = compute_dataset_fingerprint(synthetic_data, metadata)
        
        # Device management
        if device == 'auto':
            self.device = get_device()
        elif device == 'cuda':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
        else:  # cpu
            self.device = 'cpu'
        
        logger.info(f"Using device: {self.device}")
        
        # GPU acceleration settings
        if self.device == 'cuda':
            self.batch_size = 2000  # Larger batches for GPU
            logger.info("GPU acceleration enabled")
        else:
            self.batch_size = 1000  # Smaller batches for CPU
            logger.info("CPU processing mode")
        
        # Available metrics for selection
        self.available_metrics = [
            'tabular_diversity', 'text_diversity'
        ]
        
        # Use all metrics if none specified
        self.selected_metrics = (
            selected_metrics.copy() if selected_metrics else self.available_metrics.copy()
        )
        self.unavailable_metrics: Dict[str, str] = {}
        if 'text_diversity' in self.selected_metrics and not FLAIR_AVAILABLE:
            reason = "requires Flair (and PyTorch) for sentiment analysis"
            if not TORCH_AVAILABLE:
                reason = "requires PyTorch and Flair for sentiment analysis"
            elif FLAIR_IMPORT_ERROR:
                reason = f"requires Flair; import failed with: {FLAIR_IMPORT_ERROR}"
            self.unavailable_metrics['text_diversity'] = reason
            self.selected_metrics.remove('text_diversity')
        
    def _get_text_columns(self) -> List[str]:
        """Get list of text columns from metadata."""
        return [col for col, info in self.metadata['columns'].items() 
                if info['sdtype'] == 'text']
    
    def _get_structured_columns(self) -> List[str]:
        """Get list of structured columns from metadata."""
        return [col for col, info in self.metadata['columns'].items() 
                if info['sdtype'] in ['numerical', 'categorical', 'time', 'pii'] 
                and col not in ['dataset_name', 'Primary key']]
    
    def _evaluate_numerical_diversity(self, col: str) -> Dict:
        """Evaluate diversity metrics for numerical columns."""
        try:
            real_data = self.original_data[col].dropna()
            syn_data = self.synthetic_data[col].dropna()
            
            # Basic statistics
            real_stats = {
                'mean': real_data.mean(),
                'std': real_data.std(),
                'skew': real_data.skew(),
                'kurtosis': real_data.kurtosis(),
                'min': real_data.min(),
                'max': real_data.max()
            }
            
            syn_stats = {
                'mean': syn_data.mean(),
                'std': syn_data.std(),
                'skew': syn_data.skew(),
                'kurtosis': syn_data.kurtosis(),
                'min': syn_data.min(),
                'max': syn_data.max()
            }
            
            # Calculate statistical differences
            stat_diffs = {
                'mean_diff': abs(real_stats['mean'] - syn_stats['mean']) / (abs(real_stats['mean']) + 1e-10),
                'std_diff': abs(real_stats['std'] - syn_stats['std']) / (abs(real_stats['std']) + 1e-10),
                'skew_diff': abs(real_stats['skew'] - syn_stats['skew']),
                'kurtosis_diff': abs(real_stats['kurtosis'] - syn_stats['kurtosis'])
            }
            
            # Value range coverage
            overlap = min(real_stats['max'], syn_stats['max']) - max(real_stats['min'], syn_stats['min'])
            total_range = real_stats['max'] - real_stats['min']
            range_coverage = max(0, overlap / total_range) if total_range != 0 else 0
            
            # Quartile coverage
            real_quartiles = real_data.quantile([0.25, 0.5, 0.75])
            syn_quartiles = syn_data.quantile([0.25, 0.5, 0.75])
            
            quartile_coverage = {
                'q1': abs(real_quartiles[0.25] - syn_quartiles[0.25]) / (abs(real_quartiles[0.25]) + 1e-10),
                'q2': abs(real_quartiles[0.5] - syn_quartiles[0.5]) / (abs(real_quartiles[0.5]) + 1e-10),
                'q3': abs(real_quartiles[0.75] - syn_quartiles[0.75]) / (abs(real_quartiles[0.75]) + 1e-10)
            }
            
            # Distribution similarity using histogram comparison
            bins = min(50, len(real_data.unique()))
            real_hist, _ = np.histogram(real_data, bins=bins, density=True)
            syn_hist, _ = np.histogram(syn_data, bins=bins, density=True)
            
            # Normalize histograms
            real_hist = real_hist / (real_hist.sum() + 1e-10)
            syn_hist = syn_hist / (syn_hist.sum() + 1e-10)
            
            # Calculate KL divergence
            kl_div = np.sum(real_hist * np.log((real_hist + 1e-10) / (syn_hist + 1e-10)))
            
            return {
                'statistical_differences': stat_diffs,
                'range_coverage': range_coverage * 100,
                'quartile_coverage': {k: (1 - v) * 100 for k, v in quartile_coverage.items()},
                'distribution_similarity': {
                    'kl_divergence': kl_div,
                    'similarity_score': 100 * np.exp(-kl_div)  # Convert to similarity score
                },
                'real_statistics': real_stats,
                'synthetic_statistics': syn_stats
            }
            
        except Exception as e:
            logger.error(f"Error evaluating numerical diversity for column {col}: {str(e)}")
            return {}
    
    def _evaluate_categorical_diversity(self, col: str) -> Dict:
        """Evaluate diversity metrics for categorical columns."""
        try:
            real_data = self.original_data[col].dropna()
            syn_data = self.synthetic_data[col].dropna()
            
            # Calculate value counts
            real_counts = real_data.value_counts(normalize=True)
            syn_counts = syn_data.value_counts(normalize=True)
            
            # Get all unique categories
            all_categories = set(real_counts.index) | set(syn_counts.index)
            
            # Calculate category coverage
            common_categories = set(real_counts.index) & set(syn_counts.index)
            category_coverage = len(common_categories) / len(real_counts) if len(real_counts) > 0 else 0
            
            # Calculate distribution similarity
            distribution_diff = 0
            for cat in all_categories:
                real_prob = real_counts.get(cat, 0)
                syn_prob = syn_counts.get(cat, 0)
                distribution_diff += abs(real_prob - syn_prob)
            
            distribution_similarity = (1 - distribution_diff / 2) * 100  # Normalize to 0-100
            
            # Calculate entropy
            def calculate_entropy(probs):
                return -sum(p * np.log2(p) for p in probs if p > 0)
            
            real_entropy = calculate_entropy(real_counts)
            syn_entropy = calculate_entropy(syn_counts)
            
            # Most common categories coverage
            top_n = min(5, len(real_counts))
            top_categories = set(real_counts.nlargest(top_n).index)
            top_coverage = len(top_categories & set(syn_counts.index)) / top_n
            
            # Rare categories coverage (categories that appear in less than 1% of data)
            rare_threshold = 0.01
            rare_categories = set(real_counts[real_counts < rare_threshold].index)
            rare_coverage = len(rare_categories & set(syn_counts.index)) / len(rare_categories) if rare_categories else 0
            
            return {
                'category_coverage': category_coverage * 100,
                'distribution_similarity': distribution_similarity,
                'entropy': {
                    'real': real_entropy,
                    'synthetic': syn_entropy,
                    'difference': abs(real_entropy - syn_entropy)
                },
                'top_categories_coverage': top_coverage * 100,
                'rare_categories_coverage': rare_coverage * 100,
                'category_counts': {
                    'real': len(real_counts),
                    'synthetic': len(syn_counts),
                    'common': len(common_categories)
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluating categorical diversity for column {col}: {str(e)}")
            return {}
    
    def _calculate_numerical_entropy(self, data: pd.Series, bins: int = 50) -> float:
        """Calculate entropy for numerical data using histogram bins."""
        try:
            # Create histogram
            hist, _ = np.histogram(data.dropna(), bins=bins, density=True)
            # Normalize to get probabilities
            probs = hist / (hist.sum() + 1e-10)
            # Calculate entropy
            return -np.sum(probs * np.log2(probs + 1e-10))
        except Exception as e:
            logger.error(f"Error calculating numerical entropy: {str(e)}")
            return 0.0

    def _calculate_categorical_entropy(self, data: pd.Series) -> float:
        """Calculate entropy for categorical data."""
        try:
            # Get value counts and normalize
            probs = data.value_counts(normalize=True)
            # Calculate entropy
            return -np.sum(probs * np.log2(probs))
        except Exception as e:
            logger.error(f"Error calculating categorical entropy: {str(e)}")
            return 0.0

    def _evaluate_entropy(self) -> Dict:
        """Evaluate entropy metrics for the entire dataset."""
        try:
            results = {
                'column_entropy': {},
                'dataset_entropy': {
                    'real': 0.0,
                    'synthetic': 0.0,
                    'entropy_ratio': 0.0
                }
            }
            
            # Calculate entropy for each column
            total_real_entropy = 0
            total_syn_entropy = 0
            num_columns = 0
            
            for col in self.structured_columns:
                try:
                    col_type = self.metadata['columns'][col]['sdtype']
                    
                    if col_type == 'numerical':
                        real_entropy = self._calculate_numerical_entropy(self.original_data[col])
                        syn_entropy = self._calculate_numerical_entropy(self.synthetic_data[col])
                    else:  # categorical
                        real_entropy = self._calculate_categorical_entropy(self.original_data[col])
                        syn_entropy = self._calculate_categorical_entropy(self.synthetic_data[col])
                    
                    results['column_entropy'][col] = {
                        'real': real_entropy,
                        'synthetic': syn_entropy,
                        'entropy_ratio': syn_entropy / (real_entropy + 1e-10),
                        'entropy_difference': abs(real_entropy - syn_entropy)
                    }
                    
                    total_real_entropy += real_entropy
                    total_syn_entropy += syn_entropy
                    num_columns += 1
                    
                except Exception as e:
                    logger.error(f"Error calculating entropy for column {col}: {str(e)}")
                    continue
            
            # Calculate average dataset entropy
            if num_columns > 0:
                results['dataset_entropy'] = {
                    'real': total_real_entropy / num_columns,
                    'synthetic': total_syn_entropy / num_columns,
                    'entropy_ratio': (total_syn_entropy / num_columns) / ((total_real_entropy / num_columns) + 1e-10)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating entropy: {str(e)}")
            return {
                'column_entropy': {},
                'dataset_entropy': {
                    'real': 0.0,
                    'synthetic': 0.0,
                    'entropy_ratio': 0.0
                }
            }

    def evaluate_tabular_diversity(self) -> Dict:
        """
        Evaluate diversity metrics for tabular data.
        
        Returns:
            Dict: Dictionary containing coverage, uniqueness, and entropy metrics
        """
        results = {
            'coverage': {},
            'uniqueness': {},
            'numerical_metrics': {},
            'categorical_metrics': {},
            'entropy_metrics': {}
        }
        
        # Evaluate coverage for each column
        for col in self.structured_columns:
            try:
                col_type = self.metadata['columns'][col]['sdtype']
                
                if col_type == 'numerical':
                    # Basic coverage
                    real_min = self.original_data[col].min()
                    real_max = self.original_data[col].max()
                    syn_min = self.synthetic_data[col].min()
                    syn_max = self.synthetic_data[col].max()
                    
                    overlap = min(real_max, syn_max) - max(real_min, syn_min)
                    total_range = real_max - real_min
                    
                    coverage = max(0, overlap / total_range) if total_range != 0 else 0
                    results['coverage'][col] = coverage * 100
                    
                    # Enhanced numerical metrics
                    results['numerical_metrics'][col] = self._evaluate_numerical_diversity(col)
                    
                elif col_type == 'categorical':
                    # Basic coverage
                    real_categories = set(self.original_data[col].unique())
                    syn_categories = set(self.synthetic_data[col].unique())
                    
                    coverage = len(real_categories.intersection(syn_categories)) / len(real_categories)
                    results['coverage'][col] = coverage * 100
                    
                    # Enhanced categorical metrics
                    results['categorical_metrics'][col] = self._evaluate_categorical_diversity(col)
                    
            except Exception as e:
                logger.error(f"Error evaluating coverage for column {col}: {str(e)}")
                results['coverage'][col] = 0.0
        
        # Evaluate uniqueness (duplicate rows)
        try:
            # Calculate duplicate ratio for synthetic data
            total_rows = len(self.synthetic_data)
            unique_rows = len(self.synthetic_data.drop_duplicates())
            duplicate_ratio = 1 - (unique_rows / total_rows)
            
            # Calculate duplicate ratio for original data for comparison
            orig_total = len(self.original_data)
            orig_unique = len(self.original_data.drop_duplicates())
            orig_duplicate_ratio = 1 - (orig_unique / orig_total)
            
            results['uniqueness'] = {
                'synthetic_duplicate_ratio': duplicate_ratio * 100,
                'original_duplicate_ratio': orig_duplicate_ratio * 100,
                'relative_duplication': (duplicate_ratio / orig_duplicate_ratio) * 100 if orig_duplicate_ratio > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error evaluating uniqueness: {str(e)}")
            results['uniqueness'] = {
                'synthetic_duplicate_ratio': 0.0,
                'original_duplicate_ratio': 0.0,
                'relative_duplication': 0.0
            }
        
        # Evaluate entropy
        results['entropy_metrics'] = self._evaluate_entropy()
        
        return results
    
    def evaluate_text_diversity(self) -> Dict:
        """
        Evaluate diversity metrics for text data.
        
        Returns:
            Dict: Dictionary containing lexical, semantic, and sentiment diversity metrics
            for both synthetic and real datasets
        """
        results = {
            'synthetic': {},
            'real': {}
        }
        
        for col in tqdm(self.text_columns, desc="Evaluating text columns"):
            try:
                # Evaluate synthetic data
                logger.info(f"Evaluating synthetic data for column: {col}")
                # Lexical diversity
                syn_lexical_metrics = self._evaluate_lexical_diversity(self.synthetic_data, col)
                
                # Semantic diversity
                syn_semantic_metrics = self._evaluate_semantic_diversity(self.synthetic_data, col)
                
                # Sentiment diversity
                syn_sentiment_metrics = self._evaluate_sentiment_diversity(self.synthetic_data, col)
                
                results['synthetic'][col] = {
                    'lexical_diversity': syn_lexical_metrics,
                    'semantic_diversity': syn_semantic_metrics,
                    'sentiment_diversity': syn_sentiment_metrics
                }
                
                # Try to load cached results for real data
                cached_results = self.cache.load_cache("text_diversity", self.original_fingerprint, col)
                if cached_results is not None:
                    logger.info(f"Using cached results for real data column: {col}")
                    results['real'][col] = cached_results
                else:
                    logger.info(f"Evaluating real data for column: {col}")
                    # Lexical diversity
                    real_lexical_metrics = self._evaluate_lexical_diversity(self.original_data, col)
                    
                    # Semantic diversity
                    real_semantic_metrics = self._evaluate_semantic_diversity(self.original_data, col)
                    
                    # Sentiment diversity
                    real_sentiment_metrics = self._evaluate_sentiment_diversity(self.original_data, col)
                    
                    real_results = {
                        'lexical_diversity': real_lexical_metrics,
                        'semantic_diversity': real_semantic_metrics,
                        'sentiment_diversity': real_sentiment_metrics
                    }
                    
                    results['real'][col] = real_results
                    # Cache the results
                    self.cache.save_cache("text_diversity", self.original_fingerprint, real_results, col)
                
            except Exception as e:
                logger.error(f"Error evaluating text diversity for column {col}: {str(e)}")
                results['synthetic'][col] = {
                    'lexical_diversity': {},
                    'semantic_diversity': {},
                    'sentiment_diversity': {}
                }
                results['real'][col] = {
                    'lexical_diversity': {},
                    'semantic_diversity': {},
                    'sentiment_diversity': {}
                }
        
        return results
    
    def _evaluate_lexical_diversity(self, data: pd.DataFrame, text_column: str) -> Dict:
        """Evaluate lexical diversity using n-gram analysis with full dataset."""
        try:
            # Check cache first
            cached_result = self.cache.load_cache("lexical_diversity", self.original_fingerprint, text_column)
            if cached_result is not None:
                logger.info(f"Using cached lexical diversity results for {text_column}")
                return cached_result
            
            stop_words = set(stopwords.words("english"))
            tokenizer = RegexpTokenizer(r"[A-Za-z]+(?:'[A-Za-z]+)*")
            
            def tokenize_and_remove_stopwords(text: str):
                tokens = tokenizer.tokenize(str(text).lower())
                return [t for t in tokens if t not in stop_words]
            
            df = data.copy()
            df["tokens"] = df[text_column].apply(tokenize_and_remove_stopwords)
            
            # Filter out empty token lists
            df = df[df["tokens"].apply(len) > 0].reset_index(drop=True)
            
            if len(df) == 0:
                logger.warning("No valid text data for lexical analysis")
                result = {
                    'sample_size': 0,
                    'error': 'No valid text data'
                }
                self.cache.save_cache("lexical_diversity", self.original_fingerprint, result, text_column)
                return result
            
            logger.info(f"Processing {len(df)} samples for lexical diversity (no sampling)")
            
            results = {}
            all_token_lists = df["tokens"].tolist()
            
            for n in tqdm(range(1, 6), desc="Calculating n-grams"):
                try:
                    ngrams_flat = []
                    for tokens in all_token_lists:
                        ngrams_flat.extend(ngrams(tokens, n) if n > 1 else tokens)
                    
                    total = len(ngrams_flat)
                    if total == 0:
                        results[f"{n}-gram"] = {
                            'total': 0,
                            'unique': 0,
                            'unique_ratio': 0.0,
                            'entropy': 0.0,
                            'normalized_entropy': 0.0
                        }
                        continue
                    
                    counts = Counter(ngrams_flat)
                    unique = len(counts)
                    ratio = unique / total if total else 0
                    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
                    norm_ent = entropy / math.log2(unique) if unique > 1 else 0
                    
                    results[f"{n}-gram"] = {
                        'total': total,
                        'unique': unique,
                        'unique_ratio': ratio,
                        'entropy': entropy,
                        'normalized_entropy': norm_ent
                    }
                except Exception as e:
                    logger.error(f"Error calculating {n}-gram metrics: {str(e)}")
                    results[f"{n}-gram"] = {
                        'total': 0,
                        'unique': 0,
                        'unique_ratio': 0.0,
                        'entropy': 0.0,
                        'normalized_entropy': 0.0,
                        'error': str(e)
                    }
            
            results['sample_size'] = len(df)
            
            # Cache the results
            self.cache.save_cache("lexical_diversity", self.original_fingerprint, results, text_column)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in lexical diversity evaluation: {str(e)}")
            return {
                'sample_size': 0,
                'error': str(e)
            }
    
    def _evaluate_semantic_diversity(self, data: pd.DataFrame, text_column: str) -> Dict:
        """Evaluate semantic diversity using word embeddings and MST with full dataset."""
        try:
            # Check cache first
            cache_key = f"semantic_diversity_{text_column}"
            cached_result = self.cache.load_cache("semantic_diversity", self.original_fingerprint, text_column)
            if cached_result is not None:
                logger.info(f"Using cached semantic diversity results for {text_column}")
                return cached_result
            
            stop_words = set(stopwords.words("english"))
            
            def rm_sw_tok(text: str):
                # Use simple regex-based tokenization to avoid punkt_tab dependency
                import re
                tokens = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
                return [w for w in tokens if w not in stop_words]
            
            df = data.copy()
            df["tokens"] = df[text_column].apply(rm_sw_tok)
            
            # Filter out empty token lists
            df = df[df["tokens"].apply(len) > 0].reset_index(drop=True)
            
            n = len(df)
            if n < 2:
                logger.warning("Not enough samples for semantic diversity evaluation")
                result = {
                    'total_mst_weight': 0.0,
                    'average_edge_weight': 0.0,
                    'distinct_nodes': n,
                    'distinct_ratio': 1.0 if n > 0 else 0.0,
                    'sample_size': n
                }
                self.cache.save_cache("semantic_diversity", self.original_fingerprint, result, text_column)
                return result
            
            logger.info(f"Processing {n} samples for semantic diversity (no sampling)")
            
            # Train Word2Vec model
            logger.info("Training Word2Vec model...")
            w2v = Word2Vec(sentences=df["tokens"], vector_size=100,
                          window=5, min_count=1, workers=4)
            
            def sent_vec(tokens):
                vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
                return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)
            
            logger.info("Calculating embeddings...")
            df["embedding"] = df["tokens"].apply(sent_vec)
            embeddings = np.stack(df["embedding"].values)
            
            # Move to GPU if available
            if self.device == 'cuda' and TORCH_AVAILABLE:
                embeddings = torch.tensor(embeddings, device=self.device)
                logger.info("Using GPU for distance calculations")
                
                # Calculate distances in batches on GPU
                batch_size = self.batch_size
                n_batches = (n + batch_size - 1) // batch_size
                distances = []
                
                for i in tqdm(range(n_batches), desc="Calculating distances (GPU)"):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n)
                    batch_embeddings = embeddings[start_idx:end_idx]
                    
                    # Calculate cosine distances for this batch
                    batch_distances = []
                    for j in range(n_batches):
                        j_start = j * batch_size
                        j_end = min((j + 1) * batch_size, n)
                        j_embeddings = embeddings[j_start:j_end]
                        
                        # Calculate cosine distances
                        cos_sim = torch.mm(batch_embeddings, j_embeddings.T)
                        batch_dist = 1 - cos_sim
                        batch_distances.append(batch_dist.cpu().numpy())
                    
                    distances.extend(batch_distances)
                
                # Convert to full distance matrix
                dist_matrix = np.zeros((n, n))
                for i in range(n_batches):
                    for j in range(n_batches):
                        start_i = i * batch_size
                        end_i = min((i + 1) * batch_size, n)
                        start_j = j * batch_size
                        end_j = min((j + 1) * batch_size, n)
                        
                        if i < len(distances) and j < len(distances[i]):
                            dist_matrix[start_i:end_i, start_j:end_j] = distances[i][:, :end_j-start_j]
                
                # Move back to CPU for MST calculation
                dist_matrix = torch.tensor(dist_matrix, device='cpu').numpy()
            else:
                # CPU processing
                logger.info("Using CPU for distance calculations")
                distances = pdist(embeddings, metric='cosine')
                dist_matrix = squareform(distances)
            
            # Calculate MST using scipy's efficient implementation
            logger.info("Computing minimum spanning tree...")
            mst_matrix = minimum_spanning_tree(dist_matrix)
            
            # Extract MST edges and weights
            mst_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    weight = mst_matrix[i, j]
                    if weight > 0:
                        mst_edges.append((i, j, weight))
            
            logger.info(f"Found {len(mst_edges)} MST edges")
            
            # Calculate metrics
            if mst_edges:
                weights = [edge[2] for edge in mst_edges]
                total_w = sum(weights)
                avg_w = total_w / len(weights)
            else:
                total_w = 0.0
                avg_w = 0.0
            
            # Calculate distinct nodes more efficiently
            logger.info("Calculating distinct nodes...")
            rounded_embs = np.round(embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings, decimals=6)
            distinct_nodes = len(np.unique(rounded_embs, axis=0))
            ratio_dist = distinct_nodes / n if n else 0
            
            result = {
                'total_mst_weight': float(total_w),
                'average_edge_weight': float(avg_w),
                'distinct_nodes': distinct_nodes,
                'distinct_ratio': ratio_dist,
                'sample_size': n,
                'device_used': str(self.device)
            }
            
            # Cache the results
            self.cache.save_cache("semantic_diversity", self.original_fingerprint, result, text_column)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in semantic diversity evaluation: {str(e)}")
            return {
                'total_mst_weight': 0.0,
                'average_edge_weight': 0.0,
                'distinct_nodes': 0,
                'distinct_ratio': 0.0,
                'error': str(e)
            }
    
    def _evaluate_sentiment_diversity(self, data: pd.DataFrame, text_column: str) -> Dict:
        """Evaluate sentiment diversity across rating levels with full dataset."""
        try:
            # Check cache first
            cached_result = self.cache.load_cache("sentiment_diversity", self.original_fingerprint, text_column)
            if cached_result is not None:
                logger.info(f"Using cached sentiment diversity results for {text_column}")
                return cached_result
            
            stop_words = set(stopwords.words("english"))
            
            def rm_sw_tok(text: str):
                # Use simple regex-based tokenization to avoid punkt_tab dependency
                import re
                toks = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
                return [w for w in toks if w not in stop_words]
            
            df = data.copy()
            df = df[df[text_column].str.strip().astype(bool)].dropna(subset=[text_column])
            
            if len(df) == 0:
                logger.warning("No valid text data for sentiment analysis")
                result = {
                    'sentiment_by_rating': {},
                    'ideal_sentiment': {},
                    'sentiment_alignment_score': 0.0,
                    'sample_size': 0
                }
                self.cache.save_cache("sentiment_diversity", self.original_fingerprint, result, text_column)
                return result
            
            logger.info(f"Processing {len(df)} samples for sentiment diversity (no sampling)")
            
            df["cleaned_text"] = df[text_column].apply(lambda t: " ".join(rm_sw_tok(t)))
            
            # Check if rating column exists
            if 'rating' not in df.columns:
                logger.warning("Rating column not found, using default sentiment analysis")
                # Detect best available device
                require_flair()
                device = get_device()
                logger.info(f"Using device: {device} for sentiment classification")
                
                # Load sentiment classifier and move to device
                clf = TextClassifier.load("en-sentiment")
                clf = clf.to(device)  # Move to device (compatible with all Flair versions)
                
                def classify(text):
                    try:
                        s = Sentence(text)
                        # Predict without device parameter (compatible with all Flair versions)
                        clf.predict(s)
                        if not s.labels:
                            return "Neutral"
                        return "Positive" if s.labels[0].value == "POSITIVE" else "Negative"
                    except Exception as e:
                        logger.warning(f"Error classifying text: {str(e)}")
                        return "Neutral"
                
                logger.info("Classifying sentiments...")
                df["Sentiment Category"] = df["cleaned_text"].apply(classify)
                
                # Calculate basic sentiment distribution
                sentiment_counts = df["Sentiment Category"].value_counts(normalize=True)
                
                result = {
                    'sentiment_distribution': sentiment_counts.to_dict(),
                    'sentiment_alignment_score': 0.0,  # No rating-based alignment
                    'sample_size': len(df)
                }
                
                # Cache the results
                self.cache.save_cache("sentiment_diversity", self.original_fingerprint, result, text_column)
                return result
            
            # Detect best available device
            require_flair()
            device = get_device()
            logger.info(f"Using device: {device} for sentiment classification")
            
            # Load sentiment classifier and move to device
            clf = TextClassifier.load("en-sentiment")
            clf = clf.to(device)  # Move to device (compatible with all Flair versions)
            
            def classify(text):
                try:
                    s = Sentence(text)
                    # Predict without device parameter (compatible with all Flair versions)
                    clf.predict(s)
                    if not s.labels:
                        return "Neutral"
                    return "Positive" if s.labels[0].value == "POSITIVE" else "Negative"
                except Exception as e:
                    logger.warning(f"Error classifying text: {str(e)}")
                    return "Neutral"
            
            logger.info("Classifying sentiments...")
            df["Sentiment Category"] = df["cleaned_text"].apply(classify)
            
            # Calculate sentiment distribution
            try:
                counts = (df.groupby("rating")["Sentiment Category"]
                         .value_counts(normalize=True).unstack().fillna(0))
                ratings = sorted(df["rating"].unique())
                pos = counts.get("Positive", pd.Series(0, index=ratings))
                actual_pos = [pos.get(r, 0) for r in ratings]
                
                # Calculate deviation from ideal distribution
                ideal = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
                ideal_pos = [ideal.get(r, (r - 1) / 4) for r in ratings]
                
                diffs = [abs(a - b) for a, b in zip(actual_pos, ideal_pos)]
                D_sen = sum(1 - d for d in diffs) / len(ratings) if ratings else 0
                
                result = {
                    'sentiment_by_rating': dict(zip(ratings, actual_pos)),
                    'ideal_sentiment': dict(zip(ratings, ideal_pos)),
                    'sentiment_alignment_score': D_sen,
                    'sample_size': len(df)
                }
                
                # Cache the results
                self.cache.save_cache("sentiment_diversity", self.original_fingerprint, result, text_column)
                return result
                
            except Exception as e:
                logger.error(f"Error calculating sentiment distribution: {str(e)}")
                result = {
                    'sentiment_by_rating': {},
                    'ideal_sentiment': {},
                    'sentiment_alignment_score': 0.0,
                    'sample_size': len(df),
                    'error': str(e)
                }
                
                # Cache the results
                self.cache.save_cache("sentiment_diversity", self.original_fingerprint, result, text_column)
                return result
                
        except Exception as e:
            logger.error(f"Error in sentiment diversity evaluation: {str(e)}")
            return {
                'sentiment_by_rating': {},
                'ideal_sentiment': {},
                'sentiment_alignment_score': 0.0,
                'sample_size': 0,
                'error': str(e)
            }
    
    def evaluate(self) -> Dict:
        """
        Run selected diversity evaluations and return comprehensive results.
        
        Returns:
            Dict: Dictionary containing selected diversity metrics
        """
        results = {}
        
        if 'tabular_diversity' in self.selected_metrics:
            results['tabular_diversity'] = self.evaluate_tabular_diversity()
        
        # Only evaluate text diversity if there are text columns in metadata
        if 'text_diversity' in self.selected_metrics:
            text_columns = self._get_text_columns()
            if text_columns:
                logger.info(f"Evaluating text diversity for columns: {text_columns}")
                results['text_diversity'] = self.evaluate_text_diversity()
            else:
                logger.info("No text columns found in metadata, skipping text diversity evaluation")
        elif self.unavailable_metrics.get('text_diversity'):
            results['text_diversity'] = {
                'skipped': True,
                'reason': self.unavailable_metrics['text_diversity']
            }
        
        if self.unavailable_metrics:
            results['skipped_metrics'] = self.unavailable_metrics
        
        return results
