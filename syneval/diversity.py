#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_distances
import networkx as nx
from flair.models import TextClassifier
from flair.data import Sentence
import torch
import logging
import json
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU-only environment at the very beginning
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_default_device('cpu')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def get_device():
    """
    Get the best available device (CUDA if available, otherwise CPU).
    For now, force CPU to avoid device mismatch issues with Flair.
    """
    # Force CPU to avoid device mismatch issues
    device = torch.device('cpu')
    torch.set_num_threads(4)  # Set CPU threads
    return device

def ensure_tensor_device(tensor, device):
    """
    Ensure a tensor is on the specified device.
    """
    if tensor is not None and tensor.device != device:
        return tensor.to(device)
    return tensor

class DiversityEvaluator:
    def __init__(self, 
                 synthetic_data: pd.DataFrame,
                 original_data: pd.DataFrame,
                 metadata: Dict,
                 cache_dir: str = "./cache"):
        """
        Initialize the diversity evaluator.
        
        Args:
            synthetic_data: Synthetic dataset
            original_data: Original dataset
            metadata: Dictionary containing column types and other metadata
            cache_dir: Directory to store cached results
        """
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.metadata = metadata
        self.text_columns = self._get_text_columns()
        self.structured_columns = self._get_structured_columns()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, column: str) -> str:
        """Get the cache file path for a specific column."""
        return os.path.join(self.cache_dir, f"real_text_diversity_{column}.json")
        
    def _load_cached_results(self, column: str) -> Optional[Dict]:
        """Load cached results for a specific column if they exist."""
        cache_path = self._get_cache_path(column)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache for {column}: {str(e)}")
        return None
        
    def _save_cached_results(self, column: str, results: Dict):
        """Save results to cache for a specific column."""
        cache_path = self._get_cache_path(column)
        try:
            with open(cache_path, 'w') as f:
                json.dump(results, f)
        except Exception as e:
            logger.warning(f"Error saving cache for {column}: {str(e)}")
        
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
                cached_results = self._load_cached_results(col)
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
                    self._save_cached_results(col, real_results)
                
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
        """Evaluate lexical diversity using n-gram analysis."""
        stop_words = set(stopwords.words("english"))
        tokenizer = RegexpTokenizer(r"[A-Za-z]+(?:'[A-Za-z]+)*")
        
        def tokenize_and_remove_stopwords(text: str):
            tokens = tokenizer.tokenize(str(text).lower())
            return [t for t in tokens if t not in stop_words]
        
        df = data.copy()
        df["tokens"] = df[text_column].apply(tokenize_and_remove_stopwords)
        
        results = {}
        all_token_lists = df["tokens"].tolist()
        
        for n in tqdm(range(1, 6), desc="Calculating n-grams", leave=False):
            ngrams_flat = []
            for tokens in all_token_lists:
                ngrams_flat.extend(nltk.ngrams(tokens, n) if n > 1 else tokens)
            
            total = len(ngrams_flat)
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
        
        return results
    
    def _evaluate_semantic_diversity(self, data: pd.DataFrame, text_column: str) -> Dict:
        """Evaluate semantic diversity using word embeddings and MST."""
        stop_words = set(stopwords.words("english"))
        
        def rm_sw_tok(text: str):
            tokens = word_tokenize(str(text).lower())
            return [w for w in tokens if w not in stop_words]
        
        df = data.copy()
        df["tokens"] = df[text_column].apply(rm_sw_tok)
        
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
        
        logger.info("Calculating distances...")
        # Calculate distances in batches to avoid memory issues
        batch_size = 1000
        n = len(df)
        dist_m = np.zeros((n, n))
        
        for i in tqdm(range(0, n, batch_size), desc="Calculating distances"):
            end_i = min(i + batch_size, n)
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                dist_m[i:end_i, j:end_j] = cosine_distances(embeddings[i:end_i], embeddings[j:end_j])
        
        # Construct MST using sparse matrix for efficiency
        logger.info("Constructing MST...")
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Add edges in batches
        batch_size = 1000
        total_edges = (n * (n - 1)) // 2
        edges_added = 0
        
        with tqdm(total=total_edges, desc="Building graph") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    G.add_edge(i, j, weight=dist_m[i, j])
                    edges_added += 1
                    if edges_added % 1000 == 0:  # Update progress every 1000 edges
                        pbar.update(1000)
            
            # Update remaining edges
            if edges_added % 1000 != 0:
                pbar.update(edges_added % 1000)
        
        # Use a more efficient MST algorithm with progress tracking
        logger.info("Computing minimum spanning tree...")
        # Convert graph to edge list for faster processing
        edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        edges.sort(key=lambda x: x[2])  # Sort by weight
        
        # Initialize disjoint set for Union-Find
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        # Kruskal's algorithm with progress tracking
        mst_edges = []
        total_edges = len(edges)
        
        with tqdm(total=total_edges, desc="Computing MST") as pbar:
            for u, v, w in edges:
                if find(u) != find(v):
                    mst_edges.append((u, v, w))
                    union(u, v)
                pbar.update(1)
                if len(mst_edges) == n - 1:  # MST is complete
                    break
        
        # Create MST graph from edges
        mst = nx.Graph()
        mst.add_nodes_from(range(n))
        mst.add_weighted_edges_from(mst_edges)
        
        logger.info("Calculating final metrics...")
        # Calculate metrics
        tol = 1e-6
        wts = [0.0 if abs(d["weight"]) < tol else d["weight"]
               for _, _, d in mst.edges(data=True)]
        total_w = sum(wts)
        nz_wts = [w for w in wts if w > 0]
        avg_nz = sum(nz_wts) / len(nz_wts) if nz_wts else 0
        
        # Calculate distinct nodes more efficiently
        logger.info("Calculating distinct nodes...")
        rounded_embs = np.round(embeddings, decimals=6)
        distinct_nodes = len(np.unique(rounded_embs, axis=0))
        ratio_dist = distinct_nodes / n if n else 0
        
        return {
            'total_mst_weight': total_w,
            'average_edge_weight': avg_nz,
            'distinct_nodes': distinct_nodes,
            'distinct_ratio': ratio_dist
        }
    
    def _evaluate_sentiment_diversity(self, data: pd.DataFrame, text_column: str) -> Dict:
        """Evaluate sentiment diversity across rating levels."""
        stop_words = set(stopwords.words("english"))
        
        def rm_sw_tok(text: str):
            toks = word_tokenize(str(text).lower())
            return [w for w in toks if w not in stop_words]
        
        df = data.copy()
        df = df[df[text_column].str.strip().astype(bool)].dropna(subset=[text_column])
        df["cleaned_text"] = df[text_column].apply(lambda t: " ".join(rm_sw_tok(t)))
        
        # Load sentiment classifier
        clf = TextClassifier.load("en-sentiment")
        
        # Get CPU device
        device = get_device()
        logger.info(f"Using device for sentiment classification: {device}")
        
        # Move classifier to CPU
        clf = clf.to(device)
        
        def classify(text):
            s = Sentence(text)
            # Flair will handle device management internally when the model is on the correct device
            clf.predict(s)
            if not s.labels:
                return "Neutral"
            return "Positive" if s.labels[0].value == "POSITIVE" else "Negative"
        
        logger.info("Classifying sentiments...")
        df["Sentiment Category"] = df["cleaned_text"].apply(classify)
        
        # Calculate sentiment distribution
        counts = (df.groupby("rating")["Sentiment Category"]
                 .value_counts(normalize=True).unstack().fillna(0))
        ratings = sorted(df["rating"].unique())
        pos = counts.get("Positive", pd.Series(0, index=ratings))
        actual_pos = [pos.get(r, 0) for r in ratings]
        
        # Calculate deviation from ideal distribution
        ideal = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
        ideal_pos = [ideal.get(r, (r - 1) / 4) for r in ratings]
        
        diffs = [abs(a - b) for a, b in zip(actual_pos, ideal_pos)]
        D_sen = sum(1 - d for d in diffs) / len(ratings)
        
        return {
            'sentiment_by_rating': dict(zip(ratings, actual_pos)),
            'ideal_sentiment': dict(zip(ratings, ideal_pos)),
            'sentiment_alignment_score': D_sen
        }
    
    def evaluate(self) -> Dict:
        """
        Run all diversity evaluations and return comprehensive results.
        
        Returns:
            Dict: Dictionary containing all diversity metrics
        """
        results = {
            'tabular_diversity': self.evaluate_tabular_diversity()
        }
        
        # Only evaluate text diversity if there are text columns in metadata
        text_columns = self._get_text_columns()
        if text_columns:
            logger.info(f"Evaluating text diversity for columns: {text_columns}")
            results['text_diversity'] = self.evaluate_text_diversity()
        else:
            logger.info("No text columns found in metadata, skipping text diversity evaluation")
        
        return results