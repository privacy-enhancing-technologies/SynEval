import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import spacy
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_distances
import logging
from typing import Dict, List, Tuple, Optional
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import anonymeter
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
from anonymeter.stats.confidence import EvaluationResults
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import lru_cache
import threading
import warnings
import json
import hashlib
from pathlib import Path
import pickle
from sklearn.neighbors import NearestNeighbors

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment specific
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from flair.data import Sentence  # type: ignore
    from flair.models import SequenceTagger  # type: ignore
    FLAIR_AVAILABLE = True
    FLAIR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    Sentence = None  # type: ignore
    SequenceTagger = None  # type: ignore
    FLAIR_AVAILABLE = False
    FLAIR_IMPORT_ERROR = exc

# Suppress warnings
warnings.filterwarnings('ignore')

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

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()

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

# 1. Identical Match Share (IMS)
def identical_match_share(syn_df, train_df, key_cols=None):
    """
    计算合成数据与训练数据完全一致的行占比。
    key_cols: 指定主键或全部列
    返回：ims分数、判定结果、详细说明
    """
    if key_cols is None:
        # Use all common columns for IMS calculation
        key_cols = list(set(syn_df.columns) & set(train_df.columns))
    
    # Convert all values to strings for consistent comparison
    syn_set = set()
    for _, row in syn_df[key_cols].iterrows():
        syn_set.add(tuple(str(val) for val in row.values))
    
    train_set = set()
    for _, row in train_df[key_cols].iterrows():
        train_set.add(tuple(str(val) for val in row.values))
    
    ims_syn_train = len(syn_set & train_set) / len(syn_set) if len(syn_set) > 0 else 0
    
    # 训练-测试间的IMS（假设train_df有test_df可用，否则用train自身shuffle）
    train_test_ims = 0
    if hasattr(train_df, 'test_df') and train_df.test_df is not None:
        test_set = set()
        for _, row in train_df.test_df[key_cols].iterrows():
            test_set.add(tuple(str(val) for val in row.values))
        train_test_ims = len(train_set & test_set) / len(train_set) if len(train_set) > 0 else 0
    else:
        # fallback: train自身shuffle
        shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        shuffled_set = set()
        for _, row in shuffled[key_cols].iterrows():
            shuffled_set.add(tuple(str(val) for val in row.values))
        train_test_ims = len(train_set & shuffled_set) / len(train_set) if len(train_set) > 0 else 0
    
    passed = ims_syn_train <= train_test_ims
    return {
        'ims_syn_train': ims_syn_train,
        'train_test_ims': train_test_ims,
        'passed': passed,
        'method': 'Identical Match Share (IMS)',
        'desc': f'Synthetic-Train IMS: {ims_syn_train:.4f}, Train-Test IMS: {train_test_ims:.4f}, Passed: {"Yes" if passed else "No"}'
    }

def prepare_tabular_features(df, cols):
    """Prepare features by encoding categorical and boolean data."""
    df_copy = df[cols].copy()
    
    # Handle datetime columns
    datetime_cols = []
    for col in cols:
        if df_copy[col].dtype == 'datetime64[ns]' or 'datetime' in str(df_copy[col].dtype):
            datetime_cols.append(col)
            # Convert datetime to numeric (days since epoch, not seconds)
            df_copy[col] = pd.to_datetime(df_copy[col]).dt.tz_localize(None).astype(np.int64) // (10**9 * 86400)
    
    # Handle categorical and boolean columns
    categorical_cols = df_copy.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(categorical_cols) > 0:
        # One-hot encode categorical columns
        df_copy = pd.get_dummies(df_copy, columns=categorical_cols, dummy_na=True)
    
    # Fill missing values
    df_copy = df_copy.fillna(0)
    
    # Convert to numeric array
    return df_copy.astype(float).values

# 2. Distance to Closest Records (DCR)
def distance_to_closest_records(syn_df, train_df, feature_cols):
    """
    计算每条合成数据到训练集最近邻的距离分布，并与训练集自身分布对比。
    返回：5%分位数、判定结果、详细说明
    """
    try:
        X_syn = prepare_tabular_features(syn_df, feature_cols)
        X_train = prepare_tabular_features(train_df, feature_cols)
        
        # Ensure we have valid data
        if X_syn.shape[1] == 0 or X_train.shape[1] == 0:
            return {
                'error': 'No valid features after preprocessing',
                'method': 'Distance to Closest Records (DCR)'
            }
        
        nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_train)
        # 合成-训练最近距离
        dists_syn_train, _ = nn.kneighbors(X_syn, n_neighbors=1)
        dists_syn_train = dists_syn_train.flatten()
        syn_train_5pct = np.percentile(dists_syn_train, 5)
        # 训练自身最近距离（排除自身）
        dists_train_train, idxs = nn.kneighbors(X_train, n_neighbors=2)
        dists_train_train = dists_train_train[:, 1]  # 跳过自身
        train_train_5pct = np.percentile(dists_train_train, 5)
        passed = syn_train_5pct >= train_train_5pct
        return {
            'syn_train_5pct': syn_train_5pct,
            'train_train_5pct': train_train_5pct,
            'passed': passed,
            'method': 'Distance to Closest Records (DCR)',
            'desc': f'Synthetic-Train 5%: {syn_train_5pct:.4f}, Train-Train 5%: {train_train_5pct:.4f}, Passed: {"Yes" if passed else "No"}'
        }
    except Exception as e:
        return {
            'error': f'DCR calculation failed: {str(e)}',
            'method': 'Distance to Closest Records (DCR)'
        }

# 3. Nearest Neighbor Distance Ratio (NNDR)
def nearest_neighbor_distance_ratio(syn_df, train_df, feature_cols):
    """
    计算每条数据到最近邻和次近邻的距离比值分布，并与训练集自身分布对比。
    返回：5%分位数、判定结果、详细说明
    """
    try:
        X_syn = prepare_tabular_features(syn_df, feature_cols)
        X_train = prepare_tabular_features(train_df, feature_cols)
        
        # Ensure we have valid data
        if X_syn.shape[1] == 0 or X_train.shape[1] == 0:
            return {
                'error': 'No valid features after preprocessing',
                'method': 'Nearest Neighbor Distance Ratio (NNDR)'
            }
        
        nn = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(X_train)
        # 合成-训练最近邻距离比
        dists_syn_train, _ = nn.kneighbors(X_syn, n_neighbors=2)
        ratio_syn = dists_syn_train[:, 1] / (dists_syn_train[:, 0] + 1e-8)
        syn_train_5pct = np.percentile(ratio_syn, 5)
        # 训练自身最近邻距离比（排除自身）
        dists_train_train, _ = nn.kneighbors(X_train, n_neighbors=3)
        ratio_train = dists_train_train[:, 2] / (dists_train_train[:, 1] + 1e-8)
        train_train_5pct = np.percentile(ratio_train, 5)
        passed = syn_train_5pct >= train_train_5pct
        return {
            'syn_train_5pct': syn_train_5pct,
            'train_train_5pct': train_train_5pct,
            'passed': passed,
            'method': 'Nearest Neighbor Distance Ratio (NNDR)',
            'desc': f'Synthetic-Train 5%: {syn_train_5pct:.4f}, Train-Train 5%: {train_train_5pct:.4f}, Passed: {"Yes" if passed else "No"}'
        }
    except Exception as e:
        return {
            'error': f'NNDR calculation failed: {str(e)}',
            'method': 'Nearest Neighbor Distance Ratio (NNDR)'
        }

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

def get_flair_model(model_name: str = 'flair/ner-english-large') -> SequenceTagger:
    """
    Get or load Flair model with caching and device management.
    """
    if not FLAIR_AVAILABLE:
        raise ImportError(
            "Flair is required for text privacy metrics but could not be imported. "
            f"Original error: {FLAIR_IMPORT_ERROR}"
        )
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for Flair-based metrics but is not installed.")

    with _model_lock:
        if model_name not in _model_cache:
            # Load the model
            _model_cache[model_name] = SequenceTagger.load(model_name)
            
            # Disable gradient computation for inference
            _model_cache[model_name].eval()
            
            # Move model to the best available device (compatible with all Flair versions)
            device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
            _model_cache[model_name] = _model_cache[model_name].to(device)
                
        return _model_cache[model_name]

def ensure_tensor_device(tensor, device):
    """
    Ensure a tensor is on the specified device.
    """
    if tensor is not None and TORCH_AVAILABLE:
        target = torch.device(device)
        if tensor.device != target:
            return tensor.to(target)
    return tensor

class PrivacyEvaluator:
    def __init__(self, synthetic_data: pd.DataFrame, original_data: pd.DataFrame, metadata: Dict, selected_metrics: List[str] = None, device: str = 'auto'):
        """
        Initialize the privacy evaluator.
        
        Args:
            synthetic_data: DataFrame containing synthetic data
            original_data: DataFrame containing original data
            metadata: Dictionary containing metadata about the data
            selected_metrics: List of specific metrics to run
            device: Device to use for computation ('auto', 'cpu', 'cuda')
        """
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.metadata = metadata
        self.logger = logging.getLogger(__name__)
        
        # Initialize smart cache
        self.cache = SmartCache('./cache')
        
        # Compute dataset fingerprints
        self.original_fingerprint = compute_dataset_fingerprint(original_data, metadata)
        self.synthetic_fingerprint = compute_dataset_fingerprint(synthetic_data, metadata)
        
        # Available metrics for selection
        self.available_metrics = [
            'exact_matches', 'membership_inference', 'tabular_privacy', 'text_privacy', 'anonymeter'
        ]
        
        # Use all metrics if none specified
        self.selected_metrics = (
            selected_metrics.copy() if selected_metrics else self.available_metrics.copy()
        )
        self.unavailable_metrics: Dict[str, str] = {}
        
        # Device management
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
                self.logger.info("Using CUDA for acceleration")
            else:
                self.device = 'cpu'
                if not TORCH_AVAILABLE:
                    self.logger.info("PyTorch not available — defaulting to CPU.")
                else:
                    self.logger.info("CUDA device not available — using CPU.")
        elif device == 'cuda':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
                self.logger.info("Using CUDA for acceleration")
            else:
                warn_reason = "PyTorch is not installed" if not TORCH_AVAILABLE else "CUDA device not available"
                self.logger.warning(f"CUDA requested but unavailable ({warn_reason}). Falling back to CPU.")
                self.device = 'cpu'
        else:  # cpu
            self.device = 'cpu'
            self.logger.info("Using CPU")
        
        # Handle optional text privacy dependency
        skip_text_privacy = False
        if 'text_privacy' in self.selected_metrics:
            if not TORCH_AVAILABLE:
                self.unavailable_metrics['text_privacy'] = "requires PyTorch to load Flair NER models"
                skip_text_privacy = True
            elif not FLAIR_AVAILABLE:
                self.unavailable_metrics['text_privacy'] = (
                    f"requires Flair; import failed with: {FLAIR_IMPORT_ERROR}"
                )
                skip_text_privacy = True
        
        # Initialize Flair NER model with caching when available
        self.ner_tagger = None
        if 'text_privacy' in self.selected_metrics and not skip_text_privacy:
            try:
                self.logger.info("Loading Flair NER model (this may take a few minutes)...")
                self.ner_tagger = get_flair_model()
                self.logger.info("Loaded Flair NER model successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Flair NER model: {str(e)}")
                self.unavailable_metrics['text_privacy'] = f"failed to load Flair model: {e}"
                self.selected_metrics.remove('text_privacy')
                self.ner_tagger = None
        elif skip_text_privacy:
            self.selected_metrics.remove('text_privacy')
        
        # Remove metrics that could not be prepared
        if self.unavailable_metrics:
            for metric in list(self.unavailable_metrics):
                if metric in self.selected_metrics:
                    self.selected_metrics.remove(metric)
        
        # Initialize spaCy for additional linguistic features
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("spaCy model loaded successfully")
        except OSError:
            self.logger.warning("spaCy model not found, attempting to download...")
            try:
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy model downloaded and loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to download spaCy model: {str(e)}")
                self.nlp = None
        except Exception as e:
            self.logger.warning(f"Failed to load spaCy model: {str(e)}")
            self.nlp = None
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Define nominal mention patterns
        self.nominal_patterns = {
            'family': {
                'mother', 'father', 'mom', 'dad', 'sister', 'brother', 'daughter', 'son',
                'wife', 'husband', 'spouse', 'partner', 'grandmother', 'grandfather',
                'aunt', 'uncle', 'cousin', 'niece', 'nephew', 'family', 'relative'
            },
            'role': {
                'teacher', 'student', 'doctor', 'patient', 'customer', 'client',
                'employee', 'employer', 'manager', 'director', 'officer', 'agent',
                'representative', 'consultant', 'advisor', 'expert', 'specialist',
                'professional', 'worker', 'staff', 'member', 'participant', 'user'
            },
            'relationship': {
                'friend', 'colleague', 'neighbor', 'partner', 'associate', 'companion',
                'acquaintance', 'contact', 'connection', 'ally', 'supporter', 'follower',
                'leader', 'mentor', 'mentee', 'supervisor', 'subordinate', 'peer'
            }
        }



    def evaluate(self) -> Dict:
        """
        Run selected privacy evaluations and return results.
        """
        self.logger.info("Starting privacy evaluation...")
        
        results = {}
        
        if 'exact_matches' in self.selected_metrics:
            results['exact_matches'] = self._evaluate_exact_matches()
        
        if 'membership_inference' in self.selected_metrics:
            results['membership_inference'] = self._evaluate_membership_inference()
        
        # Add data type specific evaluations
        # For mixed data (both text and structured), run both evaluations
        if 'tabular_privacy' in self.selected_metrics:
            self.logger.info("Running tabular data privacy evaluations...")
            tabular_results = self._evaluate_tabular_privacy()
            results.update(tabular_results)
            
        if 'text_privacy' in self.selected_metrics and self._is_text_data():
            self.logger.info("Running text-specific privacy evaluations...")
            text_results = self._evaluate_text_privacy()
            results.update(text_results)
            
        # Add Anonymeter evaluations
        if 'anonymeter' in self.selected_metrics:
            self.logger.info("Running Anonymeter privacy evaluations...")
            try:
                anonymeter_results = self._evaluate_reidentification_risks()
                results['anonymeter'] = anonymeter_results
                self.logger.info("Anonymeter evaluations completed successfully")
            except Exception as e:
                self.logger.error(f"Anonymeter evaluation failed: {str(e)}")
                results['anonymeter'] = {
                    'error': f"Anonymeter evaluation failed: {str(e)}",
                    'risk_level': 'unknown'
                }
        
        if self.unavailable_metrics:
            results['skipped_metrics'] = self.unavailable_metrics
        
        self.logger.info("Privacy evaluation completed")
        return results

    def _evaluate_exact_matches(self) -> Dict:
        """
        Evaluate the percentage of exact row matches between synthetic and original data.
        """
        matches = 0
        total = len(self.synthetic_data)
        
        for _, row in self.synthetic_data.iterrows():
            if any((self.original_data == row).all(axis=1)):
                matches += 1
                
        match_percentage = (matches / total) * 100
        return {
            'exact_match_percentage': match_percentage,
            'risk_level': 'high' if match_percentage > 5 else 'low'
        }

    def _evaluate_membership_inference(self) -> Dict:
        """
        Perform membership inference attack using a binary classifier.
        """
        self.logger.info("Starting membership inference attack evaluation...")
        
        # Get text columns from metadata that actually exist in the data
        text_columns = [col for col, info in self.metadata['columns'].items() 
                       if info['sdtype'] == 'text' and col in self.synthetic_data.columns]
        
        # Combine text data from both datasets
        if text_columns:
            self.logger.info("Processing text columns for membership inference...")
            # Combine all text columns into one for each dataset
            syn_text = self.synthetic_data[text_columns].apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            orig_text = self.original_data[text_columns].apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            
            # Create and fit vectorizer on combined text
            vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit features to avoid dimension mismatch
                min_df=2,  # Ignore terms that appear in only one document
                max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                stop_words='english'
            )
            vectorizer.fit(pd.concat([syn_text, orig_text]))
            
            # Transform both datasets
            syn_text_features = vectorizer.transform(syn_text)
            orig_text_features = vectorizer.transform(orig_text)
            
            # Get non-text columns
            non_text_cols = [col for col in self.synthetic_data.columns if col not in text_columns]
            
            if non_text_cols:
                self.logger.info("Processing non-text columns for membership inference...")
                # Concatenate for consistent one-hot encoding
                combined = pd.concat([
                    self.synthetic_data[non_text_cols],
                    self.original_data[non_text_cols]
                ], axis=0)

                # One-hot encode together
                combined_encoded = pd.get_dummies(combined).fillna(0)

                # Split back to synthetic and original
                syn_non_text = combined_encoded.iloc[:len(self.synthetic_data), :]
                orig_non_text = combined_encoded.iloc[len(self.synthetic_data):, :]

                # Combine with text features
                synthetic_features = np.hstack([syn_non_text.values, syn_text_features.toarray()])
                original_features = np.hstack([orig_non_text.values, orig_text_features.toarray()])
            else:
                synthetic_features = syn_text_features.toarray()
                original_features = orig_text_features.toarray()
        else:
            self.logger.info("Processing only non-text columns for membership inference...")
            # Concatenate for consistent one-hot encoding
            combined = pd.concat([
                self.synthetic_data,
                self.original_data
            ], axis=0)

            # One-hot encode together
            combined_encoded = pd.get_dummies(combined).fillna(0)

            # Split back to synthetic and original
            synthetic_features = combined_encoded.iloc[:len(self.synthetic_data), :].values
            original_features = combined_encoded.iloc[len(self.synthetic_data):, :].values
        
        self.logger.info(f"Feature dimensions - Synthetic: {synthetic_features.shape}, Original: {original_features.shape}")
        
        # Create labels: 1 for synthetic, 0 for original
        X = np.vstack([synthetic_features, original_features])
        y = np.concatenate([np.ones(len(synthetic_features)), np.zeros(len(original_features))])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train classifier
        self.logger.info("Training membership inference classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        
        # Calculate additional metrics
        synthetic_pred = clf.predict_proba(synthetic_features)[:, 1]
        original_pred = clf.predict_proba(original_features)[:, 1]
        
        syn_confidence = np.mean(synthetic_pred)
        orig_confidence = np.mean(original_pred)
        
        self.logger.info(f"Membership inference results - AUC: {auc_score:.3f}, "
                        f"Avg confidence (syn/orig): {syn_confidence:.3f}/{orig_confidence:.3f}")
        
        return {
            'mia_auc_score': auc_score,
            'synthetic_confidence': syn_confidence,
            'original_confidence': orig_confidence,
            'risk_level': 'high' if auc_score > 0.7 else 'low',
            'explanation': 'High risk: The model can easily distinguish between synthetic and real data. '
                         'This suggests the synthetic data is not preserving the statistical properties '
                         'of the real data well enough.'
        }

    def _evaluate_tabular_privacy(self) -> Dict:
        """
        Evaluate privacy risks for tabular data.
        """
        results = {}
        
        # Check for PII columns
        pii_columns = self._identify_pii_columns()
        if pii_columns:
            results['pii_analysis'] = self._analyze_pii_columns(pii_columns)
        
        # Evaluate structured privacy metrics (IMS, DCR, NNDR)
        self.logger.info("Starting structured privacy metrics evaluation...")
        try:
            # Let evaluate_structured_privacy_metrics use its default logic
            # which will exclude PII columns for both feature_cols and key_cols
            structured_metrics = self.evaluate_structured_privacy_metrics()
            results['structured_privacy_metrics'] = structured_metrics
            self.logger.info("Structured privacy metrics evaluation completed")
        except Exception as e:
            self.logger.error(f"Structured privacy metrics evaluation failed: {str(e)}")
            results['structured_privacy_metrics'] = {
                'error': f"Structured privacy metrics evaluation failed: {str(e)}",
                'risk_level': 'unknown'
            }
        except Exception as e:
            self.logger.error(f"Structured privacy metrics evaluation failed: {str(e)}")
            results['structured_privacy_metrics'] = {
                'error': f"Structured privacy metrics evaluation failed: {str(e)}",
                'risk_level': 'unknown'
            }
            
        # Evaluate re-identification risks using Anonymeter
        self.logger.info("Starting Anonymeter re-identification risk evaluation...")
        try:
            reid_risks = self._evaluate_reidentification_risks()
            results['reidentification_risks'] = reid_risks
            self.logger.info("Anonymeter re-identification risk evaluation completed")
        except Exception as e:
            self.logger.error(f"Anonymeter re-identification risk evaluation failed: {str(e)}")
            results['reidentification_risks'] = {
                'error': f"Re-identification risk evaluation failed: {str(e)}",
                'risk_level': 'unknown'
            }
        
        return results

    def _evaluate_text_privacy(self) -> Dict:
        """
        Evaluate privacy risks for text data.
        """
        results = {}
        
        # Extract and analyze named entities
        results['named_entities'] = self._analyze_named_entities()
        
        # Analyze nominal mentions
        results['nominal_mentions'] = self._analyze_nominal_mentions()
        
        # Analyze stylistic outliers
        results['stylistic_outliers'] = self._analyze_stylistic_outliers()
        
        return results

    def _prepare_features(self, data: pd.DataFrame, vectorizer: Optional[TfidfVectorizer] = None) -> Tuple[np.ndarray, Optional[TfidfVectorizer]]:
        """
        Prepare features for membership inference attack.
        
        Args:
            data: DataFrame to process
            vectorizer: Optional pre-fitted TF-IDF vectorizer
            
        Returns:
            Tuple of (features array, fitted vectorizer)
        """
        # Get text columns from metadata that actually exist in the data
        text_columns = [col for col, info in self.metadata['columns'].items() 
                       if info['sdtype'] == 'text' and col in data.columns]
        
        # Get non-text columns
        non_text_cols = [col for col in data.columns if col not in text_columns]
        
        # Process non-text columns
        non_text_data = data[non_text_cols].copy()
        
        # Convert categorical variables to one-hot encoding
        categorical_cols = non_text_data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            non_text_data = pd.get_dummies(non_text_data, columns=categorical_cols)
        
        # Handle missing values
        non_text_data = non_text_data.fillna(0)
        
        # Process text columns
        if text_columns:
            # Combine all text columns into one
            combined_text = data[text_columns].apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            
            if vectorizer is None:
                # Create and fit new vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=1000,  # Limit features to avoid dimension mismatch
                    min_df=2,  # Ignore terms that appear in only one document
                    max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                    stop_words='english'
                )
                text_features = vectorizer.fit_transform(combined_text)
            else:
                # Use existing vectorizer
                text_features = vectorizer.transform(combined_text)
            
            # Combine text and non-text features
            if len(non_text_cols) > 0:
                return np.hstack([non_text_data.values, text_features.toarray()]), vectorizer
            else:
                return text_features.toarray(), vectorizer
        else:
            return non_text_data.values, None

    def _identify_pii_columns(self) -> List[str]:
        """
        Identify columns that might contain PII.
        """
        pii_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
            'address': r'^\d+\s+[a-zA-Z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|place|pl|court|ct|cir|circle)[,\s]+[a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?$'
        }
        
        pii_columns = []
        for col in self.original_data.columns:
            if self.original_data[col].dtype == 'object':
                sample = self.original_data[col].dropna().astype(str)
                for pattern in pii_patterns.values():
                    if any(re.match(pattern, str(val)) for val in sample):
                        pii_columns.append(col)
                        break
                        
        return pii_columns

    def _analyze_pii_columns(self, pii_columns: List[str]) -> Dict:
        """
        Analyze PII columns for privacy risks.
        """
        results = {}
        for col in pii_columns:
            original_values = set(self.original_data[col].dropna())
            synthetic_values = set(self.synthetic_data[col].dropna())
            
            # Calculate exact match percentage
            matches = len(original_values.intersection(synthetic_values))
            total = len(original_values)
            match_percentage = (matches / total) * 100 if total > 0 else 0
            
            results[col] = {
                'exact_match_percentage': match_percentage,
                'risk_level': 'high' if match_percentage > 5 else 'low'
            }
            
        return results

    def _evaluate_reidentification_risks(self) -> Dict:
        """
        Evaluate re-identification risks using Anonymeter's singling out, linkability, and inference attacks.
        
        Returns:
            Dictionary containing risk scores and explanations for each attack type
        """
        try:
            self.logger.info("Preparing data for Anonymeter evaluation...")
            # Prepare data for Anonymeter
            # Convert categorical variables to string type
            syn_data = self.synthetic_data.copy()
            orig_data = self.original_data.copy()
            
            self.logger.info(f"Original data shape: {orig_data.shape}")
            self.logger.info(f"Synthetic data shape: {syn_data.shape}")
            
            # Convert boolean columns to int
            for col in syn_data.select_dtypes(include=['bool']).columns:
                syn_data[col] = syn_data[col].astype(int)
                orig_data[col] = orig_data[col].astype(int)
            
            # Convert other categorical variables to string
            for col in syn_data.select_dtypes(include=['object', 'category']).columns:
                syn_data[col] = syn_data[col].astype(str)
                orig_data[col] = orig_data[col].astype(str)
            
            # Handle missing values
            syn_data = syn_data.fillna('')
            orig_data = orig_data.fillna('')
            
            # Create a control dataset by sampling from original data
            control_data = orig_data.sample(n=min(len(orig_data), 1000), random_state=42)
            self.logger.info(f"Control data shape: {control_data.shape}")
            
            results = {}
            
            # 1. Singling Out Attack - Univariate
            self.logger.info("Starting univariate singling out attack...")
            singling_out_uni = SinglingOutEvaluator(
                ori=orig_data,
                syn=syn_data,
                control=control_data,
                n_attacks=300
            )
            try:
                self.logger.info("Running univariate singling out evaluation...")
                singling_out_uni.evaluate(mode='univariate')
                uni_results = singling_out_uni.results()
                self.logger.info(f"Univariate singling out results - Attack rate: {uni_results.attack_rate}")
                risk = uni_results.risk()
                
                # Check if attack is worse than baseline
                if uni_results.attack_rate.value <= uni_results.baseline_rate.value:
                    self.logger.warning("Univariate singling out attack is not better than baseline")
                    results['singling_out_univariate'] = {
                        'attack_rate': float(uni_results.attack_rate.value),
                        'baseline_rate': float(uni_results.baseline_rate.value),
                        'control_rate': float(uni_results.control_rate.value),
                        'risk': 0.0,  # Set risk to 0 if attack is not better than baseline
                        'error': float(risk.error) if hasattr(risk, 'error') else None,
                        'warning': 'Attack is not better than baseline model'
                    }
                else:
                    results['singling_out_univariate'] = {
                        'attack_rate': float(uni_results.attack_rate.value),
                        'baseline_rate': float(uni_results.baseline_rate.value),
                        'control_rate': float(uni_results.control_rate.value),
                        'risk': float(risk.value),
                        'error': float(risk.error) if hasattr(risk, 'error') else None
                    }
            except (RuntimeError, ValueError) as e:
                self.logger.warning(f"Univariate singling out evaluation failed: {str(e)}")
                results['singling_out_univariate'] = {
                    'error': str(e),
                    'risk_level': 'unknown'
                }
            
            # 2. Singling Out Attack - Multivariate
            self.logger.info("Starting multivariate singling out attack...")
            singling_out_multi = SinglingOutEvaluator(
                ori=orig_data,
                syn=syn_data,
                control=control_data,
                n_attacks=100,
                n_cols=4
            )
            try:
                self.logger.info("Running multivariate singling out evaluation...")
                singling_out_multi.evaluate(mode='multivariate')
                multi_results = singling_out_multi.results()
                self.logger.info(f"Multivariate singling out results - Attack rate: {multi_results.attack_rate}")
                risk = multi_results.risk()
                
                # Check if attack is worse than baseline
                if multi_results.attack_rate.value <= multi_results.baseline_rate.value:
                    self.logger.warning("Multivariate singling out attack is not better than baseline")
                    results['singling_out_multivariate'] = {
                        'attack_rate': float(multi_results.attack_rate.value),
                        'baseline_rate': float(multi_results.baseline_rate.value),
                        'control_rate': float(multi_results.control_rate.value),
                        'risk': 0.0,  # Set risk to 0 if attack is not better than baseline
                        'error': float(risk.error) if hasattr(risk, 'error') else None,
                        'warning': 'Attack is not better than baseline model'
                    }
                else:
                    results['singling_out_multivariate'] = {
                        'attack_rate': float(multi_results.attack_rate.value),
                        'baseline_rate': float(multi_results.baseline_rate.value),
                        'control_rate': float(multi_results.control_rate.value),
                        'risk': float(risk.value),
                        'error': float(risk.error) if hasattr(risk, 'error') else None
                    }
            except (RuntimeError, ValueError) as e:
                self.logger.warning(f"Multivariate singling out evaluation failed: {str(e)}")
                results['singling_out_multivariate'] = {
                    'error': str(e),
                    'risk_level': 'unknown'
                }
            
            # 3. Linkability Attack
            self.logger.info("Starting linkability attack...")
            # Define auxiliary columns for linkability attack
            aux_cols = [
                [col for col in orig_data.columns if col not in ['user_id']],
                ['user_id']
            ]
            
            linkability = LinkabilityEvaluator(
                ori=orig_data,
                syn=syn_data,
                control=control_data,
                n_attacks=50,
                aux_cols=aux_cols,
                n_neighbors=10
            )
            try:
                self.logger.info("Running linkability evaluation...")
                linkability.evaluate(n_jobs=-2)
                link_results = linkability.results()
                self.logger.info(f"Linkability results - Attack rate: {link_results.attack_rate}")
                risk = link_results.risk()
                
                # Check if attack is worse than baseline
                if link_results.attack_rate.value <= link_results.baseline_rate.value:
                    self.logger.warning("Linkability attack is not better than baseline")
                    results['linkability'] = {
                        'attack_rate': float(link_results.attack_rate.value),
                        'baseline_rate': float(link_results.baseline_rate.value),
                        'control_rate': float(link_results.control_rate.value),
                        'risk': 0.0,  # Set risk to 0 if attack is not better than baseline
                        'error': float(risk.error) if hasattr(risk, 'error') else None,
                        'warning': 'Attack is not better than baseline model'
                    }
                else:
                    results['linkability'] = {
                        'attack_rate': float(link_results.attack_rate.value),
                        'baseline_rate': float(link_results.baseline_rate.value),
                        'control_rate': float(link_results.control_rate.value),
                        'risk': float(risk.value),
                        'error': float(risk.error) if hasattr(risk, 'error') else None
                    }
            except (RuntimeError, ValueError) as e:
                self.logger.warning(f"Linkability evaluation failed: {str(e)}")
                results['linkability'] = {
                    'error': str(e),
                    'risk_level': 'unknown'
                }
            
            # 4. Inference Attack
            self.logger.info("Starting inference attack...")
            inference_results = []
            
            for secret in orig_data.columns:
                self.logger.info(f"Evaluating inference risk for secret column: {secret}")
                aux_cols = [col for col in orig_data.columns if col != secret]
                
                # Determine if regression or classification
                regression = pd.api.types.is_numeric_dtype(orig_data[secret])
                
                # Calculate appropriate number of attacks based on data size
                n_attacks = min(100, len(orig_data) // 2)  # Use at most 100 attacks or half the data size
                
                inference = InferenceEvaluator(
                    ori=orig_data,
                    syn=syn_data,
                    control=control_data,
                    aux_cols=aux_cols,
                    secret=secret,
                    regression=regression,
                    n_attacks=n_attacks
                )
                
                try:
                    self.logger.info(f"Running inference evaluation for {secret}...")
                    inference.evaluate(n_jobs=-2)
                    inf_results = inference.results()
                    self.logger.info(f"Inference results for {secret} - Attack rate: {inf_results.attack_rate}")
                    risk = inf_results.risk()
                    
                    # Check if attack is worse than baseline
                    if inf_results.attack_rate.value <= inf_results.baseline_rate.value:
                        self.logger.warning(f"Inference attack for {secret} is not better than baseline")
                        inference_results.append({
                            'secret_column': secret,
                            'attack_rate': float(inf_results.attack_rate.value),
                            'baseline_rate': float(inf_results.baseline_rate.value),
                            'control_rate': float(inf_results.control_rate.value),
                            'risk': 0.0,  # Set risk to 0 if attack is not better than baseline
                            'error': float(risk.error) if hasattr(risk, 'error') else None,
                            'warning': 'Attack is not better than baseline model'
                        })
                    else:
                        inference_results.append({
                            'secret_column': secret,
                            'attack_rate': float(inf_results.attack_rate.value),
                            'baseline_rate': float(inf_results.baseline_rate.value),
                            'control_rate': float(inf_results.control_rate.value),
                            'risk': float(risk.value),
                            'error': float(risk.error) if hasattr(risk, 'error') else None
                        })
                except (RuntimeError, ValueError) as e:
                    self.logger.warning(f"Inference evaluation failed for {secret}: {str(e)}")
                    inference_results.append({
                        'secret_column': secret,
                        'error': str(e),
                        'risk_level': 'unknown'
                    })
            
            results['inference'] = inference_results
            
            # Generate visualization
            self.logger.info("Generating privacy visualizations...")
            self._generate_privacy_visualizations(results)
            
            # Calculate overall risk
            overall_risk = max(
                results.get('singling_out_univariate', {}).get('risk', 0),
                results.get('singling_out_multivariate', {}).get('risk', 0),
                results.get('linkability', {}).get('risk', 0),
                max([r.get('risk', 0) for r in results.get('inference', [])], default=0)
            )
            
            results['overall_risk'] = {
                'risk_score': overall_risk,
                'risk_level': 'high' if overall_risk > 0.5 else 'low',
                'explanation': 'Overall risk is determined by the highest risk score among all attacks.'
            }
            
            self.logger.info("Anonymeter evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in re-identification risk evaluation: {str(e)}")
        return {
                'error': f"Failed to evaluate re-identification risks: {str(e)}",
                'risk_level': 'unknown'
            }

    def _generate_privacy_visualizations(self, results: Dict):
        """Generate visualizations for privacy evaluation results."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs('privacy_visualizations', exist_ok=True)
            
            # 1. Inference Risk Bar Plot
            if 'inference' in results and results['inference']:
                plt.figure(figsize=(12, 6))
                risks = [r.get('risk', 0) for r in results['inference'] if 'risk' in r]
                columns = [r['secret_column'] for r in results['inference'] if 'risk' in r]
                
                if risks and columns:  # Only plot if we have valid data
                    plt.bar(x=columns, height=risks, alpha=0.5, ecolor='black', capsize=10)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel("Measured inference risk")
                    plt.xlabel("Secret column")
                    plt.title("Inference Risk by Column")
                    plt.tight_layout()
                    plt.savefig('privacy_visualizations/inference_risk.png')
                plt.close()
            
            # 2. Attack Success Rates Comparison
            plt.figure(figsize=(10, 6))
            attack_types = ['Singling Out (Uni)', 'Singling Out (Multi)', 'Linkability']
            attack_rates = [
                results.get('singling_out_univariate', {}).get('attack_rate', 0),
                results.get('singling_out_multivariate', {}).get('attack_rate', 0),
                results.get('linkability', {}).get('attack_rate', 0)
            ]
            
            plt.bar(x=attack_types, height=attack_rates, alpha=0.5)
            plt.ylabel("Attack Success Rate")
            plt.title("Attack Success Rates Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('privacy_visualizations/attack_success_rates.png')
            plt.close()
            
            # 3. Risk Scores Heatmap
            risk_scores = {
                'Singling Out (Uni)': results.get('singling_out_univariate', {}).get('risk', 0),
                'Singling Out (Multi)': results.get('singling_out_multivariate', {}).get('risk', 0),
                'Linkability': results.get('linkability', {}).get('risk', 0),
                'Overall': results.get('overall_risk', {}).get('risk_score', 0)
            }
            
            plt.figure(figsize=(8, 4))
            sns.heatmap(
                pd.DataFrame([risk_scores]).T,
                annot=True,
                cmap='RdYlGn_r',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Risk Score'}
            )
            plt.title("Privacy Risk Scores")
            plt.tight_layout()
            plt.savefig('privacy_visualizations/risk_scores_heatmap.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating privacy visualizations: {str(e)}")

    def _process_entities_batch(self, texts: List[str]) -> List[Tuple]:
        """
        Process a batch of texts to extract named entities using Flair.
        """
        if self.ner_tagger is None:
            self.logger.warning("Skipping entity processing because the Flair NER model is unavailable.")
            return []
        
        results = []
        batch_size = 64 if self.device == 'cuda' else 16  # Larger batch for GPU
        
        # Ensure model is on the correct device (compatible with all Flair versions)
        self.ner_tagger = self.ner_tagger.to(self.device)
        
        # Process texts in batches with progress bar
        with tqdm(total=len(texts), desc="Processing entities", unit="text") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                sentences = [Sentence(text) for text in batch_texts]
                
                # Run NER on the batch (compatible with all Flair versions)
                self.ner_tagger.predict(sentences)
                
                # Process results
                for sentence in sentences:
                    valid_entities = []
                    for entity in sentence.get_spans('ner'):
                        # Skip if entity is too short or too long
                        if len(entity.text) < 2 or len(entity.text) > 50:
                            continue
                            
                        # Skip if entity is just a number or special character
                        if entity.text.isdigit() or not any(c.isalnum() for c in entity.text):
                            continue
                            
                        # Skip if entity contains URLs or email addresses
                        if 'http' in entity.text.lower() or '@' in entity.text or '.' in entity.text.split()[-1]:
                            continue
                        
                        # Skip if entity contains incomplete words or phrases
                        if any(word.endswith('-') or word.endswith('&') for word in entity.text.split()):
                            continue
                        
                        # Skip if entity is all uppercase (likely a label or category)
                        if entity.text.isupper() and len(entity.text) > 1:
                            continue
                        
                        # Map Flair labels to our categories
                        label = entity.tag
                        if label.startswith('B-') or label.startswith('I-'):
                            label = label[2:]  # Remove B- or I- prefix
                        
                        if label in {'PER', 'ORG', 'LOC', 'MISC'}:
                            valid_entities.append((entity.text, label))
                    
                    # Get number of tokens (excluding punctuation and whitespace)
                    if self.nlp is not None:
                        doc = self.nlp(sentence.text)
                        num_tokens = len([token for token in doc if not token.is_punct and not token.is_space])
                    else:
                        # Fallback: simple tokenization
                        num_tokens = len([word for word in sentence.text.split() if word.strip()])
                    
                    results.append((valid_entities, len(valid_entities), num_tokens))
                
                pbar.update(len(batch_texts))
        
        return results

    def _process_nominals_batch(self, texts: List[str]) -> List[Tuple]:
        """
        Process a batch of texts to extract nominal mentions with improved detection.
        Focuses only on nouns and proper nouns that represent people, roles, or relationships.
        """
        results = []
        batch_size = 100  # Larger batch size for spaCy
        
        # Common words to skip
        skip_words = {
            'nan', 'none', 'null', 'n/a', 'na', 'no', 'yes', 'ok', 'okay',
            'good', 'bad', 'great', 'nice', 'fine', 'well', 'ok', 'okay',
            'five', 'stars', 'star', 'rating', 'review', 'reviews', 'product',
            'item', 'items', 'price', 'quality', 'size', 'sizes', 'color',
            'colors', 'shipping', 'delivery', 'order', 'orders', 'buy', 'bought',
            'purchase', 'purchased', 'seller', 'sellers', 'buyer', 'buyers',
            'customer', 'customers', 'thank', 'thanks', 'thank you', 'please',
            'help', 'helpful', 'unhelpful', 'recommend', 'recommended'
        }
        
        # Process texts in batches with progress bar
        with tqdm(total=len(texts), desc="Processing nominals", unit="text") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                if self.nlp is not None:
                    docs = list(self.nlp.pipe(batch_texts))
                else:
                    # Fallback: simple processing without spaCy
                    docs = [None] * len(batch_texts)
                
                for doc in docs:
                    valid_nominals = set()
                    
                    if doc is None:
                        # Fallback processing without spaCy
                        num_tokens = len([word for word in batch_texts[0].split() if word.strip()])
                        results.append((valid_nominals, len(valid_nominals), num_tokens))
                        continue
                    
                    # Process each token
                    for token in doc:
                        # Skip if token is too short or too long
                        if len(token.text) < 2 or len(token.text) > 30:
                            continue
                            
                        # Skip if token is just a number or special character
                        if token.text.isdigit() or not any(c.isalnum() for c in token.text):
                            continue
                            
                        # Skip stopwords and common words
                        if token.text.lower() in self.stop_words or token.text.lower() in skip_words:
                            continue
                        
                        # Only consider nouns and proper nouns
                        if token.pos_ not in {'NOUN', 'PROPN'}:
                            continue
                        
                        # Skip if it's a common noun that's too short
                        if token.pos_ == 'NOUN' and len(token.text) < 3 and not token.text[0].isupper():
                            continue
                            
                        # Skip if it's a common verb form used as noun
                        if token.pos_ == 'NOUN' and token.lemma_ in {'be', 'have', 'do', 'get', 'make', 'take', 'go', 'come'}:
                            continue
                            
                        # Skip if it's a single letter or number
                        if token.pos_ == 'PROPN' and (len(token.text) <= 1 or token.text.isdigit()):
                            continue
                        
                        # Check if the token represents a person, role, or relationship
                        is_nominal = False
                        
                        # 1. Check if it's a proper noun (likely a person or organization)
                        if token.pos_ == 'PROPN':
                            is_nominal = True
                        
                        # 2. Check if it's a noun in subject position
                        elif token.dep_ in {'nsubj', 'nsubjpass'}:
                            is_nominal = True
                        
                        # 3. Check if it's a noun representing a role or relationship
                        elif token.pos_ == 'NOUN' and any(token.text.lower() in pattern_set 
                                                       for pattern_set in self.nominal_patterns.values()):
                            is_nominal = True
                        
                        if is_nominal:
                            valid_nominals.add(token.text)
                    
                    if doc is not None:
                        num_tokens = len([t for t in doc if not t.is_punct and not t.is_space])
                    else:
                        num_tokens = len([word for word in batch_texts[0].split() if word.strip()])
                    results.append((valid_nominals, len(valid_nominals), num_tokens))
                
                pbar.update(len(batch_texts))
        
        return results

    def _analyze_named_entities(self) -> Dict:
        """
        Analyze named entities in text data for both synthetic and real data.
        Uses cache if available.
        """
        text_column = self._get_text_column()
        if not text_column:
            return {'error': 'No text column found'}
        
        # Define helper function for grouping entities
        def group_entities(entities):
            grouped = {}
            for entity, type_ in entities:
                if type_ not in grouped:
                    grouped[type_] = []
                grouped[type_].append(entity)
            return grouped
            
        # Check cache for original data
        cached_orig_data = self.cache.load_cache('named_entities_original', self.original_fingerprint, text_column)
        if cached_orig_data is None:
            self.logger.info("Processing original data entities (not in cache)...")
            orig_texts = self.original_data[text_column].astype(str).tolist()
            orig_results = self._process_entities_batch(orig_texts)
            
            # Extract all entities
            orig_entities = set()
            for entities, _, _ in orig_results:
                orig_entities.update(entities)
            
            # Save to cache
            self.cache.save_cache('named_entities_original', self.original_fingerprint, 
                                {'entities_by_type': group_entities(orig_entities)}, text_column)
        else:
            self.logger.info("Using cached original data entities...")
            # Reconstruct entities set from entities_by_type
            orig_entities = set()
            for entity_type, entities in cached_orig_data['entities_by_type'].items():
                orig_entities.update((entity, entity_type) for entity in entities)
        
        # Process synthetic data (always process as it changes)
        self.logger.info("Processing synthetic data entities...")
        syn_texts = self.synthetic_data[text_column].astype(str).tolist()
        syn_results = self._process_entities_batch(syn_texts)
        
        # Extract all entities from synthetic data
        syn_entities = set()
        for entities, _, _ in syn_results:
            syn_entities.update(entities)
        
        # Save synthetic data entities to cache
        self.cache.save_cache('named_entities_synthetic', self.synthetic_fingerprint, 
                            {'entities_by_type': group_entities(syn_entities)}, text_column)
        
        # Calculate statistics
        syn_total_entities = len(syn_entities)
        syn_total_tokens = sum(num_tokens for _, _, num_tokens in syn_results)
        syn_avg_entity_density = syn_total_entities / syn_total_tokens if syn_total_tokens > 0 else 0
        
        orig_total_entities = len(orig_entities)
        orig_total_tokens = sum(num_tokens for _, _, num_tokens in orig_results) if 'orig_results' in locals() else 0
        orig_avg_entity_density = orig_total_entities / orig_total_tokens if orig_total_tokens > 0 else 0
        
        # Calculate entity overlap
        common_entities = syn_entities.intersection(orig_entities)
        overlap_percentage = (len(common_entities) / len(orig_entities) * 100) if orig_entities else 0
        
        return {
            'synthetic': {
                'total_entities': syn_total_entities,
                'avg_entity_density': syn_avg_entity_density,
                'risk_level': 'high' if syn_avg_entity_density > 0.1 else 'low'
            },
            'original': {
                'total_entities': orig_total_entities,
                'avg_entity_density': orig_avg_entity_density,
                'risk_level': 'high' if orig_avg_entity_density > 0.1 else 'low'
            },
            'overlap': {
                'overlap_percentage': overlap_percentage,
                'risk_level': 'high' if overlap_percentage > 50 else 'low'
            }
        }

    def _analyze_nominal_mentions(self) -> Dict:
        """
        Analyze nominal mentions in text data for both synthetic and real data.
        Uses cache if available.
        """
        text_column = self._get_text_column()
        if not text_column:
            return {'error': 'No text column found'}
            
        # Check cache for original data nominal mentions
        cached_orig_nominals = self.cache.load_cache('nominal_mentions_original', self.original_fingerprint, text_column)
        if cached_orig_nominals is None:
            self.logger.info("Processing original data nominal mentions (not in cache)...")
            orig_texts = self.original_data[text_column].astype(str).tolist()
            orig_results = self._process_nominals_batch(orig_texts)
            
            # Extract all nominal mentions
            orig_nominals = set()
            for nominals, _, _ in orig_results:
                orig_nominals.update(nominals)
            
            self.cache.save_cache('nominal_mentions_original', self.original_fingerprint, 
                                {'nominals': list(orig_nominals)}, text_column)
        else:
            self.logger.info("Using cached original data nominal mentions...")
            orig_nominals = set(cached_orig_nominals['nominals'])
        
        # Process synthetic data (always process as it changes)
        self.logger.info("Processing synthetic data nominal mentions...")
        syn_texts = self.synthetic_data[text_column].astype(str).tolist()
        syn_results = self._process_nominals_batch(syn_texts)
        
        # Extract all nominal mentions from synthetic data
        syn_nominals = set()
        for nominals, _, _ in syn_results:
            syn_nominals.update(nominals)
        
        # Save synthetic data nominal mentions to cache
        self.cache.save_cache('nominal_mentions_synthetic', self.synthetic_fingerprint, 
                            {'nominals': list(syn_nominals)}, text_column)
        
        # Calculate statistics
        syn_total_nominals = len(syn_nominals)
        syn_total_tokens = sum(num_tokens for _, _, num_tokens in syn_results)
        syn_avg_nominal_density = syn_total_nominals / syn_total_tokens if syn_total_tokens > 0 else 0
        
        orig_total_nominals = len(orig_nominals)
        orig_total_tokens = sum(num_tokens for _, _, num_tokens in orig_results) if 'orig_results' in locals() else 0
        orig_avg_nominal_density = orig_total_nominals / orig_total_tokens if orig_total_tokens > 0 else 0
        
        # Calculate nominal mention overlap
        common_nominals = syn_nominals.intersection(orig_nominals)
        overlap_percentage = (len(common_nominals) / len(orig_nominals) * 100) if orig_nominals else 0
        
        return {
            'synthetic': {
                'total_nominals': syn_total_nominals,
                'avg_nominal_density': syn_avg_nominal_density,
                'risk_level': 'high' if syn_avg_nominal_density > 0.15 else 'low'
            },
            'original': {
                'total_nominals': orig_total_nominals,
                'avg_nominal_density': orig_avg_nominal_density,
                'risk_level': 'high' if orig_avg_nominal_density > 0.15 else 'low'
            },
            'overlap': {
                'overlap_percentage': overlap_percentage,
                'risk_level': 'high' if overlap_percentage > 50 else 'low'
            }
        }

    def _analyze_stylistic_outliers(self) -> Dict:
        """
        Analyze stylistic outliers in text data for both synthetic and real data.
        Uses cache if available.
        """
        text_column = self._get_text_column()
        if not text_column:
            return {'error': 'No text column found'}
        
        # Check cache for original data stylistic outliers
        cached_orig_outliers = self.cache.load_cache('stylistic_outliers_original', self.original_fingerprint, text_column)
        if cached_orig_outliers is None:
            self.logger.info("Processing original data stylistic outliers (not in cache)...")
            orig_texts = self.original_data[text_column].astype(str).tolist()
            orig_embeddings = self._generate_text_embeddings(orig_texts)
            orig_distances = cosine_distances(orig_embeddings)
            orig_mean_dists = np.mean(orig_distances, axis=1)
            orig_global_mean = np.mean(orig_mean_dists)
            orig_global_std = np.std(orig_mean_dists)
            
            # Find texts that are significantly different from others
            outlier_scores = (orig_mean_dists - orig_global_mean) / orig_global_std
            outlier_threshold = 2.0  # Texts more than 2 standard deviations away
            
            # Get unique outlier texts
            outlier_texts = []
            seen_texts = set()
            for text, score in zip(orig_texts, outlier_scores):
                if score > outlier_threshold and text not in seen_texts:
                    outlier_texts.append(text)
                    seen_texts.add(text)
            
            self.cache.save_cache('stylistic_outliers_original', self.original_fingerprint, 
                                {'outlier_texts': outlier_texts}, text_column)
        else:
            self.logger.info("Using cached original data stylistic outliers...")
            outlier_texts = cached_orig_outliers['outlier_texts']
            orig_outliers = np.array([text in outlier_texts for text in self.original_data[text_column].astype(str).tolist()])
        
        # Process synthetic data (always process as it changes)
        self.logger.info("Processing synthetic data stylistic outliers...")
        syn_texts = self.synthetic_data[text_column].astype(str).tolist()
        syn_embeddings = self._generate_text_embeddings(syn_texts)
        syn_distances = cosine_distances(syn_embeddings)
        syn_mean_dists = np.mean(syn_distances, axis=1)
        syn_global_mean = np.mean(syn_mean_dists)
        syn_global_std = np.std(syn_mean_dists)
        
        # Find texts that are significantly different from others
        outlier_scores = (syn_mean_dists - syn_global_mean) / syn_global_std
        outlier_threshold = 2.0  # Texts more than 2 standard deviations away
        
        # Get unique outlier texts
        outlier_texts = []
        seen_texts = set()
        for text, score in zip(syn_texts, outlier_scores):
            if score > outlier_threshold and text not in seen_texts:
                outlier_texts.append(text)
                seen_texts.add(text)
        
        self.cache.save_cache('stylistic_outliers_synthetic', self.synthetic_fingerprint, 
                            {'outlier_texts': outlier_texts}, text_column)
        
        # Get original outlier count for comparison
        orig_outlier_count = len(cached_orig_outliers['outlier_texts']) if cached_orig_outliers else 0
        
        return {
            'synthetic': {
                'num_outliers': len(outlier_texts),
                'outlier_percentage': float(len(outlier_texts) / len(syn_texts) * 100),
                'risk_level': 'high' if len(outlier_texts) / len(syn_texts) > 0.1 else 'low'
            },
            'original': {
                'num_outliers': orig_outlier_count,
                'outlier_percentage': float(orig_outlier_count / len(self.original_data) * 100),
                'risk_level': 'high' if orig_outlier_count / len(self.original_data) > 0.1 else 'low'
            },
            'comparison': {
                'outlier_ratio': float(len(outlier_texts) / orig_outlier_count) if orig_outlier_count > 0 else float('inf'),
                'risk_level': 'high' if abs(len(outlier_texts) / len(syn_texts) - orig_outlier_count / len(self.original_data)) > 0.1 else 'low'
            }
        }

    def _get_text_column(self) -> Optional[str]:
        """
        Identify the text column in the dataset.
        """
        if 'text_columns' in self.metadata:
            text_col = self.metadata['text_columns'][0]
            # Check if the text column actually exists in the data
            if text_col in self.synthetic_data.columns:
                return text_col
            else:
                self.logger.warning(f"Text column '{text_col}' from metadata not found in data")
                return None
        return None

    def _is_text_data(self) -> bool:
        """
        Check if the data is text data.
        """
        return 'text_columns' in self.metadata

    def _generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        # Tokenize and remove stopwords
        tokenized_texts = [
            [word for word in word_tokenize(text.lower()) if word not in self.stop_words]
            for text in texts
        ]
        
        # Train Word2Vec model
        model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        
        # Generate embeddings
        embeddings = []
        for tokens in tokenized_texts:
            if tokens:
                vectors = [model.wv[word] for word in tokens if word in model.wv]
                if vectors:
                    embeddings.append(np.mean(vectors, axis=0))
                else:
                    embeddings.append(np.zeros(model.vector_size))
            else:
                embeddings.append(np.zeros(model.vector_size))
                
        return np.array(embeddings)

    def evaluate_structured_privacy_metrics(self, feature_cols=None, key_cols=None):
        """
        评估结构化数据的隐私指标，包括IMS、DCR、NNDR。
        feature_cols: 用于距离计算的特征列
        key_cols: 用于完全匹配的主键列
        返回：dict，包含每个指标的详细分数和判定
        """
        self.logger.info(f"Evaluating structured privacy metrics with feature_cols: {feature_cols}, key_cols: {key_cols}")
        
        # Get all non-text columns (excluding all text columns)
        all_tabular_cols = [col for col, info in self.metadata['columns'].items() 
                           if info['sdtype'] in ['numerical', 'categorical', 'boolean', 'datetime'] 
                           and info['sdtype'] != 'text'  # Exclude all text columns
                           and col in self.synthetic_data.columns and col in self.original_data.columns]
        
        # For IMS, we should exclude PII columns to avoid exact matches
        non_pii_tabular_cols = [col for col in all_tabular_cols 
                               if not self.metadata['columns'][col].get('pii', False)]
        
        self.logger.info(f"all_tabular_cols: {all_tabular_cols}")
        self.logger.info(f"non_pii_tabular_cols: {non_pii_tabular_cols}")
        
        # Validate that columns exist in both datasets
        if feature_cols:
            feature_cols = [col for col in feature_cols 
                           if col in self.synthetic_data.columns and col in self.original_data.columns]
        else:
            # Use non-PII tabular columns for feature analysis (same as IMS)
            feature_cols = non_pii_tabular_cols if non_pii_tabular_cols else all_tabular_cols
        
        if key_cols:
            key_cols = [col for col in key_cols 
                       if col in self.synthetic_data.columns and col in self.original_data.columns]
        else:
            # Use non-PII tabular columns for IMS calculation (avoid exact matches from PII)
            key_cols = non_pii_tabular_cols if non_pii_tabular_cols else all_tabular_cols
        
        self.logger.info(f"Final feature_cols: {feature_cols}")
        self.logger.info(f"Final key_cols: {key_cols}")
        
        # Debug: Show sample rows for IMS calculation
        if key_cols:
            self.logger.info(f"Sample synthetic row for IMS: {self.synthetic_data[key_cols].iloc[0].to_dict()}")
            self.logger.info(f"Sample original row for IMS: {self.original_data[key_cols].iloc[0].to_dict()}")
        
        # Ensure we have at least some columns to work with
        if not feature_cols:
            self.logger.warning("No feature columns available for structured privacy metrics")
            return {
                'IMS': {'error': 'No feature columns available'},
                'DCR': {'error': 'No feature columns available'},
                'NNDR': {'error': 'No feature columns available'}
            }
        
        try:
            ims = identical_match_share(self.synthetic_data, self.original_data, key_cols=key_cols)
            self.logger.info(f"IMS evaluation completed: {ims}")
        except Exception as e:
            self.logger.error(f"IMS evaluation failed: {str(e)}")
            ims = {'error': f'IMS evaluation failed: {str(e)}'}
        
        try:
            dcr = distance_to_closest_records(self.synthetic_data, self.original_data, feature_cols=feature_cols)
            self.logger.info(f"DCR evaluation completed: {dcr}")
        except Exception as e:
            self.logger.error(f"DCR evaluation failed: {str(e)}")
            dcr = {'error': f'DCR evaluation failed: {str(e)}'}
        
        try:
            nndr = nearest_neighbor_distance_ratio(self.synthetic_data, self.original_data, feature_cols=feature_cols)
            self.logger.info(f"NNDR evaluation completed: {nndr}")
        except Exception as e:
            self.logger.error(f"NNDR evaluation failed: {str(e)}")
            nndr = {'error': f'NNDR evaluation failed: {str(e)}'}
        
        return {'IMS': ims, 'DCR': dcr, 'NNDR': nndr}
