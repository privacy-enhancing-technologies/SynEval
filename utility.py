#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

class UtilityEvaluator:
    def __init__(self, 
                 synthetic_data: pd.DataFrame,
                 original_data: pd.DataFrame,
                 metadata: Dict,
                 input_columns: List[str],
                 output_columns: List[str],
                 task_type: str = 'auto'):
        """
        Initialize the utility evaluator.
        
        Args:
            synthetic_data: Synthetic dataset
            original_data: Original dataset
            metadata: Dictionary containing column types and other metadata
            input_columns: List of columns to use as features
            output_columns: List of columns to predict
            task_type: Type of task ('classification', 'regression', or 'auto')
        """
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.metadata = metadata
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.task_type = self._determine_task_type(task_type)
        
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
                logger.info("GPU acceleration available - using CUDA")
            else:
                logger.info("GPU not available - using CPU")
        
    def _determine_task_type(self, task_type: str) -> str:
        """Determine the type of task based on output columns."""
        if task_type != 'auto':
            return task_type
            
        for col in self.output_columns:
            if col in self.metadata['columns']:
                col_type = self.metadata['columns'][col]['sdtype']
                if col_type == 'text':
                    return 'text_classification'
                elif col_type == 'categorical':
                    return 'classification'
                else:
                    return 'regression'
            else:
                logger.warning(f"Column {col} not found in metadata")
                return 'classification'  # default to classification if column not found
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input and output data for model training."""
        X = data[self.input_columns]
        y = data[self.output_columns]
        
        # Handle text columns
        text_columns = [col for col in self.input_columns 
                       if col in self.metadata['columns'] and 
                       self.metadata['columns'][col]['sdtype'] == 'text']
        
        if text_columns:
            # Fill NaN values with empty string for text columns
            text_data = data[text_columns].fillna('')
            vectorizer = TfidfVectorizer(max_features=1000)
            text_features = vectorizer.fit_transform(text_data[text_columns[0]])
            
            # Combine text features with other features
            other_columns = [col for col in self.input_columns if col not in text_columns]
            if other_columns:
                # Fill NaN values with 0 for numerical columns
                other_features = data[other_columns].fillna(0).values
                X = np.hstack([text_features.toarray(), other_features])
            else:
                X = text_features.toarray()
        
        # Handle NaN values in output columns
        if isinstance(y, pd.DataFrame):
            y = y.fillna(y.mean() if self.task_type == 'regression' else y.mode().iloc[0])
        else:
            y = pd.Series(y).fillna(pd.Series(y).mean() if self.task_type == 'regression' else pd.Series(y).mode().iloc[0])
        
        return X, y
    
    def _get_model(self):
        """Get appropriate model based on task type with GPU acceleration support."""
        if DEVICE == 'cuda':
            # Use GPU-accelerated models when available
            try:
                import xgboost as xgb
                if self.task_type == 'text_classification':
                    return Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=1000)),
                        ('classifier', xgb.XGBClassifier(n_estimators=100, tree_method='gpu_hist'))
                    ])
                elif self.task_type == 'classification':
                    return xgb.XGBClassifier(n_estimators=100, tree_method='gpu_hist')
                else:  # regression
                    return xgb.XGBRegressor(n_estimators=100, tree_method='gpu_hist')
            except ImportError:
                logger.info("XGBoost not available, falling back to scikit-learn models")
                # Fallback to scikit-learn models
                if self.task_type == 'text_classification':
                    return Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=1000)),
                        ('classifier', RandomForestClassifier(n_estimators=100))
                    ])
                elif self.task_type == 'classification':
                    return RandomForestClassifier(n_estimators=100)
                else:  # regression
                    return RandomForestRegressor(n_estimators=100)
        else:
            # CPU models
            if self.task_type == 'text_classification':
                return Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=1000)),
                    ('classifier', RandomForestClassifier(n_estimators=100))
                ])
            elif self.task_type == 'classification':
                return RandomForestClassifier(n_estimators=100)
            else:  # regression
                return RandomForestRegressor(n_estimators=100)
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate model performance based on task type."""
        if self.task_type in ['classification', 'text_classification']:
            return accuracy_score(y_true, y_pred)
        else:  # regression
            return r2_score(y_true, y_pred)
    
    def evaluate(self) -> Dict:
        """
        Evaluate utility using TSTR methodology.
        
        Returns:
            Dict: Dictionary containing task information and classification reports for both models
        """
        try:
            # Get synthetic data size
            syn_size = len(self.synthetic_data)
            logger.info(f"Using all synthetic data ({syn_size} samples) for training")
            
            # Sample from real data for training
            if len(self.original_data) > syn_size:
                train_data = self.original_data.sample(n=syn_size, random_state=42)
                remaining_data = self.original_data.drop(train_data.index)
                # Sample same amount from remaining data for testing
                test_data = remaining_data.sample(n=syn_size, random_state=43)
            else:
                train_data = self.original_data
                test_data = pd.DataFrame()  # Empty test set if real data is smaller than synthetic
                logger.warning("Real data is smaller than synthetic data. No test set available.")
            
            logger.info(f"Training size: {syn_size}, Test size: {len(test_data)}")
            
            # Prepare features and targets for all datasets
            # Training data (real)
            X_train_real = train_data[self.input_columns]
            y_train_real = train_data[self.output_columns]
            
            # Training data (synthetic)
            X_train_syn = self.synthetic_data[self.input_columns]
            y_train_syn = self.synthetic_data[self.output_columns]
            
            # Test data (from remaining real data)
            if not test_data.empty:
                X_test = test_data[self.input_columns]
                y_test = test_data[self.output_columns]
            else:
                return {
                    'error': 'No test data available - real data is smaller than synthetic data',
                    'task_type': self.task_type,
                    'input_columns': self.input_columns,
                    'output_columns': self.output_columns
                }
            
            # Handle text columns and prepare features
            text_columns = [col for col in self.input_columns 
                          if col in self.metadata['columns'] and 
                          self.metadata['columns'][col]['sdtype'] == 'text']
            
            if text_columns:
                # Initialize vectorizer
                vectorizer = TfidfVectorizer(max_features=1000)
                
                # Fit vectorizer on training data and transform all datasets
                X_train_real = vectorizer.fit_transform(X_train_real[text_columns[0]].fillna(''))
                X_train_syn = vectorizer.transform(X_train_syn[text_columns[0]].fillna(''))
                X_test = vectorizer.transform(X_test[text_columns[0]].fillna(''))
                
                # Convert to GPU tensors if available for faster computation
                if DEVICE == 'cuda':
                    import torch
                    X_train_real = torch.tensor(X_train_real.toarray(), device='cuda', dtype=torch.float32)
                    X_train_syn = torch.tensor(X_train_syn.toarray(), device='cuda', dtype=torch.float32)
                    X_test = torch.tensor(X_test.toarray(), device='cuda', dtype=torch.float32)
                else:
                    # Convert to dense arrays if needed
                    X_train_real = X_train_real.toarray()
                    X_train_syn = X_train_syn.toarray()
                    X_test = X_test.toarray()
            
            # Handle non-text columns if any
            other_columns = [col for col in self.input_columns if col not in text_columns]
            if other_columns:
                # Fill NaN values with 0 for numerical columns
                X_train_real_other = train_data[other_columns].fillna(0).values
                X_train_syn_other = self.synthetic_data[other_columns].fillna(0).values
                X_test_other = test_data[other_columns].fillna(0).values
                
                # Convert to GPU tensors if available
                if DEVICE == 'cuda':
                    import torch
                    X_train_real_other = torch.tensor(X_train_real_other, device='cuda', dtype=torch.float32)
                    X_train_syn_other = torch.tensor(X_train_syn_other, device='cuda', dtype=torch.float32)
                    X_test_other = torch.tensor(X_test_other, device='cuda', dtype=torch.float32)
                
                # Combine with text features if they exist
                if text_columns:
                    if DEVICE == 'cuda':
                        X_train_real = torch.cat([X_train_real, X_train_real_other], dim=1)
                        X_train_syn = torch.cat([X_train_syn, X_train_syn_other], dim=1)
                        X_test = torch.cat([X_test, X_test_other], dim=1)
                    else:
                        X_train_real = np.hstack([X_train_real, X_train_real_other])
                        X_train_syn = np.hstack([X_train_syn, X_train_syn_other])
                        X_test = np.hstack([X_test, X_test_other])
                else:
                    X_train_real = X_train_real_other
                    X_train_syn = X_train_syn_other
                    X_test = X_test_other
            
            # Handle target variable
            if isinstance(y_train_real, pd.DataFrame):
                y_train_real = y_train_real[self.output_columns[0]]
                y_train_syn = y_train_syn[self.output_columns[0]]
                y_test = y_test[self.output_columns[0]]
            
            # Fill NaN values in target
            y_train_real = y_train_real.fillna(y_train_real.mean() if self.task_type == 'regression' else y_train_real.mode().iloc[0])
            y_train_syn = y_train_syn.fillna(y_train_syn.mean() if self.task_type == 'regression' else y_train_syn.mode().iloc[0])
            y_test = y_test.fillna(y_test.mean() if self.task_type == 'regression' else y_test.mode().iloc[0])
            
            # Train two separate models
            model_real = self._get_model()
            model_syn = self._get_model()
            
            # Convert data back to CPU if using GPU tensors for scikit-learn models
            if DEVICE == 'cuda' and not hasattr(model_real, 'tree_method'):  # Not XGBoost
                import torch
                X_train_real_cpu = X_train_real.cpu().numpy()
                X_train_syn_cpu = X_train_syn.cpu().numpy()
                X_test_cpu = X_test.cpu().numpy()
            else:
                X_train_real_cpu = X_train_real
                X_train_syn_cpu = X_train_syn
                X_test_cpu = X_test
            
            # Train on real data
            logger.info("Training model on real data...")
            model_real.fit(X_train_real_cpu, y_train_real)
            real_pred = model_real.predict(X_test_cpu)
            
            # Train on synthetic data
            logger.info("Training model on synthetic data...")
            model_syn.fit(X_train_syn_cpu, y_train_syn)
            syn_pred = model_syn.predict(X_test_cpu)
            
            # Get classification reports
            from sklearn.metrics import classification_report
            real_report = classification_report(y_test, real_pred, output_dict=True)
            syn_report = classification_report(y_test, syn_pred, output_dict=True)
            
            results = {
                'task_type': self.task_type,
                'input_columns': self.input_columns,
                'output_columns': self.output_columns,
                'training_size': syn_size,
                'test_size': len(test_data),
                'total_samples': len(self.original_data),
                'real_data_model': real_report,
                'synthetic_data_model': syn_report
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in utility evaluation: {str(e)}")
            return {
                'error': str(e),
                'task_type': self.task_type,
                'input_columns': self.input_columns,
                'output_columns': self.output_columns
            }
