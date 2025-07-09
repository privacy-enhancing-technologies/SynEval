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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """Get appropriate model based on task type."""
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
                
                # Combine with text features if they exist
                if text_columns:
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
            
            # Train on real data
            logger.info("Training model on real data...")
            model_real.fit(X_train_real, y_train_real)
            real_pred = model_real.predict(X_test)
            
            # Train on synthetic data
            logger.info("Training model on synthetic data...")
            model_syn.fit(X_train_syn, y_train_syn)
            syn_pred = model_syn.predict(X_test)
            
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
