"""
Pytest configuration and shared fixtures for SynEval tests
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {
        "columns": {
            "id": {
                "sdtype": "numerical",
                "pii": False,
                "is_primary_key": True
            },
            "rating": {
                "sdtype": "categorical",
                "values": [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            "title": {
                "sdtype": "text"
            },
            "text": {
                "sdtype": "text"
            },
            "price": {
                "sdtype": "numerical"
            },
            "category": {
                "sdtype": "categorical",
                "values": ["electronics", "clothing", "books", "home"]
            },
            "verified_purchase": {
                "sdtype": "boolean",
                "values": [True, False]
            }
        },
        "text_columns": ["title", "text"],
        "utility": {
            "input_columns": ["text"],
            "output_columns": ["rating"],
            "task_type": "classification"
        }
    }


@pytest.fixture
def sample_original_data():
    """Sample original data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(1, n_samples + 1),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'title': [f"Product {i}" for i in range(1, n_samples + 1)],
        'text': [
            f"This is a great product with excellent quality. I highly recommend it! Rating: {i//20 + 1} stars."
            for i in range(n_samples)
        ],
        'price': np.random.normal(50, 20, n_samples).round(2),
        'category': np.random.choice(['electronics', 'clothing', 'books', 'home'], n_samples),
        'verified_purchase': np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_synthetic_data():
    """Sample synthetic data for testing"""
    np.random.seed(123)
    n_samples = 100
    
    data = {
        'id': range(1, n_samples + 1),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'title': [f"Synthetic Product {i}" for i in range(1, n_samples + 1)],
        'text': [
            f"This synthetic product has good quality and I would recommend it. Rating: {i//20 + 1} stars."
            for i in range(n_samples)
        ],
        'price': np.random.normal(55, 25, n_samples).round(2),
        'category': np.random.choice(['electronics', 'clothing', 'books', 'home'], n_samples),
        'verified_purchase': np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_csv_files(temp_dir, sample_original_data, sample_synthetic_data, sample_metadata):
    """Create sample CSV and JSON files for testing"""
    original_path = os.path.join(temp_dir, "original.csv")
    synthetic_path = os.path.join(temp_dir, "synthetic.csv")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    
    sample_original_data.to_csv(original_path, index=False)
    sample_synthetic_data.to_csv(synthetic_path, index=False)
    
    with open(metadata_path, 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    return {
        'original': original_path,
        'synthetic': synthetic_path,
        'metadata': metadata_path
    }


@pytest.fixture
def mock_nltk_data():
    """Mock NLTK data download to avoid network calls in tests"""
    import nltk
    from unittest.mock import patch
    
    with patch('nltk.download') as mock_download:
        mock_download.return_value = True
        yield mock_download


@pytest.fixture
def mock_flair_models():
    """Mock Flair models to avoid downloading large models in tests"""
    from unittest.mock import Mock, patch
    
    mock_sentence = Mock()
    mock_sentence.get_labels.return_value = [Mock(value='POSITIVE', score=0.8)]
    
    mock_tagger = Mock()
    mock_tagger.predict.return_value = None
    
    with patch('flair.models.TextClassifier.load') as mock_sentiment, \
         patch('flair.models.SequenceTagger.load') as mock_ner:
        mock_sentiment.return_value = Mock()
        mock_ner.return_value = mock_tagger
        yield {
            'sentiment': mock_sentiment,
            'ner': mock_ner
        }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set environment variables for testing
    os.environ['SYNEVAL_TEST_MODE'] = 'true'
    
    # Create test directories
    os.makedirs('test_plots', exist_ok=True)
    os.makedirs('test_cache', exist_ok=True)
    
    yield
    
    # Cleanup after test
    import shutil
    if os.path.exists('test_plots'):
        shutil.rmtree('test_plots')
    if os.path.exists('test_cache'):
        shutil.rmtree('test_cache') 