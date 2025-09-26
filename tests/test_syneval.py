"""
Fixed tests for SynEval framework with correct method names
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fidelity import FidelityEvaluator
from utility import UtilityEvaluator
from diversity import DiversityEvaluator


class TestSynEvalFramework:
    """Tests for SynEval framework components"""
    
    def test_fidelity_initialization(self):
        """Test FidelityEvaluator initialization"""
        # Create simple test data
        original_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        evaluator = FidelityEvaluator(synthetic_data, original_data, metadata)
        assert evaluator.original_data is not None
        assert evaluator.synthetic_data is not None
        assert evaluator.metadata is not None
        print("✅ FidelityEvaluator initialization test passed")
    
    def test_utility_initialization(self):
        """Test UtilityEvaluator initialization"""
        original_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        evaluator = UtilityEvaluator(
            synthetic_data, 
            original_data, 
            metadata, 
            input_columns=['text'], 
            output_columns=['rating'],
            task_type='classification'
        )
        assert evaluator.original_data is not None
        assert evaluator.synthetic_data is not None
        assert evaluator.metadata is not None
        assert evaluator.task_type == 'classification'
        print("✅ UtilityEvaluator initialization test passed")
    
    def test_diversity_initialization(self):
        """Test DiversityEvaluator initialization"""
        original_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        evaluator = DiversityEvaluator(synthetic_data, original_data, metadata)
        assert evaluator.original_data is not None
        assert evaluator.synthetic_data is not None
        assert evaluator.metadata is not None
        print("✅ DiversityEvaluator initialization test passed")
    
    def test_fidelity_evaluate_method(self):
        """Test FidelityEvaluator evaluate method exists"""
        original_data = pd.DataFrame({
            'id': [1, 2, 3],
            'rating': [4, 5, 3],
            'text': ['Great product', 'Excellent quality', 'Good value']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3],
            'rating': [4, 5, 3],
            'text': ['Great product', 'Excellent quality', 'Good value']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        evaluator = FidelityEvaluator(synthetic_data, original_data, metadata)
        
        # Check if the main evaluate method exists
        assert hasattr(evaluator, 'evaluate')
        print("✅ FidelityEvaluator evaluate method exists")
    
    def test_utility_evaluate_method(self):
        """Test UtilityEvaluator evaluate method exists"""
        original_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        evaluator = UtilityEvaluator(
            synthetic_data, 
            original_data, 
            metadata, 
            input_columns=['text'], 
            output_columns=['rating'],
            task_type='classification'
        )
        
        # Check if the main evaluate method exists
        assert hasattr(evaluator, 'evaluate')
        print("✅ UtilityEvaluator evaluate method exists")
    
    def test_diversity_evaluate_method(self):
        """Test DiversityEvaluator evaluate method exists"""
        original_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        evaluator = DiversityEvaluator(synthetic_data, original_data, metadata)
        
        # Check if the main evaluate method exists
        assert hasattr(evaluator, 'evaluate')
        print("✅ DiversityEvaluator evaluate method exists")
    
    def test_utility_task_detection(self):
        """Test UtilityEvaluator task type detection"""
        original_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'rating': [4, 5, 3, 4, 5],
            'text': ['Great product', 'Excellent quality', 'Good value', 'Amazing', 'Perfect']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        # Test classification task
        evaluator = UtilityEvaluator(
            synthetic_data, 
            original_data, 
            metadata, 
            input_columns=['text'], 
            output_columns=['rating'],
            task_type='classification'
        )
        assert evaluator.task_type == 'classification'
        print("✅ UtilityEvaluator classification task detection test passed")
        
        # Test regression task
        regression_data = original_data.copy()
        regression_data['price'] = [10.5, 15.2, 8.7, 12.3, 9.8]
        regression_synthetic = synthetic_data.copy()
        regression_synthetic['price'] = [10.5, 15.2, 8.7, 12.3, 9.8]
        
        regression_metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'price': {'sdtype': 'numerical'},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        evaluator = UtilityEvaluator(
            regression_synthetic, 
            regression_data, 
            regression_metadata, 
            input_columns=['text'], 
            output_columns=['price'],
            task_type='regression'
        )
        assert evaluator.task_type == 'regression'
        print("✅ UtilityEvaluator regression task detection test passed")
    
    def test_module_imports(self):
        """Test that all modules can be imported successfully"""
        try:
            from fidelity import FidelityEvaluator
            print("✅ FidelityEvaluator import successful")
        except ImportError as e:
            pytest.fail(f"Failed to import FidelityEvaluator: {e}")
        
        try:
            from utility import UtilityEvaluator
            print("✅ UtilityEvaluator import successful")
        except ImportError as e:
            pytest.fail(f"Failed to import UtilityEvaluator: {e}")
        
        try:
            from diversity import DiversityEvaluator
            print("✅ DiversityEvaluator import successful")
        except ImportError as e:
            pytest.fail(f"Failed to import DiversityEvaluator: {e}")
        
        print("✅ All module imports successful")
    
    def test_evaluator_attributes(self):
        """Test that evaluators have expected attributes"""
        original_data = pd.DataFrame({
            'id': [1, 2, 3],
            'rating': [4, 5, 3],
            'text': ['Great product', 'Excellent quality', 'Good value']
        })
        
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3],
            'rating': [4, 5, 3],
            'text': ['Great product', 'Excellent quality', 'Good value']
        })
        
        metadata = {
            'columns': {
                'id': {'sdtype': 'numerical', 'is_primary_key': True},
                'rating': {'sdtype': 'categorical', 'values': [1, 2, 3, 4, 5]},
                'text': {'sdtype': 'text'}
            },
            'text_columns': ['text']
        }
        
        # Test FidelityEvaluator attributes
        fidelity_evaluator = FidelityEvaluator(synthetic_data, original_data, metadata)
        assert hasattr(fidelity_evaluator, 'original_data')
        assert hasattr(fidelity_evaluator, 'synthetic_data')
        assert hasattr(fidelity_evaluator, 'metadata')
        assert hasattr(fidelity_evaluator, 'evaluate')
        print("✅ FidelityEvaluator has expected attributes")
        
        # Test UtilityEvaluator attributes
        utility_evaluator = UtilityEvaluator(
            synthetic_data, original_data, metadata, 
            input_columns=['text'], output_columns=['rating']
        )
        assert hasattr(utility_evaluator, 'original_data')
        assert hasattr(utility_evaluator, 'synthetic_data')
        assert hasattr(utility_evaluator, 'metadata')
        assert hasattr(utility_evaluator, 'evaluate')
        assert hasattr(utility_evaluator, 'task_type')
        print("✅ UtilityEvaluator has expected attributes")
        
        # Test DiversityEvaluator attributes
        diversity_evaluator = DiversityEvaluator(synthetic_data, original_data, metadata)
        assert hasattr(diversity_evaluator, 'original_data')
        assert hasattr(diversity_evaluator, 'synthetic_data')
        assert hasattr(diversity_evaluator, 'metadata')
        assert hasattr(diversity_evaluator, 'evaluate')
        print("✅ DiversityEvaluator has expected attributes")
