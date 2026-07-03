import pytest
import pandas as pd
from evaluation.column_detector import auto_detect_columns
from tests.fixtures.test_data import get_sample_multimodal_df


def test_auto_detect_text_columns():
    """Should detect long high-cardinality text columns."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'short_text': ['abc', 'def', 'ghi'],  # Too short
        'description': [
            'This is a long description with many words',
            'Another lengthy text containing detailed information',
            'Yet another long descriptive text passage'
        ],
        'price': [100, 200, 300]
    })

    result = auto_detect_columns(df)

    assert 'description' in result['text']
    assert 'short_text' not in result['text']
    assert 'id' not in result['text']


def test_auto_detect_tabular_numeric():
    """Should detect numeric columns as tabular."""
    df = pd.DataFrame({
        'price': [100, 200, 300],
        'rating': [4.5, 3.2, 4.8],
        'count': [10, 20, 30]
    })

    result = auto_detect_columns(df)

    assert 'price' in result['tabular']
    assert 'rating' in result['tabular']
    assert 'count' in result['tabular']


def test_auto_detect_tabular_categorical():
    """Should detect low-cardinality categoricals as tabular."""
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B'],
        'status': ['active', 'inactive', 'active', 'active', 'inactive']
    })

    result = auto_detect_columns(df)

    assert 'category' in result['tabular']
    assert 'status' in result['tabular']


def test_auto_detect_skips_id_columns():
    """Should skip ID-like columns."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'index': [0, 1, 2],
        'key': ['a', 'b', 'c'],
        'price': [100, 200, 300]
    })

    result = auto_detect_columns(df)

    assert 'id' not in result['text']
    assert 'id' not in result['tabular']
    assert 'index' not in result['text']
    assert 'key' not in result['text']


def test_auto_detect_multimodal_dataset():
    """Should correctly detect text and tabular in multimodal dataset."""
    df = get_sample_multimodal_df()

    result = auto_detect_columns(df)

    assert 'description' in result['text']
    assert 'price' in result['tabular']
    assert 'accommodates' in result['tabular']
    assert 'property_type' in result['tabular']
