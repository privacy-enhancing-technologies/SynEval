# SynEval Test Suite

This directory contains comprehensive tests for the SynEval evaluation framework.

## Test Structure

### Test Files

- `conftest.py` - Pytest configuration and shared fixtures
- `test_correct.py` - Basic functionality tests for all modules
- `test_basic.py` - Additional basic tests (without privacy module)
- `test_simple.py` - Simple tests (if needed)

### Test Categories

#### 1. **FidelityEvaluator Tests**
- Initialization and configuration
- Dataset fingerprint computation
- Diagnostic evaluation (data validity, structure)
- Quality evaluation (column shapes, trends)
- Numerical statistics analysis
- Text analysis (length, keywords, sentiment)
- Caching functionality
- Error handling

#### 2. **UtilityEvaluator Tests**
- Initialization and configuration
- Task type detection (classification/regression)
- Feature preprocessing (text TF-IDF, categorical encoding)
- Model training and evaluation
- Performance metrics calculation
- Data splitting functionality
- Error handling

#### 3. **DiversityEvaluator Tests**
- Initialization and configuration
- Tabular diversity metrics (coverage, uniqueness)
- Numerical diversity analysis
- Categorical diversity analysis
- Text diversity (lexical, semantic, sentiment)
- Entropy metrics
- Error handling

#### 4. **PrivacyEvaluator Tests**
- Initialization and configuration
- Exact match analysis
- Membership inference attacks
- Named entity recognition
- Nominal mentions analysis
- Stylistic outliers analysis
- Anonymeter risk assessment
- Error handling

## Running Tests

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock pytest-xdist coverage
```

2. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('stopwords')"
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### Running Tests

#### Run All Tests
```bash
python -m pytest tests/ -v
```

#### Run Specific Test File
```bash
python -m pytest tests/test_correct.py -v
```

#### Run with Coverage
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

#### Run with Parallel Execution
```bash
python -m pytest tests/ -n auto
```

#### Run Only Fast Tests
```bash
python -m pytest tests/ -m "not slow"
```

### Using the Test Runner Script

```bash
python run_tests.py
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

- Test discovery patterns
- Coverage settings (minimum 70%)
- Warning filters
- Markers for test categorization

### GitHub Actions CI (`.github/workflows/ci.yml`)

- Multi-Python version testing (3.10, 3.11, 3.12)
- Dependency caching
- Automated test execution
- Coverage reporting
- Linting checks
- Package building

## Test Fixtures

### Sample Data Fixtures

- `sample_original_data` - Original dataset for testing
- `sample_synthetic_data` - Synthetic dataset for testing
- `sample_metadata` - Metadata configuration
- `sample_csv_files` - Temporary CSV and JSON files

### Mock Fixtures

- `mock_nltk_data` - Mock NLTK data download
- `mock_flair_models` - Mock Flair models
- `temp_dir` - Temporary directory for test files

## Test Coverage

The test suite aims for comprehensive coverage of:

- ✅ Module initialization and configuration
- ✅ Basic functionality verification
- ✅ Error handling and edge cases
- ✅ Method existence verification
- ✅ Data type validation
- ✅ Task type detection
- ✅ Import verification

## Continuous Integration

The project uses GitHub Actions for automated testing:

1. **Test Job**: Runs tests on multiple Python versions
2. **Lint Job**: Performs code quality checks
3. **Build Job**: Tests package building

## Adding New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Structure

```python
def test_feature_name(self):
    """Test description"""
    # Arrange
    # Act
    # Assert
    assert condition
```

### Mocking Heavy Dependencies

Use pytest-mock for mocking expensive operations:

```python
def test_with_mock(self, mocker):
    mock_function = mocker.patch('module.function')
    mock_function.return_value = expected_value
    # Test code
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **NLTK Data Missing**: Run NLTK data download commands
3. **spaCy Model Missing**: Download en_core_web_sm model
4. **Memory Issues**: Use smaller test datasets or mock heavy operations

### Debug Mode

Run tests with verbose output:
```bash
python -m pytest tests/ -v -s --tb=long
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve test coverage
4. Update this README if needed
