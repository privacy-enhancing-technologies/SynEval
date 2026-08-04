# Fidelity Evaluation Metrics

## Overview

Fidelity evaluation measures how well synthetic data preserves the statistical properties and patterns of the original data. The `FidelityEvaluator` class in `evaluation/fidelity.py` provides comprehensive metrics using both SDV (Synthetic Data Vault) framework and custom statistical analysis.

## Available Metrics

The fidelity evaluator supports the following metrics:

- **diagnostic**: SDV-based diagnostic evaluation
- **quality**: SDV-based quality evaluation
- **text**: Text-specific statistical analysis
- **numerical_statistics**: Comprehensive numerical column analysis

## Metric Details

### 1. Diagnostic Metrics (SDV-based)

**Metric Type**: `diagnostic`  
**Data Type**: Structured data only  
**Dependencies**: Requires `sdv` package

#### Data Validity

**Description**: Measures the percentage of valid data in the synthetic dataset.

**Algorithm**:
- SDV's diagnostic evaluation checks for:
  - Data type consistency
  - Missing value patterns
  - Constraint violations
  - Format compliance

**Score Calculation**:
- Percentage of rows that pass all validity checks divided by total rows
- Score range: 0.0 to 1.0

**Interpretation**:
- **0.9-1.0**: Excellent data validity
- **0.8-0.9**: Good data validity
- **0.7-0.8**: Fair data validity
- **<0.7**: Poor data validity

**Output Format**:
```json
{
  "diagnostic": {
    "Data Validity": 0.95,
    "Data Structure": 0.87,
    "Overall": {"score": 0.91}
  }
}
```

#### Data Structure

**Description**: Evaluates how well the synthetic data maintains the structural relationships of the original data.

**Algorithm**:
- SDV analyzes:
  - Primary key uniqueness
  - Foreign key relationships
  - Referential integrity
  - Structural constraints

**Score Calculation**:
- Weighted average of structural constraint compliance scores
- Score range: 0.0 to 1.0

**Interpretation**:
- Higher scores indicate better preservation of data relationships and constraints
- **0.9-1.0**: Excellent structural preservation
- **0.8-0.9**: Good structural preservation
- **<0.8**: May have structural integrity issues

#### Overall Diagnostic Score

**Description**: Combined diagnostic score indicating overall data quality.

**Algorithm**:
- Weighted average of Data Validity and Data Structure scores

**Score Calculation**:
```
Overall Score = (Data Validity × 0.6) + (Data Structure × 0.4)
```

**Interpretation**:
- Comprehensive measure of basic data quality and structural integrity
- **0.9-1.0**: Excellent overall quality
- **0.8-0.9**: Good overall quality
- **<0.8**: May have quality issues

### 2. Quality Metrics (SDV-based)

**Metric Type**: `quality`  
**Data Type**: Structured data only  
**Dependencies**: Requires `sdv` package

#### Column Shapes

**Description**: Measures how well the synthetic data preserves the distribution shapes of individual columns.

**Algorithm**:
- SDV uses statistical tests:
  - **Kolmogorov-Smirnov test** for continuous/numerical columns
  - **Chi-square test** for categorical columns
- Compares distributions between original and synthetic data

**Score Calculation**:
- Average of distribution similarity scores across all columns
- Normalized to 0-1 scale
- Score range: 0.0 to 1.0

**Interpretation**:
- Higher scores indicate better preservation of individual column distributions
- **0.9-1.0**: Excellent distribution preservation
- **0.8-0.9**: Good distribution preservation
- **<0.8**: Distribution may differ significantly

**Output Format**:
```json
{
  "quality": {
    "Column Shapes": 0.89,
    "Column Pair Trends": 0.82,
    "Overall": {"score": 0.86}
  }
}
```

#### Column Pair Trends

**Description**: Evaluates the preservation of relationships between column pairs.

**Algorithm**:
- SDV analyzes:
  - Correlation coefficients between column pairs
  - Mutual information between columns
  - Conditional distributions
  - Pairwise dependencies

**Score Calculation**:
- Average of pairwise relationship preservation scores across all column combinations
- Score range: 0.0 to 1.0

**Interpretation**:
- Higher scores indicate better preservation of inter-column relationships and correlations
- **0.9-1.0**: Excellent relationship preservation
- **0.8-0.9**: Good relationship preservation
- **<0.8**: Relationships may be distorted

#### Overall Quality Score

**Description**: Combined quality score for statistical fidelity.

**Algorithm**:
- Weighted average of Column Shapes and Column Pair Trends

**Score Calculation**:
```
Overall Quality = (Column Shapes × 0.7) + (Column Pair Trends × 0.3)
```

**Interpretation**:
- Comprehensive measure of statistical fidelity and relationship preservation
- **0.9-1.0**: Excellent statistical fidelity
- **0.8-0.9**: Good statistical fidelity
- **<0.8**: Statistical properties may differ

### 3. Text-Specific Metrics

**Metric Type**: `text`  
**Data Type**: Text columns only  
**Dependencies**: Requires `textblob` and `sklearn`

#### Length Statistics

**Description**: Compares text length distributions between original and synthetic data.

**Algorithm**:
- Character count analysis using string operations
- Word count analysis using string splitting
- Statistical comparison of length distributions

**Measures**:
- `original_mean`: Mean character/word count in original data
- `original_std`: Standard deviation in original data
- `synthetic_mean`: Mean character/word count in synthetic data
- `synthetic_std`: Standard deviation in synthetic data

**Score Calculation**:
- Direct statistical comparison of length distributions
- No single score; provides distribution statistics

**Interpretation**:
- Similar means and standard deviations indicate good text length preservation
- Large differences may indicate generation issues

**Output Format**:
```json
{
  "text": {
    "text_column": {
      "length_stats": {
        "original_mean": 245.3,
        "original_std": 89.2,
        "synthetic_mean": 238.7,
        "synthetic_std": 92.1
      },
      "word_count_stats": {
        "original_mean": 42.5,
        "original_std": 15.3,
        "synthetic_mean": 41.2,
        "synthetic_std": 16.1
      }
    }
  }
}
```

#### Keyword Analysis

**Description**: Compares the most important keywords (TF-IDF scores) between datasets.

**Algorithm**:
1. Fit TF-IDF vectorizer on original data
2. Transform both datasets using the same vectorizer
3. Calculate mean TF-IDF scores for each term
4. Rank terms by importance scores
5. Extract top 10 keywords for each dataset

**Score Calculation**:
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Mean TF-IDF scores per term
- Top keywords ranked by importance

**Interpretation**:
- Similar top keywords and scores indicate good content preservation
- Significant differences may indicate content drift

**Output Format**:
```json
{
  "keyword_analysis": {
    "original_top_keywords": {
      "product": 0.123,
      "quality": 0.098,
      "delivery": 0.087
    },
    "synthetic_top_keywords": {
      "product": 0.119,
      "quality": 0.095,
      "delivery": 0.082
    }
  }
}
```

#### Sentiment Analysis

**Description**: Compares sentiment distributions between datasets.

**Algorithm**:
- TextBlob sentiment analysis using polarity scores
- Polarity range: -1.0 (very negative) to +1.0 (very positive)

**Score Calculation**:
1. Calculate sentiment polarity for each text
2. Compute mean and standard deviation of sentiment scores
3. Categorize into:
   - **Negative**: polarity < -0.1
   - **Neutral**: -0.1 ≤ polarity ≤ 0.1
   - **Positive**: polarity > 0.1

**Interpretation**:
- Similar sentiment distributions indicate good emotional tone preservation
- Large differences may indicate bias or generation issues

**Output Format**:
```json
{
  "sentiment_analysis": {
    "original_mean": 0.15,
    "original_std": 0.42,
    "synthetic_mean": 0.13,
    "synthetic_std": 0.44,
    "original_sentiment_distribution": {
      "negative": 15.2,
      "neutral": 28.5,
      "positive": 56.3
    },
    "synthetic_sentiment_distribution": {
      "negative": 16.1,
      "neutral": 29.2,
      "positive": 54.7
    }
  }
}
```

### 4. Numerical Statistics Analysis

**Metric Type**: `numerical_statistics`  
**Data Type**: Numerical columns only  
**Dependencies**: Requires `numpy` and optionally `torch` for GPU acceleration

#### Basic Statistics Comparison

**Description**: Compares fundamental statistical measures between original and synthetic data.

**Measures**:
- **Mean**: Average value
- **Median**: Middle value
- **Standard Deviation**: Measure of spread
- **Min/Max**: Range boundaries
- **Quartiles (Q25, Q75)**: 25th and 75th percentiles
- **Skewness**: Measure of asymmetry
- **Kurtosis**: Measure of tail heaviness

**Algorithm**:
- Direct statistical calculation using pandas/numpy functions
- Computed separately for original and synthetic data

**Score Calculation**:
- Relative differences calculated as: `|synthetic - original| / |original|`
- Lower relative differences indicate better statistical preservation

**Interpretation**:
- **<0.05**: Excellent preservation
- **0.05-0.10**: Good preservation
- **0.10-0.20**: Fair preservation
- **>0.20**: Poor preservation

#### Range Coverage

**Description**: Measures how much of the original data range is covered by synthetic data.

**Algorithm**:
- Calculate overlap between original and synthetic value ranges
- Overlap = min(max_syn, max_orig) - max(min_syn, min_orig)

**Score Calculation**:
```
Range Coverage = Overlap range / Original range
```
- Score range: 0.0 to 1.0

**Interpretation**:
- **0.9-1.0**: Excellent range coverage
- **0.8-0.9**: Good range coverage
- **<0.8**: May miss important value ranges

#### Distribution Similarity

**Description**: Compares the shape and characteristics of data distributions.

##### KL Divergence

**Algorithm**:
- Kullback-Leibler divergence using histogram binning
- Creates histograms with up to 50 bins
- Normalizes histograms to probability distributions

**Score Calculation**:
```
KL Divergence = Σ p(x) × log(p(x)/q(x))
```
where:
- `p(x)` = original distribution
- `q(x)` = synthetic distribution

**Interpretation**:
- **0.0**: Identical distributions
- **<0.1**: Very similar distributions
- **0.1-0.5**: Similar distributions
- **>0.5**: Different distributions

##### Histogram Intersection Similarity

**Algorithm**:
- Calculates intersection of normalized histograms
- Uses GPU acceleration when available for faster computation

**Score Calculation**:
```
Histogram Similarity = Σ min(hist_orig[i], hist_syn[i])
```
- Score range: 0.0 to 1.0

**Interpretation**:
- **0.9-1.0**: Excellent distribution similarity
- **0.8-0.9**: Good distribution similarity
- **<0.8**: Distributions may differ significantly

#### Overall Fidelity Score

**Description**: Combined numerical fidelity metric.

**Algorithm**:
- Weighted average of multiple preservation metrics

**Score Calculation**:
```
Overall Fidelity = mean([
  1 - min(mean_diff, 1),           # Mean preservation
  1 - min(std_diff, 1),            # Standard deviation preservation
  1 - min(skewness_diff/2, 1),    # Skewness preservation
  range_coverage,                   # Range coverage
  histogram_similarity              # Distribution similarity
])
```

**Interpretation**:
- **0.9-1.0**: Excellent fidelity - synthetic data closely matches original data distribution
- **0.8-0.9**: Good fidelity - synthetic data preserves most statistical properties
- **0.7-0.8**: Fair fidelity - synthetic data preserves basic statistical properties
- **0.6-0.7**: Poor fidelity - synthetic data shows significant deviation from original
- **<0.6**: Very poor fidelity - synthetic data does not preserve original data characteristics

**Output Format**:
```json
{
  "numerical_statistics": {
    "column_name": {
      "original_statistics": {
        "mean": 45.2,
        "std": 12.5,
        "skewness": 0.3,
        "kurtosis": 2.1
      },
      "synthetic_statistics": {
        "mean": 44.8,
        "std": 12.3,
        "skewness": 0.28,
        "kurtosis": 2.15
      },
      "relative_differences": {
        "mean_diff": 0.008,
        "std_diff": 0.016,
        "skewness_diff": 0.02,
        "kurtosis_diff": 0.05
      },
      "range_coverage": 0.95,
      "distribution_similarity": {
        "kl_divergence": 0.05,
        "histogram_similarity": 0.92
      },
      "overall_fidelity_score": 0.91,
      "fidelity_interpretation": "Excellent fidelity - synthetic data closely matches original data distribution"
    }
  }
}
```

## Usage Example

```python
from evaluation import FidelityEvaluator
import pandas as pd

# Load data
synthetic_data = pd.read_csv("synthetic_data.csv")
original_data = pd.read_csv("original_data.csv")
metadata = {...}  # Your metadata dictionary

# Initialize evaluator
evaluator = FidelityEvaluator(
    synthetic_data=synthetic_data,
    original_data=original_data,
    metadata=metadata,
    selected_metrics=["diagnostic", "quality", "text", "numerical_statistics"],
    device="auto"  # or "cpu" or "cuda"
)

# Run evaluation
results = evaluator.evaluate()

# Access results
print(f"Diagnostic Score: {results['diagnostic']['Overall']['score']}")
print(f"Quality Score: {results['quality']['Overall']['score']}")
```

## Performance Considerations

- **GPU Acceleration**: Numerical statistics analysis benefits from GPU acceleration when available
- **Caching**: Results are cached using dataset fingerprinting for faster re-runs
- **Memory Usage**: Text analysis processes data in batches for large datasets

## Dependencies

- **Required**: `pandas`, `numpy`, `sklearn`
- **For SDV metrics**: `sdv` package
- **For text analysis**: `textblob`
- **Optional for GPU**: `torch` with CUDA support

