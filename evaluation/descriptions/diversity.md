# Diversity Evaluation Metrics

## Overview

Diversity evaluation assesses the variety and uniqueness of synthetic data across multiple dimensions. The `DiversityEvaluator` class in `evaluation/diversity.py` provides comprehensive metrics for both tabular (structured) and text data.

## Available Metrics

The diversity evaluator supports the following metrics:

- **tabular_diversity**: Coverage, uniqueness, and entropy metrics for structured data
- **text_diversity**: Lexical, semantic, and sentiment diversity metrics for text data

## Metric Details

### 1. Tabular Diversity Metrics

**Metric Type**: `tabular_diversity`  
**Data Type**: Structured data only (numerical, categorical, boolean, datetime)  
**Dependencies**: Requires `pandas`, `numpy`, `scipy`

#### Coverage Metrics

**Description**: Measures how well synthetic data covers the value space of original data.

##### Numerical Column Coverage

**Algorithm**:
1. Calculate value ranges for both original and synthetic data
2. Find overlap between ranges
3. Compute coverage percentage

**Score Calculation**:
```python
overlap = min(real_max, syn_max) - max(real_min, syn_min)
total_range = real_max - real_min
coverage = (overlap / total_range) × 100 if total_range != 0 else 0
```

**Interpretation**:
- **80-100%**: Excellent coverage
- **60-80%**: Good coverage
- **40-60%**: Fair coverage
- **<40%**: Poor coverage - may miss important value ranges

##### Categorical Column Coverage

**Algorithm**:
1. Get unique categories from both datasets
2. Find common categories (intersection)
3. Compute coverage percentage

**Score Calculation**:
```python
real_categories = set(original_data[col].unique())
syn_categories = set(synthetic_data[col].unique())
common_categories = real_categories & syn_categories
coverage = (len(common_categories) / len(real_categories)) × 100
```

**Interpretation**:
- **80-100%**: Excellent category coverage
- **60-80%**: Good category coverage
- **<60%**: May miss important categories

**Output Format**:
```json
{
  "coverage": {
    "column1": 85.5,
    "column2": 92.3,
    "column3": 78.1
  }
}
```

#### Uniqueness Metrics

**Description**: Measures duplicate rows and uniqueness of synthetic data.

##### Synthetic Duplicate Ratio

**Algorithm**:
- Use `pandas.drop_duplicates()` to identify unique rows
- Calculate percentage of duplicate rows

**Score Calculation**:
```python
total_rows = len(synthetic_data)
unique_rows = len(synthetic_data.drop_duplicates())
duplicate_ratio = (1 - unique_rows / total_rows) × 100
```

**Interpretation**:
- **<5%**: Excellent uniqueness
- **5-10%**: Good uniqueness
- **10-20%**: Fair uniqueness
- **>20%**: Poor uniqueness - high duplication

##### Original Duplicate Ratio

**Description**: Duplicate ratio in original data for comparison.

**Algorithm**: Same as synthetic duplicate ratio

**Interpretation**: Provides baseline for comparison

##### Relative Duplication

**Description**: How synthetic duplication compares to original duplication.

**Score Calculation**:
```python
relative_duplication = (syn_duplicate_ratio / orig_duplicate_ratio) × 100
```

**Interpretation**:
- **<100%**: Synthetic data has fewer duplicates than original (good)
- **100%**: Same duplication level as original
- **>100%**: Synthetic data has more duplicates than original (concerning)

**Output Format**:
```json
{
  "uniqueness": {
    "synthetic_duplicate_ratio": 3.2,
    "original_duplicate_ratio": 2.8,
    "relative_duplication": 114.3
  }
}
```

#### Numerical Metrics

**Description**: Enhanced diversity metrics for numerical columns.

##### Statistical Differences

**Measures**:
- **Mean Difference**: Relative difference in means
- **Std Difference**: Relative difference in standard deviations
- **Skewness Difference**: Absolute difference in skewness
- **Kurtosis Difference**: Absolute difference in kurtosis

**Score Calculation**:
```python
mean_diff = |syn_mean - real_mean| / |real_mean|
std_diff = |syn_std - real_std| / |real_std|
skewness_diff = |syn_skewness - real_skewness|
kurtosis_diff = |syn_kurtosis - real_kurtosis|
```

**Interpretation**:
- Lower differences indicate better preservation of statistical properties
- **<0.05**: Excellent preservation
- **0.05-0.10**: Good preservation
- **>0.10**: Significant differences

##### Range Coverage

**Description**: Percentage of original value range covered by synthetic data.

**Algorithm**: Same as basic coverage metric but with enhanced analysis

##### Quartile Coverage

**Description**: How well synthetic data covers the 25th, 50th, and 75th percentiles.

**Score Calculation**:
```python
real_quartiles = real_data.quantile([0.25, 0.5, 0.75])
syn_quartiles = syn_data.quantile([0.25, 0.5, 0.75])
quartile_diff = |real_quartile - syn_quartile| / |real_quartile|
quartile_coverage = (1 - quartile_diff) × 100
```

**Interpretation**:
- Higher coverage indicates better preservation of distribution shape
- **>90%**: Excellent quartile coverage
- **80-90%**: Good quartile coverage
- **<80%**: May not preserve distribution well

##### Distribution Similarity

**Description**: Measures similarity between distributions using KL divergence.

**Algorithm**:
1. Create histograms with up to 50 bins
2. Normalize histograms to probability distributions
3. Calculate KL divergence

**Score Calculation**:
```python
kl_divergence = Σ p(x) × log(p(x) / q(x))
similarity_score = 100 × exp(-kl_divergence)
```
where:
- `p(x)` = original distribution
- `q(x)` = synthetic distribution

**Interpretation**:
- **KL Divergence < 0.1**: Very similar distributions
- **KL Divergence 0.1-0.5**: Similar distributions
- **KL Divergence > 0.5**: Different distributions
- **Similarity Score > 90**: Excellent similarity
- **Similarity Score 80-90**: Good similarity
- **Similarity Score < 80**: Distributions differ

**Output Format**:
```json
{
  "numerical_metrics": {
    "column_name": {
      "statistical_differences": {
        "mean_diff": 0.02,
        "std_diff": 0.05,
        "skewness_diff": 0.1,
        "kurtosis_diff": 0.15
      },
      "range_coverage": 95.2,
      "quartile_coverage": {
        "q1": 92.5,
        "q2": 94.1,
        "q3": 91.8
      },
      "distribution_similarity": {
        "kl_divergence": 0.08,
        "similarity_score": 92.3
      },
      "real_statistics": {...},
      "synthetic_statistics": {...}
    }
  }
}
```

#### Categorical Metrics

**Description**: Enhanced diversity metrics for categorical columns.

##### Category Coverage

**Description**: Percentage of original categories present in synthetic data.

**Algorithm**: Same as basic coverage metric

##### Distribution Similarity

**Description**: How well synthetic data preserves category frequency distributions.

**Score Calculation**:
```python
distribution_diff = Σ |real_prob(cat) - syn_prob(cat)| for all categories
distribution_similarity = (1 - distribution_diff / 2) × 100
```

**Interpretation**:
- **>90%**: Excellent distribution preservation
- **80-90%**: Good distribution preservation
- **<80%**: Category frequencies may differ significantly

##### Entropy

**Description**: Information content comparison between original and synthetic data.

**Algorithm**: Shannon entropy calculation

**Score Calculation**:
```python
entropy = -Σ p(x) × log2(p(x))
```
where `p(x)` is the probability of category x

**Interpretation**:
- Higher entropy indicates more diversity
- Similar entropy values indicate similar diversity levels
- **Entropy difference < 0.1**: Excellent entropy preservation
- **Entropy difference 0.1-0.3**: Good entropy preservation
- **Entropy difference > 0.3**: Significant entropy differences

##### Top Categories Coverage

**Description**: How well synthetic data covers the most common categories.

**Algorithm**:
1. Identify top N (default: 5) most frequent categories in original data
2. Check how many appear in synthetic data
3. Calculate coverage percentage

**Score Calculation**:
```python
top_categories = set(real_counts.nlargest(5).index)
top_coverage = len(top_categories & syn_categories) / 5 × 100
```

**Interpretation**:
- **100%**: All top categories covered (excellent)
- **80-100%**: Most top categories covered (good)
- **<80%**: May miss important categories

##### Rare Categories Coverage

**Description**: How well synthetic data preserves rare categories (frequency < 1%).

**Algorithm**:
1. Identify categories with frequency < 1% in original data
2. Check how many appear in synthetic data
3. Calculate coverage percentage

**Score Calculation**:
```python
rare_categories = set(real_counts[real_counts < 0.01].index)
rare_coverage = len(rare_categories & syn_categories) / len(rare_categories) × 100
```

**Interpretation**:
- Important for preserving data diversity
- **>50%**: Good rare category preservation
- **<50%**: May lose important rare categories

**Output Format**:
```json
{
  "categorical_metrics": {
    "column_name": {
      "category_coverage": 88.5,
      "distribution_similarity": 85.2,
      "entropy": {
        "real": 3.45,
        "synthetic": 3.38,
        "difference": 0.07
      },
      "top_categories_coverage": 100.0,
      "rare_categories_coverage": 65.3,
      "category_counts": {
        "real": 25,
        "synthetic": 23,
        "common": 22
      }
    }
  }
}
```

#### Entropy Metrics

**Description**: Information content analysis for the entire dataset.

##### Column Entropy

**Description**: Shannon entropy for each column.

**Algorithm**:
- **Numerical columns**: Histogram-based entropy calculation
- **Categorical columns**: Direct entropy calculation from value counts

**Score Calculation**:
```python
# For numerical
hist, _ = np.histogram(data, bins=50, density=True)
probs = hist / (hist.sum() + 1e-10)
entropy = -Σ probs × log2(probs + 1e-10)

# For categorical
probs = data.value_counts(normalize=True)
entropy = -Σ probs × log2(probs)
```

##### Dataset Entropy

**Description**: Average entropy across all columns.

**Score Calculation**:
```python
dataset_entropy = mean(column_entropies)
```

##### Entropy Ratio

**Description**: Synthetic entropy relative to original entropy.

**Score Calculation**:
```python
entropy_ratio = synthetic_entropy / original_entropy
```

**Interpretation**:
- **0.9-1.1**: Excellent entropy preservation (ideal range)
- **0.8-0.9 or 1.1-1.2**: Good entropy preservation
- **0.6-0.8 or 1.2-1.4**: Fair entropy preservation
- **<0.6 or >1.4**: Poor entropy preservation

**Output Format**:
```json
{
  "entropy_metrics": {
    "column_entropy": {
      "column1": {
        "real": 4.2,
        "synthetic": 4.1,
        "entropy_ratio": 0.98,
        "entropy_difference": 0.1
      }
    },
    "dataset_entropy": {
      "real": 3.85,
      "synthetic": 3.78,
      "entropy_ratio": 0.98
    }
  }
}
```

### 2. Text Diversity Metrics

**Metric Type**: `text_diversity`  
**Data Type**: Text columns only  
**Dependencies**: Requires `nltk`, `gensim`, `networkx`, `scipy`, optionally `flair` and `torch` for sentiment analysis

#### Lexical Diversity

**Description**: Measures vocabulary diversity using n-gram analysis.

**Algorithm**:
1. Tokenize text and remove stopwords
2. Generate n-grams for n=1 to 5
3. Calculate statistics for each n-gram level

**Metrics**:
- **Total**: Total number of n-grams
- **Unique**: Number of unique n-grams
- **Unique Ratio**: Ratio of unique to total n-grams
- **Entropy**: Information content of n-gram distribution
- **Normalized Entropy**: Entropy normalized by maximum possible entropy

**Score Calculation**:
```python
# For each n-gram level
total = len(all_ngrams)
unique = len(set(all_ngrams))
unique_ratio = unique / total if total > 0 else 0

# Entropy calculation
counts = Counter(all_ngrams)
entropy = -Σ (count/total) × log2(count/total) for count in counts.values()
normalized_entropy = entropy / log2(unique) if unique > 1 else 0
```

**Interpretation**:
- **Higher unique ratio**: More diverse vocabulary
- **Higher entropy**: More uniform distribution (better diversity)
- **Unique ratio > 0.5**: Good lexical diversity
- **Unique ratio < 0.3**: Limited vocabulary diversity

**Output Format**:
```json
{
  "lexical_diversity": {
    "1-gram": {
      "total": 50000,
      "unique": 3500,
      "unique_ratio": 0.07,
      "entropy": 8.5,
      "normalized_entropy": 0.82
    },
    "2-gram": {
      "total": 48000,
      "unique": 12000,
      "unique_ratio": 0.25,
      "entropy": 10.2,
      "normalized_entropy": 0.75
    }
  }
}
```

#### Semantic Diversity

**Description**: Measures semantic variety using word embeddings and graph analysis.

**Algorithm**:
1. Train Word2Vec model on text corpus
2. Generate embeddings for each text (average of word embeddings)
3. Calculate cosine distances between all text pairs
4. Construct minimum spanning tree (MST) using Kruskal's algorithm
5. Calculate metrics from MST

**Metrics**:
- **Total MST Weight**: Sum of edge weights in MST
- **Average Edge Weight**: Average distance between semantically similar texts
- **Distinct Nodes**: Number of unique semantic representations
- **Distinct Ratio**: Ratio of distinct representations to total texts

**Score Calculation**:
```python
# Generate embeddings
embeddings = [mean(word_embeddings) for text in texts]

# Calculate distances
distances = cosine_distances(embeddings)

# Construct MST
mst = minimum_spanning_tree(distances)

# Calculate metrics
total_weight = sum(mst_edge_weights)
avg_weight = total_weight / num_edges
distinct_nodes = len(unique(round(embeddings, 6)))
distinct_ratio = distinct_nodes / num_texts
```

**GPU Acceleration**:
- When `device="cuda"`, distance calculations are performed on GPU
- Significantly faster for large datasets

**Interpretation**:
- **Higher total MST weight**: More semantic diversity
- **Higher distinct ratio**: More unique semantic representations
- **Distinct ratio > 0.8**: Excellent semantic diversity
- **Distinct ratio 0.6-0.8**: Good semantic diversity
- **Distinct ratio < 0.6**: Limited semantic diversity

**Output Format**:
```json
{
  "semantic_diversity": {
    "total_mst_weight": 1250.5,
    "average_edge_weight": 0.15,
    "distinct_nodes": 850,
    "distinct_ratio": 0.85,
    "sample_size": 1000,
    "device_used": "cuda"
  }
}
```

#### Sentiment Diversity

**Description**: Measures sentiment variety and alignment with ratings.

**Algorithm**:
1. Load Flair sentiment classifier
2. Classify sentiment for each text (Positive/Negative/Neutral)
3. Calculate sentiment distribution by rating level
4. Compare with ideal sentiment distribution

**Metrics**:
- **Sentiment by Rating**: Positive sentiment percentage for each rating level
- **Ideal Sentiment**: Expected sentiment distribution based on rating
- **Sentiment Alignment Score**: How well sentiment aligns with rating expectations

**Score Calculation**:
```python
# Ideal sentiment mapping
ideal = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}

# Actual sentiment by rating
actual = calculate_positive_sentiment_percentage_by_rating()

# Alignment score
diffs = [abs(actual[r] - ideal[r]) for r in ratings]
alignment_score = mean([1 - d for d in diffs])
```

**Interpretation**:
- **Alignment score > 0.7**: Excellent sentiment alignment
- **Alignment score 0.5-0.7**: Good sentiment alignment
- **Alignment score < 0.5**: Poor sentiment alignment

**Output Format**:
```json
{
  "sentiment_diversity": {
    "sentiment_by_rating": {
      "1": 0.05,
      "2": 0.22,
      "3": 0.48,
      "4": 0.73,
      "5": 0.95
    },
    "ideal_sentiment": {
      "1": 0.0,
      "2": 0.25,
      "3": 0.5,
      "4": 0.75,
      "5": 1.0
    },
    "sentiment_alignment_score": 0.92,
    "sample_size": 1000
  }
}
```

## Usage Example

```python
from evaluation import DiversityEvaluator
import pandas as pd

# Load data
synthetic_data = pd.read_csv("synthetic_data.csv")
original_data = pd.read_csv("original_data.csv")
metadata = {...}  # Your metadata dictionary

# Initialize evaluator
evaluator = DiversityEvaluator(
    synthetic_data=synthetic_data,
    original_data=original_data,
    metadata=metadata,
    selected_metrics=["tabular_diversity", "text_diversity"],
    device="auto"  # or "cpu" or "cuda"
)

# Run evaluation
results = evaluator.evaluate()

# Access results
coverage = results["tabular_diversity"]["coverage"]
uniqueness = results["tabular_diversity"]["uniqueness"]
text_diversity = results["text_diversity"]
```

## Performance Considerations

- **GPU Acceleration**: 
  - Semantic diversity calculations benefit significantly from GPU
  - Text processing can be accelerated with GPU
- **Caching**: 
  - Lexical, semantic, and sentiment diversity results are cached
  - Caching uses dataset fingerprinting for validity checking
- **Memory Usage**: 
  - Large text datasets are processed in batches
  - MST calculation can be memory-intensive for very large datasets
- **Processing Time**: 
  - Text diversity analysis can be time-consuming for large datasets
  - Consider using GPU for faster processing

## Best Practices

1. **Coverage Targets**: Aim for >80% coverage for important columns
2. **Uniqueness**: Keep duplicate ratio <5% for good diversity
3. **Entropy Ratio**: Target entropy ratio between 0.9-1.1
4. **Text Diversity**: Ensure distinct ratio >0.6 for semantic diversity
5. **Caching**: First run is slower; subsequent runs use cached results

## Dependencies

- **Required**: `pandas`, `numpy`, `nltk`, `gensim`, `networkx`, `scipy`
- **For sentiment analysis**: `flair`, `torch`
- **Optional for GPU**: `torch` with CUDA support

