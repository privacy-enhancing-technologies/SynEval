# Privacy Evaluation Metrics

## Overview

Privacy evaluation analyzes the privacy protection level of synthetic data by assessing various privacy risks and vulnerabilities. The `PrivacyEvaluator` class in `evaluation/privacy.py` provides comprehensive privacy metrics for both structured and text data.

## Available Metrics

The privacy evaluator supports the following metrics:

- **exact_matches**: Exact row match analysis
- **membership_inference**: Membership inference attack evaluation
- **tabular_privacy**: Structured privacy metrics (IMS, DCR, NNDR)
- **text_privacy**: Text-specific privacy analysis (NER, nominal mentions, stylistic outliers)
- **anonymeter**: Anonymeter re-identification risk evaluation

## Metric Details

### 1. Exact Match Analysis

**Metric Type**: `exact_matches`  
**Data Type**: Both structured and text data  
**Dependencies**: Requires `pandas`

#### Exact Match Percentage

**Description**: Percentage of synthetic rows that exactly match original rows.

**Algorithm**:
- Row-by-row comparison using pandas equality operations
- Checks if any synthetic row exactly matches any original row

**Score Calculation**:
```python
matches = 0
for syn_row in synthetic_data:
    if any(original_data == syn_row):
        matches += 1
exact_match_percentage = (matches / total_synthetic_rows) × 100
```

**Risk Level**:
- **High Risk**: >5% exact matches
- **Low Risk**: ≤5% exact matches

**Interpretation**:
- **<1%**: Excellent - minimal exact matches
- **1-5%**: Acceptable - some matches but within threshold
- **>5%**: Concerning - high risk of data leakage

**Note**: Always compare with original duplication rate and consider approximate matching before concluding leakage.

**Output Format**:
```json
{
  "exact_matches": {
    "exact_match_percentage": 2.3,
    "risk_level": "low"
  }
}
```

### 2. Membership Inference Attack (MIA)

**Metric Type**: `membership_inference`  
**Data Type**: Both structured and text data  
**Dependencies**: Requires `sklearn`, `pandas`, `numpy`

#### MIA AUC Score

**Description**: Area under ROC curve for membership inference classifier.

**Algorithm**:
1. Combine synthetic and original data with labels (1=synthetic, 0=original)
2. Extract features:
   - **Text columns**: TF-IDF vectorization (max_features=1000)
   - **Non-text columns**: One-hot encoding for categorical variables
3. Train RandomForest classifier to distinguish between datasets
4. Calculate ROC-AUC score

**Score Calculation**:
```python
# Create labels
y = [1] * len(synthetic) + [0] * len(original)

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
```

**Risk Level**:
- **High Risk**: AUC ≥ 0.70 (configurable threshold)
- **Low Risk**: AUC < 0.70

**Interpretation**:
- **AUC = 0.50**: Random guessing - excellent privacy
- **AUC 0.50-0.60**: Good privacy protection
- **AUC 0.60-0.70**: Moderate privacy concerns
- **AUC ≥ 0.70**: High risk - synthetic data is easily distinguishable

**Note**: AUC depends on attacker strength, class imbalance, and evaluation protocol. Always check ROC curves, baselines, and confidence intervals.

#### Synthetic/Original Confidence

**Description**: Average confidence of classifier predictions.

**Algorithm**:
- Mean of classifier prediction probabilities for each dataset type

**Score Calculation**:
```python
synthetic_confidence = mean(clf.predict_proba(synthetic_features)[:, 1])
original_confidence = mean(clf.predict_proba(original_features)[:, 1])
```

**Interpretation**:
- Large difference in confidence indicates easy distinguishability
- Similar confidence values indicate better privacy protection

**Output Format**:
```json
{
  "membership_inference": {
    "distinguishability_auc": 0.65,
    "synthetic_confidence": 0.72,
    "original_confidence": 0.28,
    "fidelity_score": 0.35,
    "interpretation": "Values near 1.0 indicate synthetic data is easily separable from the original distribution (poor fidelity). Values near 0.5 indicate high similarity."
  }
}
```

### 3. Tabular Privacy Metrics

**Metric Type**: `tabular_privacy`  
**Data Type**: Structured data only  
**Dependencies**: Requires `sklearn`, `numpy`, `pandas`

#### Identical Match Share (IMS)

**Description**: Percentage of synthetic rows that are identical to training rows.

**Algorithm**:
1. Convert all values to strings for consistent comparison
2. Create sets of row tuples for both datasets
3. Calculate intersection (identical rows)
4. Compare with train-test IMS baseline

**Score Calculation**:
```python
syn_set = set(tuple(str(val) for val in row) for row in synthetic_data)
train_set = set(tuple(str(val) for val in row) for row in train_data)
ims_syn_train = len(syn_set & train_set) / len(syn_set) if len(syn_set) > 0 else 0

# Baseline: train-test IMS
train_test_ims = calculate_train_test_ims()
passed = ims_syn_train <= train_test_ims
```

**Interpretation**:
- **IMS ≤ baseline**: Pass - acceptable level of exact matches
- **IMS > baseline**: Fail - too many exact matches
- **IMS < 0.01 (1%)**: Excellent - minimal identical matches
- **IMS 0.01-0.05**: Good - within acceptable range
- **IMS > 0.05**: Concerning - high identical match rate

**Output Format**:
```json
{
  "structured_privacy_metrics": {
    "IMS": {
      "ims_syn_train": 0.02,
      "train_test_ims": 0.015,
      "passed": false,
      "method": "Identical Match Share (IMS)",
      "desc": "Synthetic-Train IMS: 0.0200, Train-Test IMS: 0.0150, Passed: No"
    }
  }
}
```

#### Distance to Closest Records (DCR)

**Description**: Distance distribution from synthetic data points to their nearest neighbors in training set.

**Algorithm**:
1. Prepare features by encoding categorical/boolean data
2. Fit NearestNeighbors on training data
3. Calculate distances from synthetic to training data
4. Calculate distances within training data (baseline)
5. Compare 5th percentile distances

**Score Calculation**:
```python
# Synthetic to training distances
dists_syn_train, _ = nn.kneighbors(X_syn, n_neighbors=1)
syn_train_5pct = percentile(dists_syn_train, 5)

# Training to training distances (baseline)
dists_train_train, _ = nn.kneighbors(X_train, n_neighbors=2)
train_train_5pct = percentile(dists_train_train[:, 1], 5)  # Skip itself

passed = syn_train_5pct >= train_train_5pct
```

**Interpretation**:
- **syn_train_5pct ≥ train_train_5pct**: Pass - synthetic data is at least as far as training baseline
- **syn_train_5pct < train_train_5pct**: Fail - synthetic data is too close to training data
- Higher DCR values indicate better privacy protection

**Output Format**:
```json
{
  "DCR": {
    "syn_train_5pct": 12.5,
    "train_train_5pct": 10.2,
    "passed": true,
    "method": "Distance to Closest Records (DCR)",
    "desc": "Synthetic-Train 5%: 12.5000, Train-Train 5%: 10.2000, Passed: Yes"
  }
}
```

#### Nearest Neighbor Distance Ratio (NNDR)

**Description**: Distance ratio distribution comparing nearest and second-nearest neighbors.

**Algorithm**:
1. Fit NearestNeighbors with n_neighbors=3
2. Calculate distance ratios for synthetic data
3. Calculate distance ratios for training data (baseline)
4. Compare 5th percentile ratios

**Score Calculation**:
```python
# Synthetic data ratios
dists_syn, _ = nn.kneighbors(X_syn, n_neighbors=2)
ratio_syn = dists_syn[:, 1] / (dists_syn[:, 0] + 1e-8)
syn_train_5pct = percentile(ratio_syn, 5)

# Training data ratios (baseline)
dists_train, _ = nn.kneighbors(X_train, n_neighbors=3)
ratio_train = dists_train[:, 2] / (dists_train[:, 1] + 1e-8)
train_train_5pct = percentile(ratio_train, 5)

passed = syn_train_5pct >= train_train_5pct
```

**Interpretation**:
- **NNDR ≥ baseline**: Pass - good privacy protection
- **NNDR < baseline**: Fail - potential memorization risk
- **NNDR > 0.9**: Excellent - high distance ratios
- **NNDR 0.2-0.9**: Good - acceptable ratios
- **NNDR < 0.2**: Low - concerning ratios

**Output Format**:
```json
{
  "NNDR": {
    "syn_train_5pct": 0.85,
    "train_train_5pct": 0.78,
    "passed": true,
    "method": "Nearest Neighbor Distance Ratio (NNDR)",
    "desc": "Synthetic-Train 5%: 0.8500, Train-Train 5%: 0.7800, Passed: Yes"
  }
}
```

### 4. Text Privacy Metrics

**Metric Type**: `text_privacy`  
**Data Type**: Text columns only  
**Dependencies**: Requires `flair`, `torch`, `spacy`, `gensim`, `sklearn`

#### Named Entity Recognition (NER)

**Description**: Analyzes named entities (persons, organizations, locations) in text data.

**Algorithm**:
1. Load Flair NER model (`flair/ner-english-large`)
2. Process texts in batches (GPU-accelerated when available)
3. Extract entities with filtering:
   - Skip entities that are too short (<2 chars) or too long (>50 chars)
   - Skip numeric-only entities
   - Skip URLs and email addresses
   - Skip incomplete words/phrases
4. Calculate statistics for both datasets

**Entity Types**:
- **PER**: Person names
- **ORG**: Organizations
- **LOC**: Locations
- **MISC**: Miscellaneous entities

**Metrics**:
- **Total Entities**: Count of all detected entities
- **Entity Density**: Entities per token
- **Entity Overlap**: Percentage of original entities found in synthetic data

**Score Calculation**:
```python
entity_density = total_entities / total_tokens
overlap_percentage = (common_entities / original_entities) × 100
```

**Risk Level**:
- **High Risk**: Entity density > 0.1 or overlap > 50%
- **Low Risk**: Entity density ≤ 0.1 and overlap ≤ 50%

**Interpretation**:
- **Entity density < 0.05**: Low risk
- **Entity density 0.05-0.1**: Moderate risk
- **Entity density > 0.1**: High risk
- **Overlap < 30%**: Good privacy protection
- **Overlap 30-50%**: Moderate privacy concerns
- **Overlap > 50%**: High privacy risk

**Output Format**:
```json
{
  "named_entities": {
    "synthetic": {
      "total_entities": 1250,
      "avg_entity_density": 0.08,
      "risk_level": "low"
    },
    "original": {
      "total_entities": 1500,
      "avg_entity_density": 0.09,
      "risk_level": "low"
    },
    "overlap": {
      "overlap_percentage": 35.2,
      "risk_level": "low"
    }
  }
}
```

#### Nominal Mentions Analysis

**Description**: Analyzes person/role/relationship mentions in text using spaCy.

**Algorithm**:
1. Load spaCy model (`en_core_web_sm`)
2. Process texts in batches
3. Extract nouns and proper nouns that represent:
   - People (family terms, names)
   - Roles (teacher, doctor, customer, etc.)
   - Relationships (friend, colleague, neighbor, etc.)
4. Filter out common words and stopwords
5. Calculate statistics

**Detection Criteria**:
- Proper nouns (PROPN)
- Nouns in subject position
- Nouns matching role/relationship patterns

**Metrics**:
- **Total Nominals**: Count of detected nominal mentions
- **Nominal Density**: Nominals per token
- **Nominal Overlap**: Percentage of original nominals found in synthetic data

**Score Calculation**:
```python
nominal_density = total_nominals / total_tokens
overlap_percentage = (common_nominals / original_nominals) × 100
```

**Risk Level**:
- **High Risk**: Nominal density > 0.15 or overlap > 50%
- **Low Risk**: Nominal density ≤ 0.15 and overlap ≤ 50%

**Interpretation**:
- **Nominal density < 0.10**: Low risk
- **Nominal density 0.10-0.15**: Moderate risk
- **Nominal density > 0.15**: High risk
- **Overlap < 40%**: Good privacy protection
- **Overlap > 50%**: High privacy risk

**Output Format**:
```json
{
  "nominal_mentions": {
    "synthetic": {
      "total_nominals": 850,
      "avg_nominal_density": 0.12,
      "risk_level": "low"
    },
    "original": {
      "total_nominals": 920,
      "avg_nominal_density": 0.13,
      "risk_level": "low"
    },
    "overlap": {
      "overlap_percentage": 42.5,
      "risk_level": "low"
    }
  }
}
```

#### Stylistic Outliers Analysis

**Description**: Identifies stylistically unique texts that may be memorized.

**Algorithm**:
1. Generate Word2Vec embeddings for all texts
2. Calculate cosine distances between all text pairs
3. Identify texts with average distance > 2 standard deviations from mean
4. Compare outlier patterns between datasets

**Score Calculation**:
```python
# Generate embeddings
embeddings = train_word2vec(texts)

# Calculate distances
distances = cosine_distances(embeddings)
mean_dists = mean(distances, axis=1)
global_mean = mean(mean_dists)
global_std = std(mean_dists)

# Find outliers
outlier_scores = (mean_dists - global_mean) / global_std
outliers = texts[outlier_scores > 2.0]
outlier_percentage = len(outliers) / total_texts × 100
```

**Risk Level**:
- **High Risk**: Outlier percentage > 10% or significant difference from original
- **Low Risk**: Outlier percentage ≤ 10% and similar to original

**Interpretation**:
- **Outlier percentage < 5%**: Low risk
- **Outlier percentage 5-10%**: Moderate risk
- **Outlier percentage > 10%**: High risk
- Large differences from original may indicate memorization

**Output Format**:
```json
{
  "stylistic_outliers": {
    "synthetic": {
      "num_outliers": 45,
      "outlier_percentage": 4.5,
      "risk_level": "low"
    },
    "original": {
      "num_outliers": 38,
      "outlier_percentage": 3.8,
      "risk_level": "low"
    },
    "comparison": {
      "outlier_ratio": 1.18,
      "risk_level": "low"
    }
  }
}
```

### 5. Anonymeter Re-identification Risks

**Metric Type**: `anonymeter`  
**Data Type**: Structured data only  
**Dependencies**: Requires `anonymeter` package

#### Singling Out Attack (Univariate)

**Description**: Risk of identifying unique individuals using single attributes.

**Algorithm**:
1. Find unique combinations of single attributes in synthetic data
2. Check if these combinations exist in original data
3. Calculate attack success rate vs. baseline random guessing

**Score Calculation**:
```python
attack_rate = successful_attacks / total_attacks
baseline_rate = expected_random_success_rate
risk = (attack_rate - baseline_rate) / (1 - baseline_rate) if baseline_rate < 1 else 0
```

**Risk Level**:
- **High Risk**: risk > 0.5
- **Low Risk**: risk ≤ 0.5

**Interpretation**:
- **Risk < 0.3**: Good privacy protection
- **Risk 0.3-0.5**: Moderate privacy concerns
- **Risk > 0.5**: High privacy risk

**Output Format**:
```json
{
  "singling_out_univariate": {
    "attack_rate": 0.45,
    "baseline_rate": 0.20,
    "control_rate": 0.18,
    "risk": 0.31,
    "error": 0.02
  }
}
```

#### Singling Out Attack (Multivariate)

**Description**: Risk of identifying unique individuals using attribute combinations (up to 4 attributes).

**Algorithm**: Same as univariate but using combinations of multiple attributes

**Score Calculation**: Same as univariate

**Interpretation**: Same as univariate, but typically higher risk due to more identifying power

**Output Format**: Same format as univariate

#### Linkability Attack

**Description**: Risk of linking synthetic records to original records.

**Algorithm**:
1. Use auxiliary columns to find similar records
2. Attempt to link synthetic records to original records
3. Calculate success rate vs. baseline

**Score Calculation**: Same as singling out attacks

**Interpretation**:
- **Risk < 0.3**: Good privacy protection
- **Risk 0.3-0.7**: Moderate privacy concerns
- **Risk > 0.7**: High privacy risk

**Output Format**:
```json
{
  "linkability": {
    "attack_rate": 0.52,
    "baseline_rate": 0.25,
    "control_rate": 0.23,
    "risk": 0.36,
    "error": 0.03
  }
}
```

#### Inference Attack

**Description**: Risk of inferring sensitive attributes from other attributes.

**Algorithm**:
1. For each column as "secret", use other columns as auxiliary
2. Train model to predict secret from auxiliary columns
3. Test on synthetic data and calculate inference success
4. Repeat for all columns

**Score Calculation**: Same as other attacks, calculated per column

**Interpretation**:
- **Risk < 0.3**: Good privacy protection for this column
- **Risk 0.3-0.7**: Moderate privacy concerns
- **Risk > 0.7**: High privacy risk for this column

**Output Format**:
```json
{
  "inference": [
    {
      "secret_column": "salary",
      "attack_rate": 0.38,
      "baseline_rate": 0.15,
      "control_rate": 0.14,
      "risk": 0.27,
      "error": 0.02
    }
  ]
}
```

#### Overall Risk

**Description**: Maximum risk score across all attack types.

**Algorithm**:
```python
overall_risk = max(
    singling_out_univariate_risk,
    singling_out_multivariate_risk,
    linkability_risk,
    max(inference_risks)
)
```

**Risk Level**:
- **High Risk**: overall_risk > 0.5
- **Medium Risk**: 0.3 ≤ overall_risk ≤ 0.5
- **Low Risk**: overall_risk < 0.3

**Interpretation**:
- **<0.3**: Good privacy protection across all attacks
- **0.3-0.7**: Moderate privacy concerns
- **>0.7**: Significant privacy vulnerabilities

**Output Format**:
```json
{
  "overall_risk": {
    "risk_score": 0.42,
    "risk_level": "medium",
    "explanation": "Overall risk is determined by the highest risk score among all attacks."
  }
}
```

## Usage Example

```python
from evaluation import PrivacyEvaluator
import pandas as pd

# Load data
synthetic_data = pd.read_csv("synthetic_data.csv")
original_data = pd.read_csv("original_data.csv")
metadata = {...}  # Your metadata dictionary

# Initialize evaluator
evaluator = PrivacyEvaluator(
    synthetic_data=synthetic_data,
    original_data=original_data,
    metadata=metadata,
    selected_metrics=[
        "exact_matches",
        "membership_inference",
        "tabular_privacy",
        "text_privacy",
        "anonymeter"
    ],
    device="auto"  # or "cpu" or "cuda"
)

# Run evaluation
results = evaluator.evaluate()

# Access results
exact_match_pct = results["exact_matches"]["exact_match_percentage"]
mia_auc = results["membership_inference"]["distinguishability_auc"]
overall_risk = results["anonymeter"]["overall_risk"]["risk_score"]
```

## Performance Considerations

- **GPU Acceleration**: 
  - NER processing benefits significantly from GPU
  - Batch processing with larger batch sizes on GPU
- **Caching**: 
  - NER, nominal mentions, and stylistic outliers are cached
  - Caching uses dataset fingerprinting for validity
- **Memory Usage**: 
  - Large text datasets are processed in batches
  - Anonymeter can be memory-intensive for large datasets
- **Processing Time**: 
  - NER analysis can be time-consuming (model loading + processing)
  - Anonymeter evaluations can take significant time
  - Consider using GPU for faster processing

## Best Practices

1. **Risk Thresholds**: Use configurable thresholds based on your domain and compliance requirements
2. **Baseline Comparison**: Always compare with baseline rates and original data statistics
3. **Multiple Metrics**: Use multiple privacy metrics for comprehensive assessment
4. **Context Matters**: Consider domain-specific privacy requirements
5. **Regular Evaluation**: Re-evaluate privacy as data or models change

## Dependencies

- **Required**: `pandas`, `numpy`, `sklearn`
- **For text privacy**: `flair`, `torch`, `spacy`, `gensim`
- **For Anonymeter**: `anonymeter` package
- **Optional for GPU**: `torch` with CUDA support

## Important Notes

### Risk Thresholds

SynEval includes **default thresholds** as convenient alert levels, but acceptable risk varies across:
- **Domains**: Different domains have different privacy requirements
- **Attack Setups**: Attacker strength and evaluation protocol affect results
- **Compliance Standards**: Regulatory requirements may differ

Thresholds are **configurable**, and reports include:
- **Baseline comparisons**: Compare against random/control baselines
- **Uncertainty estimates**: Confidence intervals when available
- **Context**: Domain-specific interpretation guidance

### Membership Inference (MIA)

- Default alert at **AUC ≥ 0.70** (configurable)
- AUC depends heavily on attacker strength, class imbalance, and evaluation protocol
- **0.70 is not a universal cut-off**
- Always check:
  - ROC curves
  - Precision@k
  - Permutation/random baselines
  - Confidence intervals

### Exact Match Analysis

- Default alert at **>5%** (configurable)
- This is a **conservative heuristic**
- Some datasets naturally duplicate (short reviews, limited vocab)
- Always compare with:
  - Original duplication rate
  - Approximate matching (edit distance, token Jaccard)
  - Anonymeter risks (singling-out, linkability, inference)

