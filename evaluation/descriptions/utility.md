# Utility Evaluation Metrics

## Overview

Utility evaluation assesses how useful synthetic data is for downstream machine learning tasks. The `UtilityEvaluator` class in `evaluation/utility.py` implements the **Train on Synthetic, Test on Real (TSTR)** methodology, which trains models on synthetic data and evaluates their performance on real data.

## Available Metrics

The utility evaluator supports the following metrics:

- **tstr_accuracy**: TSTR methodology for model performance comparison
- **correlation_analysis**: Cross-modality correlation metrics (optional)

## Metric Details

### 1. TSTR Accuracy (Train on Synthetic, Test on Real)

**Metric Type**: `tstr_accuracy`  
**Data Type**: Both structured and text data  
**Dependencies**: Requires `sklearn`, optionally `xgboost` for GPU acceleration

#### Methodology

**TSTR (Train on Synthetic, Test on Real)** is a standard approach for evaluating synthetic data utility:

1. **Train Model A** on real data, test on real hold-out data → baseline performance
2. **Train Model B** on synthetic data, test on the same real hold-out data → synthetic data performance
3. **Compare** the performance gap between Model A and Model B

A smaller performance gap indicates better utility of synthetic data.

#### Task Type Detection

The evaluator automatically detects the task type based on metadata:

- **Classification**: For categorical target variables
- **Regression**: For numerical target variables
- **Text Classification**: For text input with categorical output

**Algorithm**:
```python
if output_column_type == "categorical":
    task_type = "classification"
elif output_column_type == "text":
    task_type = "text_classification"
else:
    task_type = "regression"
```

#### Data Splitting

**Training Data**:
- Uses all synthetic data for training Model B
- Uses real data (or provided `real_train_data`) for training Model A

**Test Data**:
- Uses real hold-out data (20% split by default)
- Can be provided via `real_test_data` parameter
- Ensures fair comparison by testing both models on the same real data

**Algorithm**:
```python
# Split real data into train/test
train_data, test_data = train_test_split(
    real_data, 
    test_size=0.2, 
    random_state=42
)

# Train Model A on real training data
model_real.fit(X_train_real, y_train_real)

# Train Model B on synthetic data
model_syn.fit(X_train_syn, y_train_syn)

# Test both on same real test data
real_pred = model_real.predict(X_test)
syn_pred = model_syn.predict(X_test)
```

#### Feature Processing

##### Text Columns

**Algorithm**: TF-IDF vectorization

**Process**:
1. Initialize `TfidfVectorizer` with:
   - `max_features=1000`: Limit vocabulary size
   - `min_df=2`: Ignore rare terms
   - `max_df=0.95`: Ignore very common terms
   - `stop_words='english'`: Remove stop words
2. Fit vectorizer on training data
3. Transform all datasets (train_real, train_syn, test) using the same vectorizer

**GPU Acceleration**:
- When `device="cuda"`, text features are converted to GPU tensors for faster computation
- Automatically converts back to CPU for scikit-learn models

##### Categorical Columns

**Algorithm**: One-hot encoding

**Process**:
1. Use `pandas.get_dummies()` to convert categorical variables to binary columns
2. Fill missing values with 0
3. Combine with text features if both exist

##### Numerical Columns

**Algorithm**: Direct use with imputation

**Process**:
1. Fill NaN values with 0
2. Convert to numpy arrays or GPU tensors
3. Combine with other feature types

#### Model Selection

The evaluator automatically selects appropriate models based on task type and device:

**For Classification Tasks**:
- **GPU available**: XGBoost Classifier with `tree_method="gpu_hist"`
- **CPU only**: Random Forest Classifier (100 estimators)

**For Regression Tasks**:
- **GPU available**: XGBoost Regressor with `tree_method="gpu_hist"`
- **CPU only**: Random Forest Regressor (100 estimators)

**For Text Classification**:
- Pipeline with TF-IDF vectorization + classifier
- Same model selection logic as above

#### Classification Metrics

**Data Type**: Categorical target variables

##### Accuracy

**Description**: Overall prediction accuracy

**Algorithm**: `sklearn.metrics.accuracy_score`

**Score Calculation**:
```
Accuracy = (Correct predictions) / (Total predictions)
```

**Interpretation**:
- **0.9-1.0**: Excellent accuracy
- **0.8-0.9**: Good accuracy
- **0.7-0.8**: Fair accuracy
- **<0.7**: Poor accuracy

##### Precision

**Description**: Precision for each class

**Algorithm**: `sklearn.metrics.precision_score` with per-class calculation

**Score Calculation**:
```
Precision = True Positives / (True Positives + False Positives)
```

**Interpretation**:
- Higher precision indicates fewer false positive predictions
- Class-specific metric

##### Recall

**Description**: Recall for each class

**Algorithm**: `sklearn.metrics.recall_score` with per-class calculation

**Score Calculation**:
```
Recall = True Positives / (True Positives + False Negatives)
```

**Interpretation**:
- Higher recall indicates fewer false negative predictions
- Class-specific metric

##### F1-Score

**Description**: Harmonic mean of precision and recall

**Algorithm**: `sklearn.metrics.f1_score`

**Score Calculation**:
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation**:
- Balanced measure of precision and recall
- **0.9-1.0**: Excellent F1
- **0.8-0.9**: Good F1
- **<0.8**: May have precision/recall issues

##### Macro Average

**Description**: Average of per-class metrics (treats all classes equally)

**Algorithm**: Average of per-class precision, recall, and F1-scores

**Interpretation**:
- Useful when all classes are equally important
- Less affected by class imbalance

##### Weighted Average

**Description**: Average weighted by class frequency

**Algorithm**: Weighted average of per-class metrics based on class support

**Interpretation**:
- Accounts for class imbalance
- More representative of overall performance when classes are imbalanced

**Output Format**:
```json
{
  "tstr_accuracy": {
    "task_type": "classification",
    "input_columns": ["text"],
    "output_columns": ["rating"],
    "training_size": 10000,
    "test_size": 2000,
    "real_data_model": {
      "accuracy": 0.85,
      "macro avg": {
        "precision": 0.82,
        "recall": 0.80,
        "f1-score": 0.81
      },
      "weighted avg": {
        "precision": 0.84,
        "recall": 0.85,
        "f1-score": 0.84
      }
    },
    "synthetic_data_model": {
      "accuracy": 0.78,
      "macro avg": {
        "precision": 0.75,
        "recall": 0.73,
        "f1-score": 0.74
      },
      "weighted avg": {
        "precision": 0.77,
        "recall": 0.78,
        "f1-score": 0.77
      }
    }
  }
}
```

#### Regression Metrics

**Data Type**: Numerical target variables

##### R² Score (Coefficient of Determination)

**Description**: Measures how well the model explains the variance in the target variable

**Algorithm**: `sklearn.metrics.r2_score`

**Score Calculation**:
```
R² = 1 - (SS_res / SS_tot)
```
where:
- `SS_res` = sum of squared residuals
- `SS_tot` = total sum of squares

**Interpretation**:
- **1.0**: Perfect fit
- **0.9-1.0**: Excellent fit
- **0.8-0.9**: Good fit
- **0.7-0.8**: Fair fit
- **<0.7**: Poor fit
- **<0**: Model performs worse than a horizontal line

##### Mean Squared Error (MSE)

**Description**: Average squared prediction error

**Algorithm**: `sklearn.metrics.mean_squared_error`

**Score Calculation**:
```
MSE = Average of (predicted - actual)²
```

**Interpretation**:
- Lower MSE indicates better predictions
- Sensitive to outliers (squares the errors)
- Units are squared (e.g., if target is in dollars, MSE is in dollars²)

##### Root Mean Squared Error (RMSE)

**Description**: Square root of mean squared error

**Algorithm**: Square root of MSE

**Score Calculation**:
```
RMSE = √(MSE)
```

**Interpretation**:
- Lower RMSE indicates better predictions
- Same units as target variable (easier to interpret than MSE)
- Also sensitive to outliers

##### Mean Absolute Error (MAE)

**Description**: Average absolute prediction error

**Algorithm**: `sklearn.metrics.mean_absolute_error`

**Score Calculation**:
```
MAE = Average of |predicted - actual|
```

**Interpretation**:
- Lower MAE indicates better predictions
- Less sensitive to outliers than MSE/RMSE
- Same units as target variable

**Output Format**:
```json
{
  "tstr_accuracy": {
    "task_type": "regression",
    "input_columns": ["feature1", "feature2"],
    "output_columns": ["target"],
    "training_size": 10000,
    "test_size": 2000,
    "real_data_model": {
      "rmse": 12.5,
      "mae": 8.3,
      "r2": 0.89
    },
    "synthetic_data_model": {
      "rmse": 14.2,
      "mae": 9.1,
      "r2": 0.85
    }
  }
}
```

#### Performance Gap Analysis

**Description**: Measures the difference in performance between models trained on real vs. synthetic data.

**Calculation**:
```python
# For classification
performance_gap = abs(real_accuracy - syn_accuracy)

# For regression
performance_gap = abs(real_r2 - syn_r2)
```

**Interpretation**:
- **<0.1 (10%)**: Good utility - synthetic data performs similarly to real data
- **0.1-0.2 (10-20%)**: Moderate utility - acceptable performance gap
- **>0.2 (20%)**: Poor utility - significant performance degradation

**Quality Assessment**:
- **Good**: Performance gap < 0.1
- **Moderate**: Performance gap 0.1-0.2
- **Poor**: Performance gap > 0.2

### 2. Correlation Analysis (Optional)

**Metric Type**: `correlation_analysis`  
**Data Type**: Mixed data types  
**Dependencies**: Requires `nltk`, `scipy`, `sklearn`

#### Available Correlation Types

The evaluator supports several cross-modality correlation metrics:

##### Sentiment-Rating Correlation

**Description**: Measures correlation between text sentiment and rating values.

**Algorithm**:
1. Calculate sentiment polarity for each text using NLTK's SentimentIntensityAnalyzer
2. Compute Spearman correlation between sentiment scores and ratings

**Output**:
```json
{
  "sentiment_rating": {
    "sentiment_rating_corr": 0.75,
    "p_value": 0.001
  }
}
```

##### Keyword-Category Correlation

**Description**: Measures association between keywords and categories using chi-square test.

**Algorithm**:
1. Extract top N keywords using TF-IDF
2. Perform chi-square test between keywords and categories
3. Return top keywords with their chi-square scores

**Output**:
```json
{
  "keyword_category": [
    {
      "keyword": "product",
      "chi2": 45.2,
      "p_value": 0.001
    }
  ]
}
```

##### Numeric-Length Correlation

**Description**: Measures correlation between text length and numeric values.

**Algorithm**:
1. Calculate word count for each text
2. Compute Spearman correlation between lengths and numeric column

**Output**:
```json
{
  "numeric_length": {
    "length_numeric_corr": 0.32,
    "p_value": 0.05
  }
}
```

##### Semantic-Tabular Correlation

**Description**: Measures correlation between text embeddings and tabular features using Canonical Correlation Analysis (CCA).

**Algorithm**:
1. One-hot encode categorical columns
2. Use CCA to find linear combinations that maximize correlation
3. Compute correlation coefficient

**Note**: Requires pre-computed text embeddings

**Output**:
```json
{
  "semantic_tabular": {
    "semantic_tabular_corr": 0.68
  }
}
```

##### PII-Text Leakage

**Description**: Measures if PII (Personally Identifiable Information) appears in text columns.

**Algorithm**:
1. Check if PII column values appear in text columns
2. Calculate leakage rate as percentage of rows with leakage

**Output**:
```json
{
  "pii_text_leakage": {
    "pii_text_leakage_rate": 0.02
  }
}
```

## Usage Example

```python
from evaluation import UtilityEvaluator
import pandas as pd

# Load data
synthetic_data = pd.read_csv("synthetic_data.csv")
original_data = pd.read_csv("original_data.csv")
metadata = {...}  # Your metadata dictionary

# Initialize evaluator
evaluator = UtilityEvaluator(
    synthetic_data=synthetic_data,
    original_data=original_data,
    metadata=metadata,
    input_columns=["text"],  # Features for prediction
    output_columns=["rating"],  # Target to predict
    task_type="auto",  # Auto-detect from metadata
    selected_metrics=["tstr_accuracy"],
    device="auto"  # or "cpu" or "cuda"
)

# Run evaluation
results = evaluator.evaluate()

# Access results
real_acc = results["tstr_accuracy"]["real_data_model"]["accuracy"]
syn_acc = results["tstr_accuracy"]["synthetic_data_model"]["accuracy"]
performance_gap = abs(real_acc - syn_acc)

print(f"Real Data Model Accuracy: {real_acc:.3f}")
print(f"Synthetic Data Model Accuracy: {syn_acc:.3f}")
print(f"Performance Gap: {performance_gap:.3f}")
```

## Performance Considerations

- **GPU Acceleration**: 
  - Text feature processing benefits from GPU when available
  - XGBoost with GPU support significantly speeds up training
- **Caching**: Results are cached using dataset fingerprinting
- **Memory Usage**: Large text datasets are processed in batches
- **Training Time**: Depends on dataset size and model complexity

## Best Practices

1. **Ensure Sufficient Test Data**: At least 20% of real data should be held out for testing
2. **Class Balance**: For classification, ensure test set has representation from all classes
3. **Feature Engineering**: Consider domain-specific feature engineering for better results
4. **Model Selection**: The evaluator uses default models; consider custom models for specific use cases
5. **Multiple Runs**: Consider running multiple times with different random seeds for stability

## Dependencies

- **Required**: `pandas`, `numpy`, `sklearn`
- **For text processing**: `nltk`, `scipy`
- **Optional for GPU**: `xgboost`, `torch` with CUDA support
- **For correlation analysis**: `nltk.sentiment`, `scipy.stats`, `sklearn.cross_decomposition`

