#!/usr/bin/env python3

import hashlib
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
# 1. Sentiment-Rating Correlation
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def sentiment_rating_correlation(df, rating_col="rating", text_col="review"):
    sia = SentimentIntensityAnalyzer()
    sentiments = df[text_col].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    correlation, p_value = spearmanr(df[rating_col], sentiments)
    return {"sentiment_rating_corr": correlation, "p_value": p_value}


# 2. Keyword-Category Correlation


def keyword_category_correlation(df, category_col, text_col="review", top_n=50):
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english")
    X = vectorizer.fit_transform(df[text_col].astype(str))
    y = df[category_col]
    chi2scores, p_values = chi2(X, y)
    top_keywords = vectorizer.get_feature_names_out()
    return pd.DataFrame(
        {"keyword": top_keywords, "chi2": chi2scores, "p_value": p_values}
    )


# 3. Numeric-Length Correlation


def numeric_length_correlation(df, numeric_col, text_col="review"):
    lengths = df[text_col].apply(lambda x: len(str(x).split()))
    correlation, p_value = spearmanr(df[numeric_col], lengths)
    return {"length_numeric_corr": correlation, "p_value": p_value}


# 4. Semantic-Tabular Correlation


def semantic_tabular_correlation(df, categorical_cols, text_embeddings):
    encoder = OneHotEncoder().fit_transform(df[categorical_cols])
    cca = CCA(n_components=1)
    cca.fit(encoder.toarray(), text_embeddings)
    X_c, Y_c = cca.transform(encoder.toarray(), text_embeddings)
    correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]
    return {"semantic_tabular_corr": correlation}


# 5. PII-Text Leakage


def pii_text_leakage(df, pii_col, text_col="review"):
    leakage_rate = df.apply(
        lambda row: str(row[pii_col]) in str(row[text_col]), axis=1
    ).mean()
    return {"pii_text_leakage_rate": leakage_rate}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GPU acceleration setup
def setup_gpu_acceleration():
    """Setup GPU acceleration if available."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


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
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        "metadata_columns": list(metadata.get("columns", {}).keys()),
        "sample_hash": hashlib.sha256(data.head(1000).to_string().encode()).hexdigest()[
            :16
        ],  # First 1000 rows hash
        "total_hash": hashlib.sha256(data.to_string().encode()).hexdigest()[
            :16
        ],  # Full dataset hash
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
                with open(self.fingerprint_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Error loading fingerprints: {str(e)}")
        return {}

    def _save_fingerprints(self):
        """Save dataset fingerprints."""
        try:
            with open(self.fingerprint_file, "w") as f:
                json.dump(self.fingerprints, f, indent=2)
        except Exception as e:
            logging.warning(f"Error saving fingerprints: {str(e)}")

    def get_cache_path(self, cache_type: str, column: str = None) -> Path:
        """Get cache file path."""
        if column:
            return self.cache_dir / f"{cache_type}_{column}.pkl"
        return self.cache_dir / f"{cache_type}.pkl"

    def is_valid_cache(
        self, cache_type: str, dataset_fingerprint: str, column: str = None
    ) -> bool:
        """Check if cache is valid for current dataset."""
        cache_key = f"{cache_type}_{column}" if column else cache_type
        return self.fingerprints.get(cache_key) == dataset_fingerprint

    def load_cache(
        self, cache_type: str, dataset_fingerprint: str, column: str = None
    ) -> Optional[Dict]:
        """Load cache if valid."""
        if not self.is_valid_cache(cache_type, dataset_fingerprint, column):
            return None

        cache_path = self.get_cache_path(cache_type, column)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Error loading cache {cache_type}: {str(e)}")
        return None

    def save_cache(
        self, cache_type: str, dataset_fingerprint: str, data: Dict, column: str = None
    ):
        """Save cache with fingerprint."""
        cache_path = self.get_cache_path(cache_type, column)
        cache_key = f"{cache_type}_{column}" if column else cache_type

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            # Update fingerprint
            self.fingerprints[cache_key] = dataset_fingerprint
            self._save_fingerprints()

            logging.info(f"Saved cache for {cache_type}")
        except Exception as e:
            logging.warning(f"Error saving cache {cache_type}: {str(e)}")


class UtilityEvaluator:
    def __init__(
        self,
        synthetic_data: pd.DataFrame,
        original_data: pd.DataFrame,
        metadata: Dict,
        input_columns: List[str],
        output_columns: List[str],
        task_type: str = "auto",
        selected_metrics: List[str] = None,
        device: str = "auto",
        real_train_data: Optional[pd.DataFrame] = None,
        real_test_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the utility evaluator.

        Args:
            synthetic_data: Synthetic dataset
            original_data: Original dataset
            metadata: Dictionary containing column types and other metadata
            input_columns: List of columns to use as features
            output_columns: List of columns to predict
            task_type: Type of task ('classification', 'regression', or 'auto')
            selected_metrics: List of specific metrics to run
            device: Device to use for computation ('auto', 'cpu', 'cuda')
            real_train_data: Optional pre-split real training data
            real_test_data: Optional pre-split real test/hold-out data
        """
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.real_train_data = real_train_data
        self.real_test_data = real_test_data
        self.metadata = metadata
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.task_type = self._determine_task_type(task_type)

        # Initialize smart cache
        self.cache = SmartCache("./cache")

        # Compute dataset fingerprints
        self.original_fingerprint = compute_dataset_fingerprint(original_data, metadata)
        self.synthetic_fingerprint = compute_dataset_fingerprint(
            synthetic_data, metadata
        )

        # Device management
        if device == "auto":
            self.device = setup_gpu_acceleration()
        elif device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    logger.warning(
                        "CUDA requested but not available. Falling back to CPU."
                    )
                    self.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not available. Falling back to CPU.")
                self.device = "cpu"
        else:  # cpu
            self.device = "cpu"

        if self.device == "cuda":
            logger.info("GPU acceleration available - using CUDA")
        else:
            logger.info("GPU not available - using CPU")

        global DEVICE
        DEVICE = self.device

        # Available metrics for selection
        self.available_metrics = ["tstr_accuracy", "correlation_analysis"]

        # Use all metrics if none specified
        self.selected_metrics = (
            selected_metrics if selected_metrics else self.available_metrics
        )

    def _determine_task_type(self, task_type: str) -> str:
        """Determine the type of task based on output columns."""
        if task_type != "auto":
            return task_type

        for col in self.output_columns:
            if col in self.metadata["columns"]:
                col_type = self.metadata["columns"][col]["sdtype"]
                if col_type == "text":
                    return "text_classification"
                elif col_type == "categorical":
                    return "classification"
                else:
                    return "regression"
            else:
                logger.warning(f"Column {col} not found in metadata")
                return "classification"  # default to classification if column not found

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input and output data for model training."""
        X = data[self.input_columns]
        y = data[self.output_columns]

        # Handle text columns
        text_columns = [
            col
            for col in self.input_columns
            if col in self.metadata["columns"]
            and self.metadata["columns"][col]["sdtype"] == "text"
        ]

        if text_columns:
            # Fill NaN values with empty string for text columns
            text_data = data[text_columns].fillna("")
            vectorizer = TfidfVectorizer(max_features=1000)
            text_features = vectorizer.fit_transform(text_data[text_columns[0]])

            # Combine text features with other features
            other_columns = [
                col for col in self.input_columns if col not in text_columns
            ]
            if other_columns:
                # Fill NaN values with 0 for numerical columns
                other_features = data[other_columns].fillna(0).values
                X = np.hstack([text_features.toarray(), other_features])
            else:
                X = text_features.toarray()

        # Handle NaN values in output columns
        if isinstance(y, pd.DataFrame):
            y = y.fillna(
                y.mean() if self.task_type == "regression" else y.mode().iloc[0]
            )
        else:
            y = pd.Series(y).fillna(
                pd.Series(y).mean()
                if self.task_type == "regression"
                else pd.Series(y).mode().iloc[0]
            )

        return X, y

    def _get_model(self):
        """Get appropriate model based on task type with GPU acceleration support."""
        if self.device == "cuda":
            # Use GPU-accelerated models when available
            try:
                import xgboost as xgb

                if self.task_type == "text_classification":
                    return Pipeline(
                        [
                            ("tfidf", TfidfVectorizer(max_features=1000)),
                            (
                                "classifier",
                                xgb.XGBClassifier(
                                    n_estimators=100, tree_method="gpu_hist"
                                ),
                            ),
                        ]
                    )
                elif self.task_type == "classification":
                    return xgb.XGBClassifier(n_estimators=100, tree_method="gpu_hist")
                else:  # regression
                    return xgb.XGBRegressor(n_estimators=100, tree_method="gpu_hist")
            except ImportError:
                logger.info(
                    "XGBoost not available, falling back to scikit-learn models"
                )
                # Fallback to scikit-learn models
                if self.task_type == "text_classification":
                    return Pipeline(
                        [
                            ("tfidf", TfidfVectorizer(max_features=1000)),
                            ("classifier", RandomForestClassifier(n_estimators=100)),
                        ]
                    )
                elif self.task_type == "classification":
                    return RandomForestClassifier(n_estimators=100)
                else:  # regression
                    return RandomForestRegressor(n_estimators=100)
        else:
            # CPU models
            if self.task_type == "text_classification":
                return Pipeline(
                    [
                        ("tfidf", TfidfVectorizer(max_features=1000)),
                        ("classifier", RandomForestClassifier(n_estimators=100)),
                    ]
                )
            elif self.task_type == "classification":
                return RandomForestClassifier(n_estimators=100)
            else:  # regression
                return RandomForestRegressor(n_estimators=100)

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate model performance based on task type."""
        if self.task_type in ["classification", "text_classification"]:
            return accuracy_score(y_true, y_pred)
        else:  # regression
            return r2_score(y_true, y_pred)

    def evaluate(self) -> Dict:
        """
        Evaluate utility using TSTR methodology.

        Returns:
            Dict: Dictionary containing task information and classification reports for both models
        """
        results = {}

        # Only run TSTR if selected
        if "tstr_accuracy" in self.selected_metrics:
            try:
                syn_size = len(self.synthetic_data)
                real_total = len(self.original_data)
                logger.info(
                    "Utility sampling stats -> real_total=%d, synthetic_total=%d",
                    real_total,
                    syn_size,
                )

                if self.real_train_data is not None:
                    train_data = self.real_train_data.copy()
                else:
                    train_data = self.original_data.copy()

                if self.real_test_data is not None and not self.real_test_data.empty:
                    test_data = self.real_test_data.copy()
                else:
                    base_real = train_data.copy()
                    if len(base_real) < 3:
                        raise ValueError(
                            "Not enough real samples to generate a hold-out split for utility evaluation."
                        )
                    test_size = max(1, int(len(base_real) * 0.2))
                    if test_size >= len(base_real):
                        test_size = max(1, len(base_real) - 1)
                    base_train, test_data = train_test_split(
                        base_real, test_size=test_size, random_state=42, shuffle=True
                    )
                    train_data = base_train
                    logger.info(
                        "Generated train/test split from real data -> train_size=%d, test_size=%d",
                        len(train_data),
                        len(test_data),
                    )

                train_data = train_data.reset_index(drop=True)
                test_data = test_data.reset_index(drop=True)
                logger.info(
                    "Final utility datasets -> train_real=%d, test_real=%d, train_syn=%d",
                    len(train_data),
                    len(test_data),
                    syn_size,
                )
                if test_data.empty:
                    raise ValueError("Real test data is empty after preprocessing.")

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
                    results["tstr_accuracy"] = {
                        "error": "No test data available - real data is smaller than synthetic data",
                        "task_type": self.task_type,
                        "input_columns": self.input_columns,
                        "output_columns": self.output_columns,
                    }
                    return results

                # Handle text columns and prepare features
                text_columns = [
                    col
                    for col in self.input_columns
                    if col in self.metadata["columns"]
                    and self.metadata["columns"][col]["sdtype"] == "text"
                ]

                if text_columns:
                    # Initialize vectorizer
                    vectorizer = TfidfVectorizer(max_features=1000)

                    # Fit vectorizer on training data and transform all datasets
                    X_train_real = vectorizer.fit_transform(
                        X_train_real[text_columns[0]].fillna("")
                    )
                    X_train_syn = vectorizer.transform(
                        X_train_syn[text_columns[0]].fillna("")
                    )
                    X_test = vectorizer.transform(X_test[text_columns[0]].fillna(""))

                    # Convert to GPU tensors if available for faster computation
                    if self.device == "cuda":
                        import torch

                        X_train_real = torch.tensor(
                            X_train_real.toarray(), device="cuda", dtype=torch.float32
                        )
                        X_train_syn = torch.tensor(
                            X_train_syn.toarray(), device="cuda", dtype=torch.float32
                        )
                        X_test = torch.tensor(
                            X_test.toarray(), device="cuda", dtype=torch.float32
                        )
                    else:
                        # Convert to dense arrays if needed
                        X_train_real = X_train_real.toarray()
                        X_train_syn = X_train_syn.toarray()
                        X_test = X_test.toarray()

                # Handle non-text columns if any
                other_columns = [
                    col for col in self.input_columns if col not in text_columns
                ]
                if other_columns:
                    # Fill NaN values with 0 for numerical columns
                    X_train_real_other = train_data[other_columns].fillna(0).values
                    X_train_syn_other = (
                        self.synthetic_data[other_columns].fillna(0).values
                    )
                    X_test_other = test_data[other_columns].fillna(0).values

                    # Convert to GPU tensors if available
                    if self.device == "cuda":
                        import torch

                        X_train_real_other = torch.tensor(
                            X_train_real_other, device="cuda", dtype=torch.float32
                        )
                        X_train_syn_other = torch.tensor(
                            X_train_syn_other, device="cuda", dtype=torch.float32
                        )
                        X_test_other = torch.tensor(
                            X_test_other, device="cuda", dtype=torch.float32
                        )

                    # Combine with text features if they exist
                    if text_columns:
                        if self.device == "cuda":
                            X_train_real = torch.cat(
                                [X_train_real, X_train_real_other], dim=1
                            )
                            X_train_syn = torch.cat(
                                [X_train_syn, X_train_syn_other], dim=1
                            )
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
                y_train_real = y_train_real.fillna(
                    y_train_real.mean()
                    if self.task_type == "regression"
                    else y_train_real.mode().iloc[0]
                )
                y_train_syn = y_train_syn.fillna(
                    y_train_syn.mean()
                    if self.task_type == "regression"
                    else y_train_syn.mode().iloc[0]
                )
                y_test = y_test.fillna(
                    y_test.mean()
                    if self.task_type == "regression"
                    else y_test.mode().iloc[0]
                )

                # For classification tasks, ensure all datasets have the same class labels
                if self.task_type in ["classification", "text_classification"]:
                    # Get all unique values from all datasets
                    all_values = set()
                    all_values.update(y_train_real.unique())
                    all_values.update(y_train_syn.unique())
                    all_values.update(y_test.unique())

                    # Convert to sorted list for consistent ordering
                    all_values = sorted(list(all_values))
                    logger.info(f"All unique target values: {all_values}")

                    # Create label encoder to map values to 0, 1, 2, ...
                    from sklearn.preprocessing import LabelEncoder

                    label_encoder = LabelEncoder()
                    label_encoder.fit(all_values)

                    # Transform all target variables
                    y_train_real = label_encoder.transform(y_train_real)
                    y_train_syn = label_encoder.transform(y_train_syn)
                    y_test = label_encoder.transform(y_test)

                    logger.info(f"Encoded classes: {label_encoder.classes_}")
                    logger.info(f"Expected class range: 0 to {len(all_values)-1}")

                # Train two separate models
                model_real = self._get_model()
                model_syn = self._get_model()

                # Convert data back to CPU if using GPU tensors for scikit-learn models
                if self.device == "cuda" and not hasattr(
                    model_real, "tree_method"
                ):  # Not XGBoost
                    import torch

                    def _to_cpu_array(tensor_or_array):
                        if hasattr(tensor_or_array, "detach") and hasattr(
                            tensor_or_array, "cpu"
                        ):
                            return tensor_or_array.detach().cpu().numpy()
                        return tensor_or_array

                    X_train_real_cpu = _to_cpu_array(X_train_real)
                    X_train_syn_cpu = _to_cpu_array(X_train_syn)
                    X_test_cpu = _to_cpu_array(X_test)
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

                training_size = len(train_data)
                test_size = len(test_data)
                total_samples = len(self.original_data)

                if self.task_type in ["classification", "text_classification"]:
                    from sklearn.metrics import classification_report

                    real_report = classification_report(
                        y_test, real_pred, output_dict=True
                    )
                    syn_report = classification_report(
                        y_test, syn_pred, output_dict=True
                    )

                    results["tstr_accuracy"] = {
                        "task_type": self.task_type,
                        "input_columns": self.input_columns,
                        "output_columns": self.output_columns,
                        "training_size": training_size,
                        "test_size": test_size,
                        "total_samples": total_samples,
                        "real_data_model": real_report,
                        "synthetic_data_model": syn_report,
                    }
                    real_acc = real_report.get("accuracy")
                    syn_acc = syn_report.get("accuracy")
                    real_f1 = real_report.get("macro avg", {}).get("f1-score")
                    syn_f1 = syn_report.get("macro avg", {}).get("f1-score")
                    logger.info(
                        "Utility metrics (classification) -> real accuracy %.4f, synthetic accuracy %.4f, "
                        "real F1 %.4f, synthetic F1 %.4f",
                        (
                            real_acc
                            if isinstance(real_acc, (int, float))
                            else float("nan")
                        ),
                        syn_acc if isinstance(syn_acc, (int, float)) else float("nan"),
                        real_f1 if isinstance(real_f1, (int, float)) else float("nan"),
                        syn_f1 if isinstance(syn_f1, (int, float)) else float("nan"),
                    )
                else:

                    def _regression_metrics(y_true, y_hat):
                        mse = mean_squared_error(y_true, y_hat)
                        rmse = float(np.sqrt(mse))
                        mae = float(mean_absolute_error(y_true, y_hat))
                        try:
                            r2 = float(r2_score(y_true, y_hat))
                        except Exception:
                            r2 = float("nan")
                        return {"rmse": rmse, "mae": mae, "r2": r2}

                    real_metrics = _regression_metrics(y_test, real_pred)
                    syn_metrics = _regression_metrics(y_test, syn_pred)

                    results["tstr_accuracy"] = {
                        "task_type": self.task_type,
                        "input_columns": self.input_columns,
                        "output_columns": self.output_columns,
                        "training_size": training_size,
                        "test_size": test_size,
                        "total_samples": total_samples,
                        "real_data_model": real_metrics,
                        "synthetic_data_model": syn_metrics,
                    }
                    logger.info(
                        "Utility metrics (regression) -> real RMSE %.4f, synthetic RMSE %.4f, "
                        "real MAE %.4f, synthetic MAE %.4f, real R² %.4f, synthetic R² %.4f",
                        real_metrics.get("rmse", float("nan")),
                        syn_metrics.get("rmse", float("nan")),
                        real_metrics.get("mae", float("nan")),
                        syn_metrics.get("mae", float("nan")),
                        real_metrics.get("r2", float("nan")),
                        syn_metrics.get("r2", float("nan")),
                    )

            except Exception as e:
                logger.error(f"Error in TSTR evaluation: {str(e)}")
                results["tstr_accuracy"] = {
                    "error": str(e),
                    "task_type": self.task_type,
                    "input_columns": self.input_columns,
                    "output_columns": self.output_columns,
                }

        # Only run correlation analysis if selected
        if "correlation_analysis" in self.selected_metrics:
            # This would be implemented separately or use the existing evaluate_correlation method
            results["correlation_analysis"] = {
                "note": "Correlation analysis requires specific correlation types to be specified"
            }

        return results

    def evaluate_correlation(self, correlation_types, **kwargs):
        """
        Evaluate selected cross-modality correlation metrics.
        correlation_types: list of str, e.g. ['sentiment_rating', 'keyword_category']
        kwargs: extra params for each correlation type
        """
        results = {}
        df = self.synthetic_data  # or allow user to choose real/syn
        for corr in correlation_types:
            if corr == "sentiment_rating":
                results["sentiment_rating"] = sentiment_rating_correlation(
                    df,
                    rating_col=kwargs.get("rating_col", "rating"),
                    text_col=kwargs.get("text_col", "review"),
                )
            elif corr == "keyword_category":
                results["keyword_category"] = keyword_category_correlation(
                    df,
                    category_col=kwargs.get("category_col", "category"),
                    text_col=kwargs.get("text_col", "review"),
                    top_n=kwargs.get("top_n", 50),
                ).to_dict(orient="records")
            elif corr == "numeric_length":
                results["numeric_length"] = numeric_length_correlation(
                    df,
                    numeric_col=kwargs.get("numeric_col", "price"),
                    text_col=kwargs.get("text_col", "review"),
                )
            elif corr == "semantic_tabular":
                # User needs to provide text_embeddings
                results["semantic_tabular"] = semantic_tabular_correlation(
                    df,
                    categorical_cols=kwargs.get("categorical_cols", ["category"]),
                    text_embeddings=kwargs["text_embeddings"],
                )
            elif corr == "pii_text_leakage":
                results["pii_text_leakage"] = pii_text_leakage(
                    df,
                    pii_col=kwargs.get("pii_col", "user_id"),
                    text_col=kwargs.get("text_col", "review"),
                )
        return results
