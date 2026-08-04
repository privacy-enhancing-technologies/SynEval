#!/usr/bin/env python3
"""
SynEval Core: Semantic Quantization and Space Mapping

Step 1 — Fit (on real data):
  - Tabular columns:
      Continuous numeric (e.g. rating) -> quantile binning (10 bins)
      Categorical (e.g. sector, fraudulent) -> direct category encoding
  - Text column:
      SBERT embeddings -> K-Means clustering (K=20)

Step 2 — Transform (applied to all data):
  Maps original, baseline, and tilted datasets onto a discrete 2-D grid (C_X, C_T)
  C_X: tabular bin or category
  C_T: text K-Means cluster ID

This quantized representation is the foundation for the four-dimension evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("Importing libraries...")
print("  pandas, numpy")
print("  scikit-learn (KBins, KMeans)")
print("  SentenceTransformer")

# --- Paths (relative to this script) ---
BASE_DIR = Path(__file__).parent
EXPERIMENT_DIR = BASE_DIR / "experiments" / "baselines_filtered_20260428_195011"
SYNTHETIC_DATA_DIR = EXPERIMENT_DIR / "synthetic_data"
OUTPUT_DIR = EXPERIMENT_DIR / "syneval"
FITTED_MODELS_DIR = OUTPUT_DIR / "fitted_models"
QUANTIZED_DATA_DIR = OUTPUT_DIR / "quantized_data"
LOGS_DIR = OUTPUT_DIR / "logs"

FITTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
QUANTIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Quantization parameters ---
N_BINS_NUMERIC = 10    # bins for continuous variables
N_CLUSTERS_TEXT = 20   # K-Means clusters for text
SBERT_MODEL = 'all-MiniLM-L6-v2'

# --- Dataset configuration ---
DATASETS = {
    'amazon_reviews': {
        'original': 'amazon_reviews_original.csv',
        'text_col': 'text',
        'tabular_cols': ['rating'],
        'tabular_types': {'rating': 'ordinal'},
        'files': [
            'amazon_reviews_original.csv',
            'amazon_reviews_baseline1_independent.csv',
            'amazon_reviews_baseline2_sequential.csv',
            'amazon_reviews_baseline3_joint_llm.csv',
            'amazon_reviews_baseline4_tabsyn.csv',
            'amazon_reviews_tilted.csv'
        ]
    },
    'kiva_loans': {
        'original': 'kiva_loans_original.csv',
        'text_col': 'use',
        'tabular_cols': ['sector'],
        'tabular_types': {'sector': 'categorical'},
        'baseline4_is_encoded': True,
        'files': [
            'kiva_loans_original.csv',
            'kiva_loans_baseline1_independent.csv',
            'kiva_loans_baseline2_sequential.csv',
            'kiva_loans_baseline3_joint_llm.csv',
            'kiva_loans_baseline4_tabsyn.csv',
            'kiva_loans_tilted.csv'
        ]
    },
    'fake_jobs': {
        'original': 'fake_jobs_original.csv',
        'text_col': 'description',
        'tabular_cols': ['fraudulent', 'has_company_logo'],
        'tabular_types': {'fraudulent': 'ordinal', 'has_company_logo': 'ordinal'},
        'files': [
            'fake_jobs_original.csv',
            'fake_jobs_baseline1_independent.csv',
            'fake_jobs_baseline2_sequential.csv',
            'fake_jobs_baseline3_joint_llm.csv',
            'fake_jobs_baseline4_tabsyn.csv',
            'fake_jobs_tilted.csv'
        ]
    }
}


class SemanticQuantizer:
    """
    Maps multimodal data to a discrete 2-D space (C_X, C_T).

    C_X: discretized tabular column (bin index or category code)
    C_T: text K-Means cluster ID
    """

    def __init__(self, dataset_name, text_col, tabular_cols, tabular_types,
                 n_bins=10, n_clusters=20, sbert_model='all-MiniLM-L6-v2'):
        self.dataset_name = dataset_name
        self.text_col = text_col
        self.tabular_cols = tabular_cols
        self.tabular_types = tabular_types
        self.n_bins = n_bins
        self.n_clusters = n_clusters
        self.sbert_model_name = sbert_model

        self.sbert_model = None
        self.kmeans = None
        self.discretizers = {}       # {col_name: KBinsDiscretizer} for numeric cols
        self.category_encoders = {}  # {col_name: {category: code}} for categorical cols
        self.fitted = False

    def fit(self, real_df):
        """
        Fit the quantizer on real data.

        Args:
            real_df: DataFrame of real (reference) data
        """
        print(f"\n{'='*80}")
        print(f"Fitting quantizer: {self.dataset_name.replace('_', ' ').title()}")
        print(f"{'='*80}")

        # 1. Fit tabular columns
        print(f"\n[1/2] Fitting tabular column quantizers...")
        for col in self.tabular_cols:
            col_type = self.tabular_types[col]

            if col_type == 'numeric':
                print(f"  - {col} (numeric) -> quantile binning ({self.n_bins} bins)")
                discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode='ordinal',
                    strategy='quantile',
                    subsample=None
                )
                discretizer.fit(real_df[[col]].values)
                self.discretizers[col] = discretizer
                edges = discretizer.bin_edges_[0]
                print(f"    bin edges: {edges[:3].tolist()} ... {edges[-3:].tolist()}")

            elif col_type == 'ordinal':
                print(f"  - {col} (ordinal) -> use raw values directly")
                unique_values = sorted(real_df[col].unique())
                print(f"    values: {unique_values}")

            elif col_type == 'categorical':
                print(f"  - {col} (categorical) -> category encoding")
                categories = real_df[col].unique()
                encoder = {cat: idx for idx, cat in enumerate(categories)}
                self.category_encoders[col] = encoder
                print(f"    {len(categories)} categories ({list(categories[:5])}...)")

        # 2. Fit text column (SBERT + K-Means)
        print(f"\n[2/2] Fitting text column quantizer...")
        print(f"  - Loading SBERT model: {self.sbert_model_name}")
        self.sbert_model = SentenceTransformer(self.sbert_model_name)

        print(f"  - Encoding text to embeddings...")
        texts = [str(t) for t in real_df[self.text_col] if pd.notna(t)]
        embeddings = self.sbert_model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype(np.float64)

        print(f"  - K-Means clustering (K={self.n_clusters})...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(embeddings)

        cluster_sizes = pd.Series(self.kmeans.labels_).value_counts().sort_index()
        print(f"    cluster sizes (first 10):")
        for cluster_id, size in cluster_sizes.head(10).items():
            print(f"      Cluster {cluster_id}: {size} samples")
        if len(cluster_sizes) > 10:
            print(f"      ... ({len(cluster_sizes)} clusters total)")

        self.fitted = True
        print(f"\nQuantizer fitted successfully.")

    def transform(self, df, is_baseline4=False, baseline4_is_encoded=False):
        """
        Transform data to the discrete (C_X, C_T) space.

        Args:
            df: DataFrame to transform
            is_baseline4: whether the data comes from baseline4 (pre-computed SBERT embeddings)
            baseline4_is_encoded: whether baseline4's categorical column is already integer-encoded

        Returns:
            quantized_df: DataFrame with C_X and C_T columns
        """
        if not self.fitted:
            raise RuntimeError("Quantizer must be fitted before calling transform().")

        result = pd.DataFrame()

        # 1. Transform tabular columns -> C_X
        tabular_bins = []
        for col in self.tabular_cols:
            col_type = self.tabular_types[col]

            if col_type == 'numeric':
                bins = self.discretizers[col].transform(df[[col]].values).astype(int).flatten()
                tabular_bins.append(bins)
                result[f'C_X_{col}'] = bins

            elif col_type == 'ordinal':
                bins = df[col].astype(int).values
                tabular_bins.append(bins)
                result[f'C_X_{col}'] = bins

            elif col_type == 'categorical':
                if is_baseline4 and baseline4_is_encoded:
                    bins = df[col].astype(int).values
                    tabular_bins.append(bins)
                    result[f'C_X_{col}'] = bins
                else:
                    encoder = self.category_encoders[col]
                    bins = df[col].map(lambda x: encoder.get(x, -1))
                    tabular_bins.append(bins.values)
                    result[f'C_X_{col}'] = bins

        if len(self.tabular_cols) == 1:
            result['C_X'] = result[f'C_X_{self.tabular_cols[0]}']
        else:
            combined = np.column_stack(tabular_bins)
            result['C_X'] = [tuple(row) for row in combined]

        # 2. Transform text column -> C_T
        if is_baseline4:
            sbert_cols = [c for c in df.columns if c.startswith('sbert_')]
            embeddings = df[sbert_cols].values
        else:
            texts = [str(t) for t in df[self.text_col] if pd.notna(t)]
            embeddings = self.sbert_model.encode(texts, show_progress_bar=False, batch_size=32)
            embeddings = embeddings.astype(np.float64)

        clusters = self.kmeans.predict(embeddings)
        result['C_T'] = clusters

        return result

    def save(self, filepath):
        """Persist the fitted quantizer to disk."""
        if not self.fitted:
            raise RuntimeError("Quantizer is not fitted; nothing to save.")

        save_data = {
            'dataset_name': self.dataset_name,
            'text_col': self.text_col,
            'tabular_cols': self.tabular_cols,
            'tabular_types': self.tabular_types,
            'n_bins': self.n_bins,
            'n_clusters': self.n_clusters,
            'sbert_model_name': self.sbert_model_name,
            'kmeans': self.kmeans,
            'discretizers': self.discretizers,
            'category_encoders': self.category_encoders,
            'fitted': self.fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"  Saved quantizer: {filepath.name}")

    @classmethod
    def load(cls, filepath):
        """Load a previously saved quantizer from disk."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        quantizer = cls(
            dataset_name=save_data['dataset_name'],
            text_col=save_data['text_col'],
            tabular_cols=save_data['tabular_cols'],
            tabular_types=save_data['tabular_types'],
            n_bins=save_data['n_bins'],
            n_clusters=save_data['n_clusters'],
            sbert_model=save_data['sbert_model_name']
        )

        quantizer.kmeans = save_data['kmeans']
        quantizer.discretizers = save_data['discretizers']
        quantizer.category_encoders = save_data['category_encoders']
        quantizer.fitted = save_data['fitted']
        quantizer.sbert_model = SentenceTransformer(save_data['sbert_model_name'])

        return quantizer


def process_dataset(dataset_name, config):
    """Fit quantizer on one dataset and transform all associated files."""

    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name.replace('_', ' ').title()}")
    print(f"{'='*80}")

    original_file = SYNTHETIC_DATA_DIR / config['original']
    real_df = pd.read_csv(original_file)
    print(f"  Loaded real data: {len(real_df)} rows")

    quantizer = SemanticQuantizer(
        dataset_name=dataset_name,
        text_col=config['text_col'],
        tabular_cols=config['tabular_cols'],
        tabular_types=config['tabular_types'],
        n_bins=N_BINS_NUMERIC,
        n_clusters=N_CLUSTERS_TEXT,
        sbert_model=SBERT_MODEL
    )

    quantizer.fit(real_df)

    model_file = FITTED_MODELS_DIR / f"{dataset_name}_quantizer.pkl"
    quantizer.save(model_file)

    print(f"\n{'='*80}")
    print(f"Transforming all files to discrete (C_X, C_T) space")
    print(f"{'='*80}")

    results_summary = []

    for file_name in config['files']:
        file_path = SYNTHETIC_DATA_DIR / file_name

        if not file_path.exists():
            print(f"\n  Skipping (not found): {file_name}")
            continue

        print(f"\n--- {file_name} ---")
        df = pd.read_csv(file_path)
        print(f"  Loaded: {len(df)} rows")

        is_baseline4 = 'baseline4' in file_name
        baseline4_is_encoded = config.get('baseline4_is_encoded', False)

        quantized_df = quantizer.transform(df, is_baseline4=is_baseline4,
                                           baseline4_is_encoded=baseline4_is_encoded)

        output_name = file_name.replace('.csv', '_quantized.csv')
        output_file = QUANTIZED_DATA_DIR / output_name
        quantized_df.to_csv(output_file, index=False)

        n_unique_CX = quantized_df['C_X'].nunique()
        n_unique_CT = quantized_df['C_T'].nunique()
        coverage = n_unique_CX * n_unique_CT

        print(f"  Quantization complete:")
        print(f"    Unique C_X: {n_unique_CX}")
        print(f"    Unique C_T: {n_unique_CT}")
        print(f"    Grid coverage: {coverage} cells")
        print(f"  Saved: {output_name}")

        results_summary.append({
            'file': file_name,
            'rows': len(df),
            'unique_CX': n_unique_CX,
            'unique_CT': n_unique_CT,
            'coverage': coverage
        })

    return results_summary


def main():
    print("="*80)
    print("SynEval — Semantic Quantization")
    print("="*80)
    print("\nProcedure:")
    print("  1. Fit: learn quantizer on real data")
    print("     - Tabular: numeric binning | category encoding")
    print("     - Text: SBERT + K-Means")
    print("  2. Transform: map all datasets to discrete (C_X, C_T) space")
    print("\nParameters:")
    print(f"  Numeric bins: {N_BINS_NUMERIC}")
    print(f"  Text clusters: {N_CLUSTERS_TEXT}")
    print(f"  SBERT model: {SBERT_MODEL}")

    all_results = {}

    for dataset_name, config in DATASETS.items():
        results = process_dataset(dataset_name, config)
        all_results[dataset_name] = results

    summary_file = OUTPUT_DIR / "quantization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Semantic quantization complete.")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  Fitted models:   {FITTED_MODELS_DIR}")
    print(f"  Quantized data:  {QUANTIZED_DATA_DIR}")
    print(f"  Summary report:  {summary_file}")

    print(f"\n{'='*80}")
    print("Quantization summary")
    print(f"{'='*80}")

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        print(f"{'File':<40} {'Rows':<8} {'C_X':<8} {'C_T':<8} {'Coverage':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['file']:<40} {r['rows']:<8} {r['unique_CX']:<8} {r['unique_CT']:<8} {r['coverage']:<10}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext step: run syneval_four_dimensions.py to compute evaluation metrics.")


if __name__ == "__main__":
    main()
