#!/usr/bin/env python3
"""
Create Tilted Data — adversarial baseline for the Stitching Fallacy evaluation.

Tilted data keeps all tabular columns intact and randomly shuffles the text
column across rows, destroying cross-modal joint logic while preserving
marginal distributions perfectly.

Expected behavior:
  - Traditional isolated metrics (SDV, BERTScore): high scores (marginals are perfect)
  - Joint evaluation framework (JSD, NMI gap): low scores (correlation is broken)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXPERIMENT_DIR = BASE_DIR / "experiments" / "baselines_filtered_20260428_195011"
SYNTHETIC_DATA_DIR = EXPERIMENT_DIR / "synthetic_data"

DATASETS = {
    'amazon_reviews': {
        'original_file': 'amazon_reviews_original.csv',
        'text_col': 'text',
        'tabular_cols': ['rating']
    },
    'kiva_loans': {
        'original_file': 'kiva_loans_original.csv',
        'text_col': 'use',
        'tabular_cols': ['sector']
    },
    'fake_jobs': {
        'original_file': 'fake_jobs_original.csv',
        'text_col': 'description',
        'tabular_cols': ['fraudulent', 'has_company_logo']
    }
}


def create_tilted_data(dataset_name, config):
    """
    Shuffle the text column of a dataset to break cross-modal correlations.

    Args:
        dataset_name: dataset identifier
        config: dict with 'original_file', 'text_col', 'tabular_cols'

    Returns:
        Path to the saved tilted CSV file
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.replace('_', ' ').title()}")
    print(f"{'='*80}")

    original_file = SYNTHETIC_DATA_DIR / config['original_file']
    df_original = pd.read_csv(original_file)

    print(f"  Loaded: {len(df_original)} rows, columns: {list(df_original.columns)}")

    df_tilted = df_original.copy()
    text_col = config['text_col']
    tabular_cols = config['tabular_cols']

    # Shuffle text column with a fixed seed for reproducibility
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df_tilted))
    df_tilted[text_col] = df_original[text_col].iloc[shuffled_indices].values

    print(f"  Shuffled text column: '{text_col}'")
    print(f"  Tabular columns unchanged: {tabular_cols}")

    # Verify marginal distributions are preserved
    print(f"\n  Marginal distribution check:")
    for col in tabular_cols:
        orig_dist = df_original[col].value_counts(normalize=True).sort_index()
        tilt_dist = df_tilted[col].value_counts(normalize=True).sort_index()
        identical = orig_dist.equals(tilt_dist)
        print(f"    {col}: {'PASS (identical)' if identical else 'FAIL (different)'}")

    orig_texts = set(df_original[text_col].dropna())
    tilt_texts = set(df_tilted[text_col].dropna())
    print(f"    {text_col} content set: {'PASS (identical)' if orig_texts == tilt_texts else 'FAIL (different)'}")

    # Verify joint distribution is broken
    n_check = min(5, len(df_original))
    n_mismatches = sum(
        df_original[text_col].iloc[i] != df_tilted[text_col].iloc[i]
        for i in range(n_check)
    )
    print(f"\n  Joint distribution check (first {n_check} rows):")
    print(f"    Mismatches: {n_mismatches}/{n_check} — {'BROKEN (expected)' if n_mismatches > 0 else 'WARNING: not broken'}")

    output_file = SYNTHETIC_DATA_DIR / f"{dataset_name}_tilted.csv"
    df_tilted.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)")

    # Show a brief side-by-side comparison
    print(f"\n  Sample comparison (first 3 rows):")
    print(f"  Original:")
    for i in range(min(3, len(df_original))):
        tab = ', '.join(f"{col}={df_original[col].iloc[i]}" for col in tabular_cols)
        txt = str(df_original[text_col].iloc[i])[:60] + "..."
        print(f"    [{i}] {tab} | {text_col}={txt}")
    print(f"  Tilted (text shuffled):")
    for i in range(min(3, len(df_tilted))):
        tab = ', '.join(f"{col}={df_tilted[col].iloc[i]}" for col in tabular_cols)
        txt = str(df_tilted[text_col].iloc[i])[:60] + "..."
        print(f"    [{i}] {tab} | {text_col}={txt}")

    return output_file


def main():
    print("="*80)
    print("Create Tilted Data — Adversarial Baseline")
    print("="*80)
    print("\nGoal:")
    print("  Marginal distributions: 100% preserved")
    print("  Joint distribution:     deliberately destroyed")
    print("\nExpected evaluation outcome:")
    print("  Traditional isolated metrics (SDV, BERTScore): high scores")
    print("  Joint evaluation framework (JSD, NMI gap):     low scores")

    created_files = []

    for dataset_name, config in DATASETS.items():
        output_file = create_tilted_data(dataset_name, config)
        created_files.append(output_file.name)

    print(f"\n{'='*80}")
    print("Tilted data created.")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    for filename in created_files:
        print(f"  {filename}")

    print(f"\nOutput directory: {SYNTHETIC_DATA_DIR}")
    print("\nNext step: run syneval_four_dimensions.py to verify the framework detects the broken joint logic.")


if __name__ == "__main__":
    main()
