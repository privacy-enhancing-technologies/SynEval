#!/usr/bin/env python3
"""
Category Coverage Analysis.

For each categorical column, measures:
  1. Coverage:    fraction of real categories present in synthetic data
                  (|synth ∩ real| / |real|)
  2. Fabrication: number of categories in synthetic data not seen in real data
                  (|synth - real|)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXPERIMENT_DIR = BASE_DIR / "experiments" / "baselines_filtered_20260428_195011"
SYNTHETIC_DATA_DIR = EXPERIMENT_DIR / "synthetic_data"

DATASETS = {
    'amazon_reviews': {
        'original': 'amazon_reviews_original.csv',
        'categorical_cols': ['rating'],
        'baselines': [
            'amazon_reviews_baseline1_independent.csv',
            'amazon_reviews_baseline2_sequential.csv',
            'amazon_reviews_baseline3_joint_llm.csv',
            'amazon_reviews_baseline4_tabsyn.csv',
            'amazon_reviews_tilted.csv'
        ]
    },
    'kiva_loans': {
        'original': 'kiva_loans_original.csv',
        'categorical_cols': ['sector'],
        'baselines': [
            'kiva_loans_baseline1_independent.csv',
            'kiva_loans_baseline2_sequential.csv',
            'kiva_loans_baseline3_joint_llm.csv',
            'kiva_loans_baseline4_tabsyn.csv',
            'kiva_loans_tilted.csv'
        ]
    },
    'fake_jobs': {
        'original': 'fake_jobs_original.csv',
        'categorical_cols': ['fraudulent', 'has_company_logo'],
        'baselines': [
            'fake_jobs_baseline1_independent.csv',
            'fake_jobs_baseline2_sequential.csv',
            'fake_jobs_baseline3_joint_llm.csv',
            'fake_jobs_baseline4_tabsyn.csv',
            'fake_jobs_tilted.csv'
        ]
    }
}


def compute_category_coverage(real_categories, synth_categories):
    """
    Compute coverage and fabrication statistics for a single column.

    Returns:
        dict with keys: coverage, real_count, synth_count, missing, fabricated
    """
    real_set = set(real_categories)
    synth_set = set(synth_categories)

    intersection = real_set & synth_set
    coverage = len(intersection) / len(real_set) if real_set else 0.0
    missing = real_set - synth_set
    fabricated = synth_set - real_set

    return {
        'coverage': coverage,
        'real_count': len(real_set),
        'synth_count': len(synth_set),
        'missing': sorted(list(missing)),
        'fabricated': sorted(list(fabricated))
    }


def main():
    print("="*80)
    print("Category Coverage Analysis")
    print("="*80)

    all_results = []

    for dataset_name, config in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")

        real_file = SYNTHETIC_DATA_DIR / config['original']
        real_df = pd.read_csv(real_file)
        categorical_cols = config['categorical_cols']

        print(f"\nReal data category distributions:")
        for col in categorical_cols:
            real_categories = real_df[col].unique()
            print(f"  {col}: {len(real_categories)} categories — {sorted(real_categories)[:10]}")

        for baseline_file in config['baselines']:
            baseline_name = baseline_file.replace('.csv', '').replace(f'{dataset_name}_', '')

            synth_file = SYNTHETIC_DATA_DIR / baseline_file
            if not synth_file.exists():
                continue

            synth_df = pd.read_csv(synth_file)
            print(f"\n--- {baseline_name} ---")

            for col in categorical_cols:
                real_categories = real_df[col].dropna().unique()

                if col not in synth_df.columns:
                    print(f"  {col}: column absent (baseline4 format)")
                    continue

                synth_categories = synth_df[col].dropna().unique()
                result = compute_category_coverage(real_categories, synth_categories)

                coverage_pct = result['coverage'] * 100
                status = "OK" if result['coverage'] == 1.0 else "WARN"
                print(f"  {col}: [{status}] Coverage = {coverage_pct:.1f}%  "
                      f"({result['synth_count']}/{result['real_count']} categories)")

                if result['missing']:
                    print(f"    Missing:    {result['missing']}")
                if result['fabricated']:
                    print(f"    Fabricated: {result['fabricated']}")

                all_results.append({
                    'dataset': dataset_name,
                    'baseline': baseline_name,
                    'column': col,
                    'coverage': result['coverage'],
                    'real_categories': result['real_count'],
                    'synth_categories': result['synth_count'],
                    'missing_count': len(result['missing']),
                    'fabricated_count': len(result['fabricated'])
                })

    # Summary table
    print(f"\n{'='*80}")
    print("Summary — Category Coverage")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)

    for dataset_name in DATASETS.keys():
        dataset_results = results_df[results_df['dataset'] == dataset_name]
        if len(dataset_results) == 0:
            continue

        print(f"\n{dataset_name.upper()}:")
        print(f"{'Baseline':<25} {'Column':<20} {'Coverage':<10} {'Real':<6} {'Synth':<6} {'Missing':<8} {'Fabricated':<10}")
        print("-" * 95)

        for _, row in dataset_results.iterrows():
            coverage_str = f"{row['coverage']*100:.1f}%"
            marker = " *" if row['coverage'] < 1.0 or row['fabricated_count'] > 0 else ""
            print(f"{row['baseline']:<25} {row['column']:<20} {coverage_str:<10} {row['real_categories']:<6} "
                  f"{row['synth_categories']:<6} {row['missing_count']:<8} {row['fabricated_count']:<10}{marker}")

    # Tilted data verification
    print(f"\n{'='*80}")
    print("Tilted Data Verification")
    print(f"{'='*80}")
    print("\nTilted data should achieve 100% coverage (it is a row-permutation of real data):")

    tilted_results = results_df[results_df['baseline'] == 'tilted']
    for _, row in tilted_results.iterrows():
        coverage_pct = row['coverage'] * 100
        status = "PASS" if row['coverage'] == 1.0 else "FAIL"
        print(f"  {row['dataset']:20s} - {row['column']:20s}: [{status}] {coverage_pct:.1f}%")

    if len(tilted_results) > 0 and tilted_results['coverage'].min() == 1.0:
        print("\nVerification passed: tilted data covers all categories (marginals preserved).")
    elif len(tilted_results) > 0:
        print("\nWarning: tilted data does not fully cover all categories — check generation.")


if __name__ == "__main__":
    main()
