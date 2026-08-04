#!/usr/bin/env python3
"""
Dataset Discovery Pipeline.

Scans Data/ folder for CSV files, computes tabular-text correlation strength,
outputs ranked report identifying top datasets for multimodal experiments.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats.contingency import association

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.quantization import SemanticQuantizer
from evaluation.column_detector import auto_detect_columns


def scan_datasets(data_dir: str, sample_size: int = 10000):
    """
    Scan directory for CSV files.

    Args:
        data_dir: Path to data directory
        sample_size: Max rows to sample per dataset

    Yields:
        (dataset_name, dataframe) tuples
    """
    data_path = Path(data_dir)

    for csv_file in data_path.rglob("*.csv"):
        # Skip very large files
        if csv_file.stat().st_size > 500 * 1024 * 1024:  # 500MB
            print(f"Skipping {csv_file} (too large)")
            continue

        try:
            # Read with sampling
            df = pd.read_csv(csv_file)
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)

            dataset_name = csv_file.name
            yield dataset_name, df

        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue


def compute_correlation_metrics(df: pd.DataFrame, text_col: str, tabular_cols: list):
    """
    Compute correlation metrics between text and tabular columns.

    Args:
        df: DataFrame
        text_col: Text column name
        tabular_cols: List of tabular column names

    Returns:
        Dict of correlation metrics
    """
    # Clean data: remove rows with NaN in text or tabular columns
    cols_to_check = [text_col] + tabular_cols
    df_clean = df[cols_to_check].dropna()

    # Need at least 100 samples for meaningful statistics
    if len(df_clean) < 100:
        return None

    # Quantize data
    quantizer = SemanticQuantizer(
        text_columns=[text_col],
        tabular_columns=tabular_cols,
        adaptive=True
    )

    try:
        quantizer.fit(df_clean)
        quantized = quantizer.transform(df_clean)

        text_clusters = quantized['text_clusters']
        tabular_bins = quantized['tabular_bins']

        # Compute NMI
        nmi = normalized_mutual_info_score(text_clusters, tabular_bins)

        # Compute Cramér's V
        contingency = pd.crosstab(text_clusters, tabular_bins)
        cramers_v = association(contingency, method='cramer')

        # Predictive accuracy (tabular → text cluster)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        X = df_clean[tabular_cols]

        # Handle categorical columns
        X_encoded = pd.get_dummies(X)

        cv_scores = cross_val_score(clf, X_encoded, text_clusters, cv=3, scoring='f1_weighted')
        predictive_accuracy = cv_scores.mean()

        # Composite score
        correlation_score = 0.5 * nmi + 0.3 * cramers_v + 0.2 * predictive_accuracy

        return {
            "correlation_score": correlation_score,
            "nmi": nmi,
            "cramers_v": cramers_v,
            "predictive_accuracy": predictive_accuracy
        }

    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None


def classify_correlation_type(df: pd.DataFrame, text_col: str, tabular_cols: list, nmi: float):
    """
    Classify type of correlation.

    Returns:
        String describing correlation type
    """
    correlation_types = []

    # Check for continuous columns (semantic conditioning)
    continuous_cols = [c for c in tabular_cols
                      if df[c].dtype in ['float64', 'int64'] and df[c].nunique() > 20]
    if continuous_cols:
        correlation_types.append("semantic_conditioning")

    # Check for categorical columns (logical constraints)
    categorical_cols = [c for c in tabular_cols
                       if df[c].dtype in ['object', 'category'] or df[c].nunique() < 20]
    if categorical_cols and nmi > 0.3:
        correlation_types.append("logical_constraints")

    # Check for demographic columns
    demographic_keywords = ['age', 'gender', 'education', 'income', 'location', 'region']
    if any(keyword in col.lower() for col in tabular_cols for keyword in demographic_keywords):
        correlation_types.append("demographic_alignment")

    return " + ".join(correlation_types) if correlation_types else "weak_correlation"


def discover_datasets(data_dir: str, sample_size: int = 10000, min_nmi: float = 0.0, verbose: bool = False):
    """
    Main discovery pipeline.

    Args:
        data_dir: Path to data directory
        sample_size: Max rows to sample
        min_nmi: Minimum NMI threshold
        verbose: Print verbose output

    Returns:
        Discovery results dict
    """
    results = []
    datasets_scanned = 0

    print(f"Scanning datasets in {data_dir}...")

    for dataset_name, df in scan_datasets(data_dir, sample_size):
        datasets_scanned += 1

        if verbose:
            print(f"\n=== {dataset_name} ===")
            print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

        # Auto-detect columns
        detected = auto_detect_columns(df)

        if not detected['text'] or not detected['tabular']:
            if verbose:
                print(f"Skipping {dataset_name}: no multimodal data detected")
            continue

        text_col = detected['text'][0]  # Use first text column
        tabular_cols = detected['tabular']

        if verbose:
            print(f"Text column: {text_col}")
            print(f"Tabular columns: {tabular_cols}")

        # Compute correlations
        metrics = compute_correlation_metrics(df, text_col, tabular_cols)

        if metrics is None or metrics['nmi'] < min_nmi:
            if verbose:
                print(f"Skipping {dataset_name}: low correlation (NMI={metrics['nmi'] if metrics else 'N/A'})")
            continue

        # Classify correlation type
        correlation_type = classify_correlation_type(df, text_col, tabular_cols, metrics['nmi'])

        # Determine recommendation
        if metrics['correlation_score'] > 0.7:
            recommendation = "STRONG"
        elif metrics['correlation_score'] > 0.5:
            recommendation = "MODERATE"
        else:
            recommendation = "WEAK"

        result = {
            "dataset_name": dataset_name,
            "correlation_score": float(metrics['correlation_score']),
            "metrics": {
                "nmi": float(metrics['nmi']),
                "cramers_v": float(metrics['cramers_v']),
                "predictive_accuracy": float(metrics['predictive_accuracy'])
            },
            "detected_columns": {
                "text": detected['text'],
                "tabular": detected['tabular']
            },
            "correlation_type": correlation_type,
            "recommendation": recommendation,
            "n_samples": len(df)
        }

        results.append(result)

        if verbose:
            print(f"Correlation score: {metrics['correlation_score']:.3f}")
            print(f"Recommendation: {recommendation}")

    # Sort by correlation score
    results.sort(key=lambda x: x['correlation_score'], reverse=True)

    # Add ranks
    for i, result in enumerate(results, 1):
        result['rank'] = i

    # Get top recommendations
    top_recommendations = [r['dataset_name'] for r in results[:5]]

    return {
        "discovery_timestamp": datetime.now().isoformat(),
        "datasets_scanned": datasets_scanned,
        "datasets_with_multimodal": len(results),
        "ranked_results": results,
        "top_recommendations": top_recommendations
    }


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_summary(results: dict):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print("DATASET DISCOVERY SUMMARY")
    print("="*60)
    print(f"Datasets scanned: {results['datasets_scanned']}")
    print(f"Multimodal datasets found: {results['datasets_with_multimodal']}")
    print(f"\nTop Recommendations:")

    for result in results['ranked_results'][:5]:
        print(f"\n{result['rank']}. {result['dataset_name']} - {result['recommendation']}")
        print(f"   Score: {result['correlation_score']:.3f}")
        print(f"   NMI: {result['metrics']['nmi']:.3f}")
        print(f"   Type: {result['correlation_type']}")
        print(f"   Text: {result['detected_columns']['text'][0]}")
        print(f"   Tabular: {', '.join(result['detected_columns']['tabular'][:3])}")


def main():
    parser = argparse.ArgumentParser(description="Discover datasets with strong tabular-text correlations")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--sample-size", type=int, default=10000, help="Max rows to sample per dataset")
    parser.add_argument("--min-nmi", type=float, default=0.0, help="Minimum NMI threshold")
    parser.add_argument("--output", default="../artifacts/discovery_report.json", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run discovery
    results = discover_datasets(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        min_nmi=args.min_nmi,
        verbose=args.verbose
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(results, args.output)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
