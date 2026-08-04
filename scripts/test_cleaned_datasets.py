"""
Test prediction accuracy on cleaned datasets.

Compares raw vs cleaned data to see if accuracy improves.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.quantization.semantic_quantizer import SemanticQuantizer


def test_dataset_accuracy(df, text_column, target_column, dataset_name, k_clusters=None):
    """
    Test cross-modal prediction accuracy for a dataset.

    Args:
        df: DataFrame
        text_column: Name of text column
        target_column: Name of target column
        dataset_name: Dataset name for display
        k_clusters: Number of text clusters (None = adaptive)

    Returns:
        dict: Results including accuracy, NMI, F1
    """
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(df):,}")
    print(f"Text column: {text_column}")
    print(f"Target column: {target_column}")
    print(f"Classes: {df[target_column].nunique()}")

    # Get number of classes
    n_classes = df[target_column].nunique()
    print(f"Class distribution:")
    print(df[target_column].value_counts().head(10))

    # Set K clusters
    if k_clusters is None:
        # Adaptive: K = n_classes * 1.5, bounded by sample size
        k_clusters = min(max(int(n_classes * 1.5), 5), len(df) // 20)
    print(f"\nUsing K={k_clusters} text clusters")

    try:
        # Initialize quantizer
        quantizer = SemanticQuantizer(
            text_columns=[text_column],
            tabular_columns=[target_column],
            adaptive=False,  # Manual K
            text_clusters=k_clusters,
            tabular_bins=n_classes
        )

        # Fit on full data
        print("\nFitting semantic quantizer...")
        quantizer.fit(df)

        # Get text cluster assignments
        text_embeddings = quantizer.text_clusterer.get_embeddings([text_column], df)
        text_clusters = quantizer.text_clusterer.clusterer.predict(text_embeddings)

        # Encode target labels
        le = LabelEncoder()
        target_labels = le.fit_transform(df[target_column])

        # Train classifier: text_cluster -> target_label
        print("Training Text-to-Attribute classifier...")
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(text_clusters.reshape(-1, 1), target_labels)

        # Predict
        predictions = clf.predict(text_clusters.reshape(-1, 1))

        # Calculate metrics
        accuracy = accuracy_score(target_labels, predictions)
        f1 = f1_score(target_labels, predictions, average='weighted')
        nmi = normalized_mutual_info_score(target_labels, text_clusters)

        print(f"\n{'='*60}")
        print(f"RESULTS: {dataset_name}")
        print(f"{'='*60}")
        print(f"✅ Prediction Accuracy: {accuracy*100:.1f}%")
        print(f"   F1-Score (weighted): {f1:.3f}")
        print(f"   NMI: {nmi:.3f}")
        print(f"   K clusters: {k_clusters}")
        print(f"   Classes: {n_classes}")

        return {
            'dataset': dataset_name,
            'samples': len(df),
            'classes': n_classes,
            'k_clusters': k_clusters,
            'accuracy': accuracy,
            'f1_score': f1,
            'nmi': nmi,
            'status': '✅' if accuracy >= 0.5 else '❌'
        }

    except Exception as e:
        print(f"\n❌ Error testing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'samples': len(df),
            'accuracy': 0.0,
            'f1_score': 0.0,
            'nmi': 0.0,
            'status': '❌ Error'
        }


def main():
    """Test cleaned datasets."""
    print("="*60)
    print("TESTING CLEANED DATASETS")
    print("="*60)

    staging_dir = Path(__file__).parent.parent.parent / "Data" / "Staging"
    results = []

    # 1. Test Kiva Loans (cleaned)
    print("\n" + "="*60)
    print("1. KIVA LOANS (CLEANED)")
    print("="*60)

    kiva_path = staging_dir / "kiva_loans_clean.csv"
    if kiva_path.exists():
        df_kiva = pd.read_csv(kiva_path)
        result = test_dataset_accuracy(
            df_kiva,
            text_column='use',
            target_column='sector',
            dataset_name='Kiva Loans (cleaned)',
            k_clusters=15  # Match number of sectors
        )
        results.append(result)
    else:
        print(f"⚠️ Kiva cleaned data not found: {kiva_path}")

    # 2. Test Fake Job Postings (cleaned)
    print("\n" + "="*60)
    print("2. FAKE JOB POSTINGS (CLEANED)")
    print("="*60)

    fake_jobs_path = staging_dir / "fake_jobs_clean.csv"
    if fake_jobs_path.exists():
        df_fake = pd.read_csv(fake_jobs_path)
        result = test_dataset_accuracy(
            df_fake,
            text_column='description',
            target_column='fraudulent',
            dataset_name='Fake Jobs (cleaned)',
            k_clusters=2  # Binary classification
        )
        results.append(result)
    else:
        print(f"⚠️ Fake Jobs cleaned data not found: {fake_jobs_path}")

    # 3. Test Amazon Reviews (cleaned)
    print("\n" + "="*60)
    print("3. AMAZON REVIEWS (CLEANED)")
    print("="*60)

    amazon_path = staging_dir / "amazon_reviews_clean.csv"
    if amazon_path.exists():
        df_amazon = pd.read_csv(amazon_path)
        result = test_dataset_accuracy(
            df_amazon,
            text_column='text',
            target_column='rating',
            dataset_name='Amazon Reviews (cleaned)',
            k_clusters=5  # 5 star ratings
        )
        results.append(result)
    else:
        print(f"⚠️ Amazon cleaned data not found: {amazon_path}")

    # Summary Table
    print("\n" + "="*60)
    print("SUMMARY: CLEANED DATASETS")
    print("="*60)
    print(f"{'Dataset':<30} {'Samples':<10} {'Accuracy':<12} {'Status':<8}")
    print("-"*60)

    for r in results:
        print(f"{r['dataset']:<30} {r['samples']:<10,} {r['accuracy']*100:>10.1f}% {r['status']:<8}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("✅ Accuracy >= 50%: Suitable for experiments")
    print("❌ Accuracy < 50%: Not suitable (too weak correlation)")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = staging_dir / "cleaned_dataset_accuracy.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n📊 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
