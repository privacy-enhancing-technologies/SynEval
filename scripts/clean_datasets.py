"""
Clean datasets for multimodal synthetic data experiments.

This script implements the cleaning procedures from data_quality_analysis.md
to prepare datasets for experiments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def clean_kiva_loans(input_path: str, output_path: str, sample_size: int = 10000):
    """
    Clean Kiva Loans dataset.

    Cleaning steps:
    1. Select key columns and remove NaN
    2. Clean text (strip, filter short text)
    3. Filter rare categories (< 50 samples)
    4. Remove outliers in loan_amount
    5. Sample to desired size

    Args:
        input_path: Path to raw kiva_loans.csv
        output_path: Path to save cleaned data
        sample_size: Final sample size (None = no sampling)
    """
    print("=" * 60)
    print("CLEANING KIVA LOANS DATASET")
    print("=" * 60)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original size: {len(df):,} rows")

    # 1. Select key columns
    key_cols = ['use', 'sector', 'loan_amount', 'term_in_months']
    df_clean = df[key_cols].copy()

    # 2. Remove NaN
    df_clean = df_clean.dropna()
    print(f"After removing NaN: {len(df_clean):,} rows")

    # 3. Clean text
    df_clean['use'] = df_clean['use'].str.strip()
    df_clean = df_clean[df_clean['use'].str.len() >= 15]  # Filter very short text
    print(f"After text cleaning (len >= 15): {len(df_clean):,} rows")

    # 4. Filter rare categories
    sector_counts = df_clean['sector'].value_counts()
    print(f"\nSector distribution BEFORE filtering:")
    print(sector_counts)

    min_samples = 50 if len(df_clean) > 10000 else 20
    valid_sectors = sector_counts[sector_counts >= min_samples].index
    df_clean = df_clean[df_clean['sector'].isin(valid_sectors)]
    print(f"\nAfter filtering rare sectors (< {min_samples}): {len(df_clean):,} rows")
    print(f"Remaining sectors: {len(valid_sectors)}")

    # 5. Remove outliers in loan_amount
    q1, q99 = df_clean['loan_amount'].quantile([0.01, 0.99])
    df_clean = df_clean[(df_clean['loan_amount'] >= q1) &
                        (df_clean['loan_amount'] <= q99)]
    print(f"After outlier filtering (1%-99%): {len(df_clean):,} rows")

    # 6. Sample if needed
    if sample_size and len(df_clean) > sample_size:
        df_final = df_clean.sample(sample_size, random_state=42)
        print(f"Sampled to: {len(df_final):,} rows")
    else:
        df_final = df_clean
        print(f"Using all {len(df_final):,} rows (no sampling needed)")

    # 7. Save
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")

    # 8. Statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df_final):,}")
    print(f"Sectors: {len(df_final['sector'].unique())}")
    print(f"Text length: {df_final['use'].str.len().mean():.0f} chars (avg)")
    print(f"Text length range: {df_final['use'].str.len().min()}-{df_final['use'].str.len().max()}")
    print(f"Loan amount: ${df_final['loan_amount'].mean():.2f} ± ${df_final['loan_amount'].std():.2f}")
    print(f"\nTop sectors:")
    print(df_final['sector'].value_counts().head(10))

    return df_final


def clean_fake_jobs(input_path: str, output_path: str, balance_ratio: float = 3.0):
    """
    Clean Fake Job Postings dataset.

    Cleaning steps:
    1. Select key columns and remove NaN
    2. Clean text (strip, filter short descriptions)
    3. Balance sampling (fraudulent : non-fraudulent = 1:balance_ratio)
    4. Shuffle

    Args:
        input_path: Path to raw Fake Job Postings.csv
        output_path: Path to save cleaned data
        balance_ratio: Ratio of non-fraudulent to fraudulent (default: 3.0)
    """
    print("\n" + "=" * 60)
    print("CLEANING FAKE JOB POSTINGS DATASET")
    print("=" * 60)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original size: {len(df):,} rows")

    # 1. Select key columns
    key_cols = ['description', 'fraudulent', 'employment_type',
                'has_company_logo', 'required_experience']
    df_clean = df[key_cols].copy()

    # 2. Remove NaN (at least description and fraudulent must exist)
    df_clean = df_clean.dropna(subset=['description', 'fraudulent'])
    print(f"After removing NaN: {len(df_clean):,} rows")

    # 3. Clean text
    df_clean['description'] = df_clean['description'].str.strip()
    df_clean = df_clean[df_clean['description'].str.len() >= 50]  # Filter very short descriptions
    print(f"After text cleaning (len >= 50): {len(df_clean):,} rows")

    # Check class distribution
    fraud_counts = df_clean['fraudulent'].value_counts()
    print(f"\nClass distribution BEFORE balancing:")
    print(f"  Non-fraudulent (0): {fraud_counts.get(0, 0):,} ({fraud_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
    print(f"  Fraudulent (1): {fraud_counts.get(1, 0):,} ({fraud_counts.get(1, 0)/len(df_clean)*100:.1f}%)")

    # 4. Balance sampling
    fraud = df_clean[df_clean['fraudulent'] == 1]
    non_fraud = df_clean[df_clean['fraudulent'] == 0]

    n_fraud = len(fraud)
    n_non_fraud_target = int(n_fraud * balance_ratio)

    if len(non_fraud) > n_non_fraud_target:
        non_fraud_sampled = non_fraud.sample(n_non_fraud_target, random_state=42)
    else:
        non_fraud_sampled = non_fraud
        print(f"⚠️ Warning: Not enough non-fraudulent samples. Using all {len(non_fraud)} samples.")

    df_balanced = pd.concat([fraud, non_fraud_sampled])
    df_final = df_balanced.sample(frac=1, random_state=42)  # Shuffle

    print(f"\nAfter balancing (ratio 1:{balance_ratio}):")
    print(f"  Total samples: {len(df_final):,}")
    print(f"  Fraudulent: {len(fraud):,} ({len(fraud)/len(df_final)*100:.1f}%)")
    print(f"  Non-fraudulent: {len(non_fraud_sampled):,} ({len(non_fraud_sampled)/len(df_final)*100:.1f}%)")

    # 5. Save
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")

    # 6. Statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df_final):,}")
    print(f"Fraud ratio: {df_final['fraudulent'].mean():.1%}")
    print(f"Description length: {df_final['description'].str.len().mean():.0f} chars (avg)")
    print(f"Description length range: {df_final['description'].str.len().min()}-{df_final['description'].str.len().max()}")

    if 'has_company_logo' in df_final.columns:
        print(f"Has company logo: {df_final['has_company_logo'].mean():.1%}")

    return df_final


def clean_amazon_reviews(input_path: str, output_path: str, sample_size: int = 10000):
    """
    Clean Amazon Reviews dataset.

    Cleaning steps:
    1. Remove NaN
    2. Clean text (strip, filter very short reviews)
    3. Balance sampling across ratings
    4. Sample to desired size

    Args:
        input_path: Path to raw Amazon reviews CSV
        output_path: Path to save cleaned data
        sample_size: Final sample size
    """
    print("\n" + "=" * 60)
    print("CLEANING AMAZON REVIEWS DATASET")
    print("=" * 60)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original size: {len(df):,} rows")

    # Detect column names (may vary)
    text_col = 'review/text' if 'review/text' in df.columns else 'text'
    rating_col = 'review/score' if 'review/score' in df.columns else 'rating'

    # 1. Select key columns
    key_cols = [text_col, rating_col]
    if 'review/helpfulness' in df.columns:
        key_cols.append('review/helpfulness')
    elif 'helpful_vote' in df.columns:
        key_cols.append('helpful_vote')

    df_clean = df[key_cols].copy()

    # Rename for consistency
    df_clean = df_clean.rename(columns={
        text_col: 'text',
        rating_col: 'rating'
    })

    # 2. Remove NaN
    df_clean = df_clean.dropna(subset=['text', 'rating'])
    print(f"After removing NaN: {len(df_clean):,} rows")

    # 3. Clean text
    df_clean['text'] = df_clean['text'].str.strip()
    df_clean = df_clean[df_clean['text'].str.len() >= 10]  # Filter very short reviews
    print(f"After text cleaning (len >= 10): {len(df_clean):,} rows")

    # Check rating distribution
    rating_counts = df_clean['rating'].value_counts().sort_index()
    print(f"\nRating distribution BEFORE balancing:")
    for rating, count in rating_counts.items():
        print(f"  {rating}★: {count:,} ({count/len(df_clean)*100:.1f}%)")

    # 4. Balance sampling (stratified by rating)
    if sample_size and len(df_clean) > sample_size:
        # Stratified sampling
        df_final = df_clean.groupby('rating', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 5), random_state=42)
        )
        print(f"\nAfter stratified sampling: {len(df_final):,} rows")
    else:
        df_final = df_clean
        print(f"\nUsing all {len(df_final):,} rows (no sampling needed)")

    # 5. Save
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")

    # 6. Statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df_final):,}")
    print(f"Ratings: {sorted(df_final['rating'].unique())}")
    print(f"Text length: {df_final['text'].str.len().mean():.0f} chars (avg)")

    print(f"\nRating distribution AFTER balancing:")
    rating_counts_final = df_final['rating'].value_counts().sort_index()
    for rating, count in rating_counts_final.items():
        print(f"  {rating}★: {count:,} ({count/len(df_final)*100:.1f}%)")

    return df_final


def main():
    """Main cleaning workflow."""
    # Define paths
    data_root = Path(__file__).parent.parent.parent / "Data"
    staging_dir = data_root / "Staging"
    staging_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("DATASET CLEANING WORKFLOW")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Staging directory: {staging_dir}")
    print()

    results = {}

    # 1. Clean Kiva Loans
    kiva_input = data_root / "Kiva Crowdfunding" / "kiva_loans.csv"
    kiva_output = staging_dir / "kiva_loans_clean.csv"

    if kiva_input.exists():
        try:
            df_kiva = clean_kiva_loans(
                str(kiva_input),
                str(kiva_output),
                sample_size=10000
            )
            results['kiva'] = len(df_kiva)
        except Exception as e:
            print(f"❌ Error cleaning Kiva: {e}")
            results['kiva'] = None
    else:
        print(f"⚠️ Skipping Kiva Loans (file not found: {kiva_input})")
        results['kiva'] = None

    # 2. Clean Fake Job Postings
    fake_jobs_input = data_root / "Fake Job Postings.csv"
    fake_jobs_output = staging_dir / "fake_jobs_clean.csv"

    if fake_jobs_input.exists():
        try:
            df_fake = clean_fake_jobs(
                str(fake_jobs_input),
                str(fake_jobs_output),
                balance_ratio=3.0
            )
            results['fake_jobs'] = len(df_fake)
        except Exception as e:
            print(f"❌ Error cleaning Fake Jobs: {e}")
            results['fake_jobs'] = None
    else:
        print(f"⚠️ Skipping Fake Jobs (file not found: {fake_jobs_input})")
        results['fake_jobs'] = None

    # 3. Clean Amazon Reviews (real_10k)
    amazon_input = data_root / "Amazon" / "real_10k.csv"
    amazon_output = staging_dir / "amazon_reviews_clean.csv"

    if amazon_input.exists():
        try:
            df_amazon = clean_amazon_reviews(
                str(amazon_input),
                str(amazon_output),
                sample_size=10000
            )
            results['amazon'] = len(df_amazon)
        except Exception as e:
            print(f"❌ Error cleaning Amazon: {e}")
            results['amazon'] = None
    else:
        print(f"⚠️ Skipping Amazon Reviews (file not found: {amazon_input})")
        results['amazon'] = None

    # Summary
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    for dataset, count in results.items():
        if count:
            print(f"✅ {dataset.upper()}: {count:,} samples")
        else:
            print(f"❌ {dataset.upper()}: Failed or skipped")

    print(f"\nCleaned datasets saved to: {staging_dir}")


if __name__ == '__main__':
    main()
