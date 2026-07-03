#!/usr/bin/env python3
"""
Traditional Isolated Metrics Evaluation — demonstrating the Stitching Fallacy.

Uses industry-standard tools that evaluate modalities independently:
  1. SDV (Synthetic Data Vault) — tabular data quality
       Kolmogorov-Smirnov (KS) Complement
       Total Variation (TV) Complement
  2. BERTScore — text quality

Expected finding: Tilted Data scores highly on all these metrics because its
marginal distributions are perfect, exposing the blind spot of isolated evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_column import KSComplement, TVComplement
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Importing libraries...")
print("  SDV metrics")
print("  BERTScore")
print("  SBERT")

BASE_DIR = Path(__file__).parent
EXPERIMENT_DIR = BASE_DIR / "experiments" / "baselines_filtered_20260428_195011"
SYNTHETIC_DATA_DIR = EXPERIMENT_DIR / "synthetic_data"
OUTPUT_DIR = EXPERIMENT_DIR / "traditional_metrics"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    'amazon_reviews': {
        'original': 'amazon_reviews_original.csv',
        'text_col': 'text',
        'tabular_cols': ['rating'],
        'baselines': [
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
        'baselines': [
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
        'baselines': [
            'fake_jobs_original.csv',
            'fake_jobs_baseline1_independent.csv',
            'fake_jobs_baseline2_sequential.csv',
            'fake_jobs_baseline3_joint_llm.csv',
            'fake_jobs_baseline4_tabsyn.csv',
            'fake_jobs_tilted.csv'
        ]
    }
}


def evaluate_tabular_sdv(real_df, synth_df, tabular_cols):
    """
    Evaluate tabular data quality with SDV metrics.

    Returns KS Complement and TV Complement for each column (higher is better).
    """
    results = {}

    for col in tabular_cols:
        real_col = real_df[col]
        synth_col = synth_df[col]

        if real_col.dtype == 'object' or synth_col.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder

            if real_col.dtype == 'object' and synth_col.dtype in ['int64', 'int32', 'float64', 'float32']:
                le = LabelEncoder()
                real_col_encoded = le.fit_transform(real_col)
                synth_col_encoded = synth_col.astype(int)
            elif real_col.dtype in ['int64', 'int32', 'float64', 'float32'] and synth_col.dtype == 'object':
                le = LabelEncoder()
                synth_col_encoded = le.fit_transform(synth_col)
                real_col_encoded = real_col.astype(int)
            else:
                all_categories = pd.concat([real_col, synth_col]).unique()
                category_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
                real_col_encoded = real_col.map(category_mapping)
                synth_col_encoded = synth_col.map(category_mapping)
        else:
            real_col_encoded = real_col
            synth_col_encoded = synth_col

        try:
            ks_score = KSComplement.compute(
                real_data=real_col_encoded,
                synthetic_data=synth_col_encoded
            )
            results[f'{col}_KS'] = ks_score
        except Exception:
            results[f'{col}_KS'] = None

        try:
            tv_score = TVComplement.compute(
                real_data=real_col,
                synthetic_data=synth_col
            )
            results[f'{col}_TV'] = tv_score
        except Exception:
            results[f'{col}_TV'] = None

    ks_scores = [v for k, v in results.items() if 'KS' in k and v is not None]
    tv_scores = [v for k, v in results.items() if 'TV' in k and v is not None]
    results['avg_KS'] = np.mean(ks_scores) if ks_scores else None
    results['avg_TV'] = np.mean(tv_scores) if tv_scores else None

    return results


def evaluate_embeddings_similarity(real_embeddings, synth_embeddings, sample_size=500):
    """
    Compare embedding vectors directly (used for baseline4 which stores SBERT embeddings).

    Returns precision, recall, and F1 analogues based on maximum cosine similarity.
    """
    if len(synth_embeddings) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(synth_embeddings), sample_size, replace=False)
        synth_sample = synth_embeddings[indices]
    else:
        synth_sample = synth_embeddings

    if len(real_embeddings) > len(synth_sample):
        np.random.seed(42)
        indices = np.random.choice(len(real_embeddings), len(synth_sample), replace=False)
        real_sample = real_embeddings[indices]
    else:
        real_sample = real_embeddings

    similarities = cosine_similarity(synth_sample, real_sample)
    max_similarities = similarities.max(axis=1)
    avg_similarity = max_similarities.mean()

    return {
        'precision': avg_similarity,
        'recall': avg_similarity,
        'f1': avg_similarity
    }


def evaluate_text_bertscore(real_texts, synth_texts, sample_size=500):
    """
    Evaluate text quality with BERTScore (precision, recall, F1).

    Samples up to `sample_size` texts to keep runtime manageable.
    """
    real_texts = [str(t) for t in real_texts if pd.notna(t)]
    synth_texts = [str(t) for t in synth_texts if pd.notna(t)]

    if len(synth_texts) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(synth_texts), sample_size, replace=False)
        synth_texts_sample = [synth_texts[i] for i in indices]
    else:
        synth_texts_sample = synth_texts

    if len(real_texts) > len(synth_texts_sample):
        np.random.seed(42)
        real_texts_sample = list(np.random.choice(real_texts, len(synth_texts_sample), replace=False))
    else:
        real_texts_sample = real_texts

    print(f"    Computing BERTScore ({len(synth_texts_sample)} texts)...")

    P, R, F1 = bert_score(
        synth_texts_sample,
        real_texts_sample,
        lang='en',
        model_type='roberta-large',
        device='cpu',
        batch_size=16,
        verbose=False
    )

    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


def evaluate_dataset(dataset_name, config):
    """Evaluate all baselines for a single dataset."""

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.replace('_', ' ').title()}")
    print(f"{'='*80}")

    real_file = SYNTHETIC_DATA_DIR / config['original']
    real_df = pd.read_csv(real_file)
    print(f"  Loaded real data: {len(real_df)} rows")

    text_col = config['text_col']
    tabular_cols = config['tabular_cols']
    results = []

    for baseline_file in config['baselines']:
        baseline_name = baseline_file.replace('.csv', '').replace(f'{dataset_name}_', '')
        print(f"\n--- {baseline_name} ---")

        synth_file = SYNTHETIC_DATA_DIR / baseline_file
        if not synth_file.exists():
            print(f"  File not found, skipping.")
            continue

        synth_df = pd.read_csv(synth_file)
        print(f"  Loaded: {len(synth_df)} rows")

        result = {
            'dataset': dataset_name,
            'baseline': baseline_name,
            'synth_rows': len(synth_df)
        }

        # Tabular evaluation (SDV)
        print(f"  [1/2] Tabular (SDV)...")
        tabular_results = evaluate_tabular_sdv(real_df, synth_df, tabular_cols)
        result.update({f'tabular_{k}': v for k, v in tabular_results.items()})

        if tabular_results['avg_KS'] is not None:
            print(f"    KS Complement: {tabular_results['avg_KS']:.4f}")
        if tabular_results['avg_TV'] is not None:
            print(f"    TV Complement: {tabular_results['avg_TV']:.4f}")

        # Text / embedding evaluation
        print(f"  [2/2] Text / embedding...")

        if text_col in synth_df.columns:
            try:
                text_results = evaluate_text_bertscore(
                    real_df[text_col].tolist(),
                    synth_df[text_col].tolist()
                )
                result.update({f'text_{k}': v for k, v in text_results.items()})
                print(f"    BERTScore F1: {text_results['f1']:.4f}")
                print(f"    Precision:    {text_results['precision']:.4f}")
                print(f"    Recall:       {text_results['recall']:.4f}")
            except Exception as e:
                print(f"    BERTScore failed: {e}")
                result.update({'text_precision': None, 'text_recall': None, 'text_f1': None})
        else:
            # Baseline4 stores SBERT embeddings instead of raw text
            print(f"    Using embedding similarity (baseline4 format)...")
            try:
                sbert_cols = [c for c in synth_df.columns if c.startswith('sbert_')]
                synth_embeddings = synth_df[sbert_cols].values

                print(f"    Encoding real text with SBERT...")
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                real_texts = [str(t) for t in real_df[text_col] if pd.notna(t)]
                real_embeddings = sbert_model.encode(real_texts, show_progress_bar=False)

                emb_results = evaluate_embeddings_similarity(real_embeddings, synth_embeddings)
                result.update({f'text_{k}': v for k, v in emb_results.items()})
                print(f"    Embedding similarity F1: {emb_results['f1']:.4f}")
            except Exception as e:
                print(f"    Embedding similarity failed: {e}")
                result.update({'text_precision': None, 'text_recall': None, 'text_f1': None})

        results.append(result)

    return results


def main():
    print("="*80)
    print("Traditional Isolated Metrics Evaluation")
    print("="*80)
    print("\nMetrics:")
    print("  1. SDV (Synthetic Data Vault) — KS Complement, TV Complement")
    print("  2. BERTScore — text semantic similarity")
    print("\nHypothesis:")
    print("  Tilted Data scores highly (proves isolated metrics miss cross-modal logic)")
    print("  Baseline 1 may also score highly (independent generation, no joint constraint)")

    all_results = []

    for dataset_name, config in DATASETS.items():
        dataset_results = evaluate_dataset(dataset_name, config)
        all_results.extend(dataset_results)

    results_df = pd.DataFrame(all_results)

    csv_file = OUTPUT_DIR / "traditional_metrics_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n  Saved CSV: {csv_file}")

    json_file = OUTPUT_DIR / "traditional_metrics_results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved JSON: {json_file}")

    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")

    for dataset_name in DATASETS.keys():
        dataset_results = results_df[results_df['dataset'] == dataset_name]
        if len(dataset_results) == 0:
            continue

        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        print(f"{'Baseline':<25} {'KS':<8} {'TV':<8} {'BERTScore F1':<15}")
        print("-" * 60)

        for _, row in dataset_results.iterrows():
            baseline = row['baseline']
            ks = f"{row['tabular_avg_KS']:.4f}" if pd.notna(row['tabular_avg_KS']) else "N/A"
            tv = f"{row['tabular_avg_TV']:.4f}" if pd.notna(row['tabular_avg_TV']) else "N/A"
            bert = f"{row['text_f1']:.4f}" if pd.notna(row['text_f1']) else "N/A"
            marker = " *" if 'tilted' in baseline else "  "
            print(f"{baseline:<25} {ks:<8} {tv:<8} {bert:<15} {marker}")

    print(f"\n{'='*80}")
    print("Key Findings")
    print(f"{'='*80}")

    tilted_results = results_df[results_df['baseline'].str.contains('tilted')]

    if len(tilted_results) > 0:
        avg_ks = tilted_results['tabular_avg_KS'].mean()
        avg_tv = tilted_results['tabular_avg_TV'].mean()
        avg_bert = tilted_results['text_f1'].mean()

        print(f"\nTilted Data average scores:")
        print(f"  KS Complement:  {avg_ks:.4f}  (higher is better)")
        print(f"  TV Complement:  {avg_tv:.4f}  (higher is better)")
        print(f"  BERTScore F1:   {avg_bert:.4f}  (higher is better)")

        if avg_ks > 0.8 or avg_tv > 0.8 or avg_bert > 0.7:
            print("\nConclusion: Traditional metrics assign high scores to Tilted Data,")
            print("confirming they cannot detect cross-modal logic corruption.")
            print("This validates the need for joint evaluation (SynEval).")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
