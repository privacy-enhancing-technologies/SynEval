"""
Complete Experimental Pipeline for Multimodal Synthetic Data Generation

This script runs the full experimental pipeline:
1. Generate synthetic datasets using 4 methods on 3 datasets (12 total)
2. Evaluate all synthetic datasets on 4 metrics
3. Generate comparison tables and visualizations
4. Save comprehensive results report

Methods:
- CTGAN+LLM Stitcher (baseline - demonstrates stitching fallacy)
- Prompt-LLM (proposed method)
- Multimodal Diffusion (proposed method)
- Tilted Data (negative control)

Datasets:
- Kiva Loans (10K samples)
- Fake Job Postings (3.4K samples)
- Amazon Reviews (9.6K samples)

Metrics:
- Fidelity: Jensen-Shannon Divergence (JSD)
- Utility: Text-to-Attribute (T2A) and Attribute-to-Text (A2T) accuracy
- Diversity: Joint Entropy
- Privacy: Distance to Closest Record (DCR)
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
syngen_dir = Path(__file__).parent
syneval_dir = syngen_dir.parent / "SynEval"
sys.path.insert(0, str(syngen_dir))
sys.path.insert(0, str(syneval_dir))

# Import generators
from generators.ctgan_llm_stitcher import CTGANLLMStitcher
from generators.prompt_llm import PromptLLMGenerator
from generators.multimodal_diffusion import MultimodalDiffusionGenerator
from generators.tilted import TiltedGenerator

# Import unified evaluator from SynEval
from evaluation.evaluator import MultimodalEvaluator


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_step(step_num, total_steps, description):
    """Print a step indicator."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 80)


def load_datasets():
    """Load all cleaned datasets.

    Looks first in SynGen/data/ for pre-cleaned CSVs. If that directory does
    not exist, falls back to the pre-generated experiment data directory where
    *_original.csv files serve as the real-data reference.
    """
    print_step("PREP", "PREP", "Loading cleaned datasets")

    syngen_dir = Path(__file__).parent
    data_dir = syngen_dir / "data"

    # Fallback: use the *_original.csv files from the pre-generated experiment
    fallback_dir = (
        syngen_dir
        / "experiments"
        / "baselines_filtered_20260428_195011"
        / "synthetic_data"
    )

    def find_csv(primary: Path, fallback: Path) -> Path | None:
        if primary.exists():
            return primary
        if fallback.exists():
            return fallback
        return None

    datasets = {}

    # Kiva Loans
    kiva_path = find_csv(
        data_dir / "kiva_loans_clean.csv",
        fallback_dir / "kiva_loans_original.csv",
    )
    if kiva_path:
        datasets['kiva'] = {
            'df': pd.read_csv(kiva_path),
            'text_cols': ['use'],
            'tabular_cols': ['sector', 'loan_amount', 'term_in_months'],
            'name': 'Kiva Loans',
            'samples': 1000,
        }
        print(f"✓ Loaded Kiva Loans: {len(datasets['kiva']['df'])} samples ({kiva_path.name})")

    # Fake Job Postings
    fake_jobs_path = find_csv(
        data_dir / "fake_jobs_clean.csv",
        fallback_dir / "fake_jobs_original.csv",
    )
    if fake_jobs_path:
        datasets['fake_jobs'] = {
            'df': pd.read_csv(fake_jobs_path),
            'text_cols': ['description'],
            'tabular_cols': ['fraudulent', 'has_company_logo'],
            'name': 'Fake Job Postings',
            'samples': 500,
        }
        print(f"✓ Loaded Fake Jobs: {len(datasets['fake_jobs']['df'])} samples ({fake_jobs_path.name})")

    # Amazon Reviews
    amazon_path = find_csv(
        data_dir / "amazon_reviews_clean.csv",
        fallback_dir / "amazon_reviews_original.csv",
    )
    if amazon_path:
        datasets['amazon'] = {
            'df': pd.read_csv(amazon_path),
            'text_cols': ['text'],
            'tabular_cols': ['rating'],
            'name': 'Amazon Reviews',
            'samples': 1000,
        }
        print(f"✓ Loaded Amazon Reviews: {len(datasets['amazon']['df'])} samples ({amazon_path.name})")

    return datasets


def generate_synthetic_data(datasets, output_dir):
    """
    Generate synthetic datasets using all 4 methods.

    Returns:
        dict: Mapping of (dataset_name, method_name) -> synthetic_df
    """
    print_banner("PHASE 1: SYNTHETIC DATA GENERATION")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_tasks = len(datasets) * 4
    current_task = 0

    for dataset_key, dataset_info in datasets.items():
        real_df = dataset_info['df']
        text_cols = dataset_info['text_cols']
        tabular_cols = dataset_info['tabular_cols']
        n_samples = dataset_info['samples']
        dataset_name = dataset_info['name']

        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name} ({dataset_key})")
        print(f"{'='*80}")

        # Method 1: CTGAN+LLM Stitcher (Stitching Fallacy)
        current_task += 1
        print(f"\n[{current_task}/{total_tasks}] CTGAN+LLM Stitcher")
        try:
            start_time = time.time()
            generator = CTGANLLMStitcher(
                provider='openai',
                model='gpt-4o-mini',
                n_few_shot=3,
                random_seed=42
            )
            generator.fit(real_df, text_cols, tabular_cols)
            synth_df = generator.generate(n_samples)

            # Save
            output_path = output_dir / f"{dataset_key}_ctgan_llm.csv"
            synth_df.to_csv(output_path, index=False)

            elapsed = time.time() - start_time
            print(f"✓ Generated {len(synth_df)} samples in {elapsed:.1f}s")
            print(f"  Saved to: {output_path}")

            results[(dataset_key, 'ctgan_llm')] = synth_df

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        # Method 2: Prompt-LLM (Proposed)
        current_task += 1
        print(f"\n[{current_task}/{total_tasks}] Prompt-LLM")
        try:
            start_time = time.time()
            generator = PromptLLMGenerator(
                provider='openai',
                model='gpt-4o-mini',
                temperature=0.8,
                batch_size=10,
                random_seed=42
            )
            generator.fit(real_df, text_cols, tabular_cols)
            synth_df = generator.generate(n_samples)

            # Save
            output_path = output_dir / f"{dataset_key}_prompt_llm.csv"
            synth_df.to_csv(output_path, index=False)

            elapsed = time.time() - start_time
            print(f"✓ Generated {len(synth_df)} samples in {elapsed:.1f}s")
            print(f"  Saved to: {output_path}")

            results[(dataset_key, 'prompt_llm')] = synth_df

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        # Method 3: Multimodal Diffusion (Proposed)
        current_task += 1
        print(f"\n[{current_task}/{total_tasks}] Multimodal Diffusion")
        try:
            start_time = time.time()
            generator = MultimodalDiffusionGenerator(
                latent_dim=128,  # Fixed parameter name
                random_seed=42
            )
            generator.fit(real_df, text_cols, tabular_cols)
            synth_df = generator.generate(n_samples)

            # Save
            output_path = output_dir / f"{dataset_key}_diffusion.csv"
            synth_df.to_csv(output_path, index=False)

            elapsed = time.time() - start_time
            print(f"✓ Generated {len(synth_df)} samples in {elapsed:.1f}s")
            print(f"  Saved to: {output_path}")

            results[(dataset_key, 'diffusion')] = synth_df

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        # Method 4: Tilted Data (Negative Control)
        current_task += 1
        print(f"\n[{current_task}/{total_tasks}] Tilted Data")
        try:
            start_time = time.time()
            generator = TiltedGenerator(
                shuffle_strategy='random',
                random_state=42  # Fixed parameter name
            )
            generator.fit(real_df, text_cols, tabular_cols)
            synth_df = generator.generate(n_samples)

            # Save
            output_path = output_dir / f"{dataset_key}_tilted.csv"
            synth_df.to_csv(output_path, index=False)

            elapsed = time.time() - start_time
            print(f"✓ Generated {len(synth_df)} samples in {elapsed:.1f}s")
            print(f"  Saved to: {output_path}")

            results[(dataset_key, 'tilted')] = synth_df

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def evaluate_synthetic_data(datasets, synthetic_results, output_dir):
    """
    Evaluate all synthetic datasets on 4 metrics using MultimodalEvaluator.

    Returns:
        pd.DataFrame: Results table with all metrics
    """
    print_banner("PHASE 2: EVALUATION")

    output_dir = Path(output_dir)

    all_results = []

    for dataset_key, dataset_info in datasets.items():
        real_df = dataset_info['df']
        text_cols = dataset_info['text_cols']
        tabular_cols = dataset_info['tabular_cols']
        dataset_name = dataset_info['name']

        print(f"\n{'='*80}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*80}")

        # Fit one evaluator per dataset (shared across methods)
        print("  Fitting evaluator on real data...")
        evaluator = MultimodalEvaluator(
            text_columns=text_cols,
            tabular_columns=tabular_cols,
            adaptive=True,
            random_seed=42,
        )
        evaluator.fit(real_df)

        for method in ['ctgan_llm', 'prompt_llm', 'diffusion', 'tilted']:
            key = (dataset_key, method)

            if key not in synthetic_results:
                print(f"⚠ Skipping {method} - not generated")
                continue

            synth_df = synthetic_results[key]

            print(f"\nMethod: {method}")
            print("-" * 40)

            try:
                raw = evaluator.evaluate(synth_df)

                metrics = {
                    'dataset': dataset_name,
                    'dataset_key': dataset_key,
                    'method': method,
                    'jsd': raw.get('jsd', np.nan),
                    't2a_accuracy': raw.get('t2a_accuracy', np.nan),
                    'a2t_accuracy': raw.get('a2t_accuracy', np.nan),
                    'joint_entropy': raw.get('joint_entropy', np.nan),
                    'dcr_mean': raw.get('dcr_mean', np.nan),
                    'dcr_min': raw.get('dcr_min', np.nan),
                    'dcr_5th_percentile': raw.get('dcr_5th_percentile', np.nan),
                }

                print(f"  JSD:     {metrics['jsd']:.4f}" if not np.isnan(metrics['jsd']) else "  JSD:     N/A")
                print(f"  T2A:     {metrics['t2a_accuracy']:.4f}" if not np.isnan(metrics['t2a_accuracy']) else "  T2A:     N/A")
                print(f"  A2T:     {metrics['a2t_accuracy']:.4f}" if not np.isnan(metrics['a2t_accuracy']) else "  A2T:     N/A")
                print(f"  Entropy: {metrics['joint_entropy']:.4f}" if not np.isnan(metrics['joint_entropy']) else "  Entropy: N/A")
                print(f"  DCR:     {metrics['dcr_mean']:.4f}" if not np.isnan(metrics['dcr_mean']) else "  DCR:     N/A")

                all_results.append(metrics)

            except Exception as e:
                print(f"✗ Evaluation error: {e}")
                import traceback
                traceback.print_exc()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save to CSV
    results_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved evaluation results to: {results_path}")

    return results_df


def generate_visualizations(results_df, output_dir):
    """Generate comparison tables and visualizations."""
    print_banner("PHASE 3: VISUALIZATION & ANALYSIS")

    output_dir = Path(output_dir)

    # Create summary report
    report_path = output_dir / "experiment_report.md"

    with open(report_path, 'w') as f:
        f.write("# Multimodal Synthetic Data Generation - Experimental Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Overall summary
        f.write("## Summary\n\n")
        f.write(f"- **Datasets**: {results_df['dataset'].nunique()}\n")
        f.write(f"- **Methods**: {results_df['method'].nunique()}\n")
        f.write(f"- **Total experiments**: {len(results_df)}\n\n")

        # Results by dataset
        for dataset_name in results_df['dataset'].unique():
            f.write(f"\n## {dataset_name}\n\n")

            dataset_results = results_df[results_df['dataset'] == dataset_name]

            # Create comparison table
            f.write("| Method | JSD ↓ | T2A ↑ | A2T ↑ | Entropy ↑ | DCR ↑ |\n")
            f.write("|--------|-------|-------|-------|-----------|-------|\n")

            for _, row in dataset_results.iterrows():
                method = row['method']
                jsd = f"{row.get('jsd', 0):.4f}" if 'jsd' in row else "N/A"
                t2a = f"{row.get('t2a_accuracy', 0):.4f}" if 't2a_accuracy' in row else "N/A"
                a2t = f"{row.get('a2t_accuracy', 0):.4f}" if 'a2t_accuracy' in row else "N/A"
                entropy = f"{row.get('joint_entropy', 0):.4f}" if 'joint_entropy' in row else "N/A"
                dcr = f"{row.get('dcr_mean', 0):.4f}" if 'dcr_mean' in row else "N/A"

                f.write(f"| {method} | {jsd} | {t2a} | {a2t} | {entropy} | {dcr} |\n")

            f.write("\n**Interpretation**:\n")
            f.write("- JSD (↓ lower is better): Measures fidelity to real data distribution\n")
            f.write("- T2A/A2T (↑ higher is better): Measures utility (cross-modal correlation)\n")
            f.write("- Entropy (↑ higher is better): Measures diversity\n")
            f.write("- DCR (↑ higher is better): Measures privacy preservation\n\n")

        # Cross-dataset comparison
        f.write("\n## Cross-Dataset Method Comparison\n\n")

        for method in results_df['method'].unique():
            f.write(f"\n### {method}\n\n")

            method_results = results_df[results_df['method'] == method]

            f.write("| Dataset | JSD | T2A | A2T | Entropy | DCR |\n")
            f.write("|---------|-----|-----|-----|---------|-----|\n")

            for _, row in method_results.iterrows():
                dataset = row['dataset']
                jsd = f"{row.get('jsd', 0):.4f}" if 'jsd' in row else "N/A"
                t2a = f"{row.get('t2a_accuracy', 0):.4f}" if 't2a_accuracy' in row else "N/A"
                a2t = f"{row.get('a2t_accuracy', 0):.4f}" if 'a2t_accuracy' in row else "N/A"
                entropy = f"{row.get('joint_entropy', 0):.4f}" if 'joint_entropy' in row else "N/A"
                dcr = f"{row.get('dcr_mean', 0):.4f}" if 'dcr_mean' in row else "N/A"

                f.write(f"| {dataset} | {jsd} | {t2a} | {a2t} | {entropy} | {dcr} |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        # Best method per metric
        if 'jsd' in results_df.columns:
            best_jsd = results_df.loc[results_df['jsd'].idxmin()]
            f.write(f"- **Best Fidelity (JSD)**: {best_jsd['method']} on {best_jsd['dataset']} ({best_jsd['jsd']:.4f})\n")

        if 't2a_accuracy' in results_df.columns:
            best_t2a = results_df.loc[results_df['t2a_accuracy'].idxmax()]
            f.write(f"- **Best Utility (T2A)**: {best_t2a['method']} on {best_t2a['dataset']} ({best_t2a['t2a_accuracy']:.4f})\n")

        if 'joint_entropy' in results_df.columns:
            best_entropy = results_df.loc[results_df['joint_entropy'].idxmax()]
            f.write(f"- **Best Diversity**: {best_entropy['method']} on {best_entropy['dataset']} ({best_entropy['joint_entropy']:.4f})\n")

        if 'dcr_mean' in results_df.columns:
            best_dcr = results_df.loc[results_df['dcr_mean'].idxmax()]
            f.write(f"- **Best Privacy**: {best_dcr['method']} on {best_dcr['dataset']} ({best_dcr['dcr_mean']:.4f})\n")

    print(f"✓ Generated experiment report: {report_path}")

    return report_path


def main():
    """Run complete experimental pipeline."""
    print_banner("MULTIMODAL SYNTHETIC DATA GENERATION - FULL EXPERIMENT")

    start_time = time.time()

    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(__file__).parent / "experiments" / timestamp
    synthetic_dir = output_root / "synthetic_data"
    results_dir = output_root / "results"

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_root}")

    # Load datasets
    datasets = load_datasets()

    if not datasets:
        print("✗ No datasets found. Please run clean_datasets.py first.")
        return

    print(f"\n✓ Loaded {len(datasets)} datasets")

    # Phase 1: Generate synthetic data
    synthetic_results = generate_synthetic_data(datasets, synthetic_dir)
    print(f"\n✓ Generated {len(synthetic_results)} synthetic datasets")

    # Phase 2: Evaluate
    evaluation_results = evaluate_synthetic_data(datasets, synthetic_results, results_dir)
    print(f"\n✓ Completed {len(evaluation_results)} evaluations")

    # Phase 3: Visualize
    report_path = generate_visualizations(evaluation_results, results_dir)

    # Final summary
    elapsed = time.time() - start_time

    print_banner("EXPERIMENT COMPLETE")

    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {output_root}")
    print(f"  - Synthetic data: {synthetic_dir}")
    print(f"  - Evaluation results: {results_dir}")
    print(f"  - Report: {report_path}")

    print("\n🎉 All experiments completed successfully!")


if __name__ == '__main__':
    main()
