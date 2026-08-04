"""
Generate Final Experiment Report

Reads the two output CSVs produced by reproduce.sh and writes a consolidated
Markdown summary:
  - traditional_metrics/traditional_metrics_results.csv  → Table 1
  - syneval/four_dimensions/four_dimensions_results.csv  → Table 2 / Figure 3
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def find_latest_experiment():
    """Find the most recent experiment directory."""
    exp_root = Path(__file__).parent / "experiments"

    if not exp_root.exists():
        return None

    exp_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir()], reverse=True)
    return exp_dirs[0] if exp_dirs else None


def load_results(exp_dir):
    """Load the two result CSVs; return (df_trad, df_four) either may be None."""
    trad_file = exp_dir / "traditional_metrics" / "traditional_metrics_results.csv"
    four_dim_file = exp_dir / "syneval" / "four_dimensions" / "four_dimensions_results.csv"

    df_trad = None
    if trad_file.exists():
        df_trad = pd.read_csv(trad_file)
        print(f"✓ Loaded traditional metrics ({len(df_trad)} rows): {trad_file}")
    else:
        print(f"⚠  Traditional metrics not found: {trad_file}")

    df_four = None
    if four_dim_file.exists():
        df_four = pd.read_csv(four_dim_file)
        print(f"✓ Loaded four-dimension results ({len(df_four)} rows): {four_dim_file}")
    else:
        print(f"⚠  Four-dimension results not found: {four_dim_file}")

    return df_trad, df_four


def _fmt(val, decimals=4):
    """Format a numeric value or return 'N/A'."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def write_traditional_section(f, df):
    """Write the Table 1 section (traditional isolated metrics)."""
    f.write("## Table 1 — Traditional Isolated Metrics\n\n")
    f.write("Metrics: KS Complement and TV Complement (tabular, SDV), BERTScore F1 (text).\n\n")

    for dataset in df['dataset'].unique():
        f.write(f"### {dataset.replace('_', ' ').title()}\n\n")
        sub = df[df['dataset'] == dataset]
        f.write("| Baseline | KS Complement | TV Complement | BERTScore F1 |\n")
        f.write("|----------|--------------|--------------|-------------|\n")
        for _, row in sub.iterrows():
            f.write(
                f"| {row['baseline']} "
                f"| {_fmt(row.get('tabular_avg_KS'))} "
                f"| {_fmt(row.get('tabular_avg_TV'))} "
                f"| {_fmt(row.get('text_f1'))} |\n"
            )
        f.write("\n")


def write_four_dim_section(f, df):
    """Write the Table 2 / Figure 3 section (SynEval four-dimension evaluation)."""
    f.write("## Table 2 — SynEval Four-Dimension Evaluation\n\n")
    f.write(
        "Axes: Fidelity (conditional JSD ↓), Utility (T2A / A2T accuracy ↑), "
        "Diversity (entropy ↑), Privacy (mean DCR ↑).\n\n"
    )

    for dataset in df['dataset'].unique():
        f.write(f"### {dataset.replace('_', ' ').title()}\n\n")
        sub = df[df['dataset'] == dataset]
        f.write("| Baseline | JSD ↓ | T2A ↑ | A2T ↑ | Entropy ↑ | DCR ↑ |\n")
        f.write("|----------|-------|-------|-------|-----------|-------|\n")
        for _, row in sub.iterrows():
            f.write(
                f"| {row['baseline']} "
                f"| {_fmt(row.get('fidelity_jsd_conditional'))} "
                f"| {_fmt(row.get('utility_t2a_accuracy'))} "
                f"| {_fmt(row.get('utility_a2t_accuracy'))} "
                f"| {_fmt(row.get('diversity_entropy'))} "
                f"| {_fmt(row.get('privacy_mean_dcr'))} |\n"
            )
        f.write("\n")

    # Best-per-metric summary
    f.write("### Best results per metric\n\n")
    checks = [
        ("Fidelity (lowest JSD)", "fidelity_jsd_conditional", "idxmin"),
        ("Utility T2A (highest)",  "utility_t2a_accuracy",      "idxmax"),
        ("Utility A2T (highest)",  "utility_a2t_accuracy",      "idxmax"),
        ("Diversity (highest entropy)", "diversity_entropy",     "idxmax"),
        ("Privacy (highest DCR)", "privacy_mean_dcr",           "idxmax"),
    ]
    for label, col, method in checks:
        if col in df.columns and not df[col].isna().all():
            idx = getattr(df[col].dropna(), method)()
            row = df.loc[idx]
            f.write(
                f"- **{label}**: `{row['baseline']}` on `{row['dataset']}`"
                f" ({_fmt(row[col])})\n"
            )
    f.write("\n")


def generate_summary_report(exp_dir, df_trad, df_four):
    """Write FINAL_REPORT.md in the experiment directory."""
    report_path = exp_dir / "FINAL_REPORT.md"

    with open(report_path, 'w') as f:
        f.write("# Multimodal Synthetic Data Evaluation — Final Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Experiment directory**: `{exp_dir}`\n\n")
        f.write("---\n\n")

        if df_trad is not None:
            write_traditional_section(f, df_trad)
        else:
            f.write("## Table 1 — Traditional Isolated Metrics\n\n")
            f.write("_Results not available. Run `python run_traditional_metrics.py` first._\n\n")

        if df_four is not None:
            write_four_dim_section(f, df_four)
        else:
            f.write("## Table 2 — SynEval Four-Dimension Evaluation\n\n")
            f.write(
                "_Results not available. Run `python syneval_quantization.py` then "
                "`python syneval_four_dimensions.py` first._\n\n"
            )

        f.write("---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"✓ Generated final report: {report_path}")
    return report_path


def main():
    """Generate final report."""
    print("=" * 80)
    print("GENERATING FINAL EXPERIMENT REPORT")
    print("=" * 80)

    exp_dir = find_latest_experiment()
    if exp_dir is None:
        print("❌ No experiment directory found under experiments/")
        sys.exit(1)

    print(f"✓ Found experiment: {exp_dir}")

    df_trad, df_four = load_results(exp_dir)

    if df_trad is None and df_four is None:
        print(
            "\n❌ No results found. Run reproduce.sh (or the individual step scripts) first.\n"
            f"   Expected:\n"
            f"     {exp_dir / 'traditional_metrics' / 'traditional_metrics_results.csv'}\n"
            f"     {exp_dir / 'syneval' / 'four_dimensions' / 'four_dimensions_results.csv'}"
        )
        sys.exit(1)

    report_path = generate_summary_report(exp_dir, df_trad, df_four)

    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nFinal report : {report_path}")
    if df_trad is not None:
        print(f"Table 1 CSV  : {exp_dir / 'traditional_metrics' / 'traditional_metrics_results.csv'}")
    if df_four is not None:
        print(f"Table 2 CSV  : {exp_dir / 'syneval' / 'four_dimensions' / 'four_dimensions_results.csv'}")
    print(f"Synthetic data: {exp_dir / 'synthetic_data'}")


if __name__ == '__main__':
    main()
