#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use("seaborn")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class SynEvalPlotter:
    def __init__(self, output_dir: str = "./plots"):
        """
        Initialize the SynEval plotter.

        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_fidelity_results(
        self, fidelity_results: Dict, save_plots: bool = True
    ) -> List[str]:
        """Generate plots for fidelity evaluation results."""
        plot_files = []

        try:
            # 1. Diagnostic and Quality Scores Comparison
            if "diagnostic" in fidelity_results and "quality" in fidelity_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Diagnostic scores
                diagnostic_scores = {
                    k: v
                    for k, v in fidelity_results["diagnostic"].items()
                    if k != "Overall" and v is not None
                }
                if diagnostic_scores:
                    ax1.bar(
                        diagnostic_scores.keys(),
                        diagnostic_scores.values(),
                        color=["#2E86AB", "#A23B72", "#F18F01"],
                    )
                    ax1.set_title("Diagnostic Scores", fontsize=14, fontweight="bold")
                    ax1.set_ylabel("Score")
                    ax1.set_ylim(0, 1)
                    ax1.grid(True, alpha=0.3)

                    # Add value labels on bars
                    for i, v in enumerate(diagnostic_scores.values()):
                        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

                # Quality scores
                quality_scores = {
                    k: v
                    for k, v in fidelity_results["quality"].items()
                    if k != "Overall" and v is not None
                }
                if quality_scores:
                    ax2.bar(
                        quality_scores.keys(),
                        quality_scores.values(),
                        color=["#2E86AB", "#A23B72", "#F18F01"],
                    )
                    ax2.set_title("Quality Scores", fontsize=14, fontweight="bold")
                    ax2.set_ylabel("Score")
                    ax2.set_ylim(0, 1)
                    ax2.grid(True, alpha=0.3)

                    # Add value labels on bars
                    for i, v in enumerate(quality_scores.values()):
                        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

                plt.tight_layout()
                if save_plots:
                    plot_file = self.output_dir / "fidelity_diagnostic_quality.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    plot_files.append(str(plot_file))
                plt.close()

            # 2. Text Analysis Plots (if text columns exist)
            if "text" in fidelity_results:
                for col_name, col_results in fidelity_results["text"].items():
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle(
                        f"Text Analysis for Column: {col_name}",
                        fontsize=16,
                        fontweight="bold",
                    )

                    # Length statistics
                    if "length_stats" in col_results:
                        length_stats = col_results["length_stats"]
                        categories = [
                            "Original Mean",
                            "Original Std",
                            "Synthetic Mean",
                            "Synthetic Std",
                        ]
                        values = [
                            length_stats.get("original_mean", 0),
                            length_stats.get("original_std", 0),
                            length_stats.get("synthetic_mean", 0),
                            length_stats.get("synthetic_std", 0),
                        ]

                        ax1.bar(
                            categories,
                            values,
                            color=["#2E86AB", "#2E86AB", "#A23B72", "#A23B72"],
                        )
                        ax1.set_title("Text Length Statistics")
                        ax1.set_ylabel("Characters")
                        ax1.grid(True, alpha=0.3)

                    # Word count statistics
                    if "word_count_stats" in col_results:
                        word_stats = col_results["word_count_stats"]
                        categories = [
                            "Original Mean",
                            "Original Std",
                            "Synthetic Mean",
                            "Synthetic Std",
                        ]
                        values = [
                            word_stats.get("original_mean", 0),
                            word_stats.get("original_std", 0),
                            word_stats.get("synthetic_mean", 0),
                            word_stats.get("synthetic_std", 0),
                        ]

                        ax2.bar(
                            categories,
                            values,
                            color=["#2E86AB", "#2E86AB", "#A23B72", "#A23B72"],
                        )
                        ax2.set_title("Word Count Statistics")
                        ax2.set_ylabel("Words")
                        ax2.grid(True, alpha=0.3)

                    # Keyword comparison
                    if (
                        "keyword_analysis" in col_results
                        and col_results["keyword_analysis"]
                    ):
                        keyword_analysis = col_results["keyword_analysis"]
                        if (
                            "original_top_keywords" in keyword_analysis
                            and "synthetic_top_keywords" in keyword_analysis
                        ):
                            # Get top 5 keywords from both
                            orig_keywords = list(
                                keyword_analysis["original_top_keywords"].items()
                            )[:5]
                            syn_keywords = list(
                                keyword_analysis["synthetic_top_keywords"].items()
                            )[:5]

                            orig_words, orig_scores = (
                                zip(*orig_keywords) if orig_keywords else ([], [])
                            )
                            syn_words, syn_scores = (
                                zip(*syn_keywords) if syn_keywords else ([], [])
                            )

                            x = np.arange(max(len(orig_words), len(syn_words)))
                            width = 0.35

                            ax3.bar(
                                x - width / 2,
                                orig_scores,
                                width,
                                label="Original",
                                color="#2E86AB",
                            )
                            ax3.bar(
                                x + width / 2,
                                syn_scores,
                                width,
                                label="Synthetic",
                                color="#A23B72",
                            )
                            ax3.set_title("Top Keywords Comparison")
                            ax3.set_ylabel("TF-IDF Score")
                            ax3.set_xticks(x)
                            ax3.set_xticklabels(orig_words, rotation=45, ha="right")
                            ax3.legend()
                            ax3.grid(True, alpha=0.3)

                    # Sentiment analysis
                    if (
                        "sentiment_analysis" in col_results
                        and col_results["sentiment_analysis"]
                    ):
                        sent_analysis = col_results["sentiment_analysis"]
                        if (
                            "original_sentiment_distribution" in sent_analysis
                            and "synthetic_sentiment_distribution" in sent_analysis
                        ):
                            orig_dist = sent_analysis["original_sentiment_distribution"]
                            syn_dist = sent_analysis["synthetic_sentiment_distribution"]

                            categories = ["Negative", "Neutral", "Positive"]
                            orig_values = [
                                orig_dist.get("negative", 0),
                                orig_dist.get("neutral", 0),
                                orig_dist.get("positive", 0),
                            ]
                            syn_values = [
                                syn_dist.get("negative", 0),
                                syn_dist.get("neutral", 0),
                                syn_dist.get("positive", 0),
                            ]

                            x = np.arange(len(categories))
                            width = 0.35

                            ax4.bar(
                                x - width / 2,
                                orig_values,
                                width,
                                label="Original",
                                color="#2E86AB",
                            )
                            ax4.bar(
                                x + width / 2,
                                syn_values,
                                width,
                                label="Synthetic",
                                color="#A23B72",
                            )
                            ax4.set_title("Sentiment Distribution")
                            ax4.set_ylabel("Percentage (%)")
                            ax4.set_xticks(x)
                            ax4.set_xticklabels(categories)
                            ax4.legend()
                            ax4.grid(True, alpha=0.3)

                    plt.tight_layout()
                    if save_plots:
                        plot_file = (
                            self.output_dir / f"fidelity_text_analysis_{col_name}.png"
                        )
                        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                        plot_files.append(str(plot_file))
                    plt.close()

        except Exception as e:
            logger.error(f"Error plotting fidelity results: {str(e)}")

        return plot_files

    def plot_utility_results(
        self, utility_results: Dict, save_plots: bool = True
    ) -> List[str]:
        """Generate plots for utility evaluation results."""
        plot_files = []

        try:
            if "error" in utility_results:
                logger.warning(
                    f"Utility evaluation had errors: {utility_results['error']}"
                )
                return plot_files

            # 1. Model Performance Comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Extract accuracy scores
            real_accuracy = utility_results.get("real_data_model", {}).get(
                "accuracy", 0
            )
            syn_accuracy = utility_results.get("synthetic_data_model", {}).get(
                "accuracy", 0
            )

            # Accuracy comparison
            models = ["Real Data Model", "Synthetic Data Model"]
            accuracies = [real_accuracy, syn_accuracy]
            colors = ["#2E86AB", "#A23B72"]

            bars = ax1.bar(models, accuracies, color=colors)
            ax1.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Accuracy")
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc:.3f}",
                    ha="center",
                    va="bottom",
                )

            # Performance gap
            performance_gap = abs(real_accuracy - syn_accuracy)
            gap_color = (
                "green"
                if performance_gap < 0.1
                else "orange"
                if performance_gap < 0.2
                else "red"
            )

            ax2.bar(["Performance Gap"], [performance_gap], color=gap_color)
            ax2.set_title("Performance Gap", fontsize=14, fontweight="bold")
            ax2.set_ylabel("Absolute Difference")
            ax2.set_ylim(0, max(0.3, performance_gap * 1.2))
            ax2.grid(True, alpha=0.3)

            # Add threshold lines
            ax2.axhline(
                y=0.1, color="orange", linestyle="--", alpha=0.7, label="Good (<0.1)"
            )
            ax2.axhline(
                y=0.2, color="red", linestyle="--", alpha=0.7, label="Poor (>0.2)"
            )
            ax2.legend()

            # Add value label
            ax2.text(
                0,
                performance_gap + 0.005,
                f"{performance_gap:.3f}",
                ha="center",
                va="bottom",
            )

            plt.tight_layout()
            if save_plots:
                plot_file = self.output_dir / "utility_model_comparison.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plot_files.append(str(plot_file))
            plt.close()

            # 2. Detailed classification metrics (if available)
            if (
                "real_data_model" in utility_results
                and "synthetic_data_model" in utility_results
            ):
                real_model = utility_results["real_data_model"]
                syn_model = utility_results["synthetic_data_model"]

                # Extract per-class metrics if available
                if "weighted avg" in real_model and "weighted avg" in syn_model:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle(
                        "Detailed Classification Metrics",
                        fontsize=16,
                        fontweight="bold",
                    )

                    metrics = ["precision", "recall", "f1-score"]
                    real_metrics = [
                        real_model["weighted avg"].get(m, 0) for m in metrics
                    ]
                    syn_metrics = [syn_model["weighted avg"].get(m, 0) for m in metrics]

                    x = np.arange(len(metrics))
                    width = 0.35

                    ax1.bar(
                        x - width / 2,
                        real_metrics,
                        width,
                        label="Real Data",
                        color="#2E86AB",
                    )
                    ax1.bar(
                        x + width / 2,
                        syn_metrics,
                        width,
                        label="Synthetic Data",
                        color="#A23B72",
                    )
                    ax1.set_title("Weighted Average Metrics")
                    ax1.set_ylabel("Score")
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(metrics)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Macro averages
                    if "macro avg" in real_model and "macro avg" in syn_model:
                        real_macro = [
                            real_model["macro avg"].get(m, 0) for m in metrics
                        ]
                        syn_macro = [syn_model["macro avg"].get(m, 0) for m in metrics]

                        ax2.bar(
                            x - width / 2,
                            real_macro,
                            width,
                            label="Real Data",
                            color="#2E86AB",
                        )
                        ax2.bar(
                            x + width / 2,
                            syn_macro,
                            width,
                            label="Synthetic Data",
                            color="#A23B72",
                        )
                        ax2.set_title("Macro Average Metrics")
                        ax2.set_ylabel("Score")
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(metrics)
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)

                    # Task information
                    task_info = [
                        f"Task Type: {utility_results.get('task_type', 'Unknown')}",
                        f"Training Size: {utility_results.get('training_size', 0):,}",
                        f"Test Size: {utility_results.get('test_size', 0):,}",
                        f"Input Columns: {', '.join(utility_results.get('input_columns', []))}",
                        f"Output Columns: {', '.join(utility_results.get('output_columns', []))}",
                    ]

                    ax3.text(
                        0.1,
                        0.8,
                        "\n".join(task_info),
                        transform=ax3.transAxes,
                        fontsize=12,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
                    )
                    ax3.set_title("Task Information")
                    ax3.axis("off")

                    # Performance summary
                    summary_text = [
                        f"Real Data Accuracy: {real_accuracy:.3f}",
                        f"Synthetic Data Accuracy: {syn_accuracy:.3f}",
                        f"Performance Gap: {performance_gap:.3f}",
                        f"Quality: {'Good' if performance_gap < 0.1 else 'Moderate' if performance_gap < 0.2 else 'Poor'}",
                    ]

                    ax4.text(
                        0.1,
                        0.8,
                        "\n".join(summary_text),
                        transform=ax4.transAxes,
                        fontsize=12,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                    )
                    ax4.set_title("Performance Summary")
                    ax4.axis("off")

                    plt.tight_layout()
                    if save_plots:
                        plot_file = self.output_dir / "utility_detailed_metrics.png"
                        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                        plot_files.append(str(plot_file))
                    plt.close()

        except Exception as e:
            logger.error(f"Error plotting utility results: {str(e)}")

        return plot_files

    def plot_diversity_results(
        self, diversity_results: Dict, save_plots: bool = True
    ) -> List[str]:
        """Generate plots for diversity evaluation results."""
        plot_files = []

        try:
            # 1. Tabular Diversity Overview
            if "tabular_diversity" in diversity_results:
                tabular = diversity_results["tabular_diversity"]

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(
                    "Tabular Diversity Analysis", fontsize=16, fontweight="bold"
                )

                # Coverage metrics
                if "coverage" in tabular:
                    coverage_data = tabular["coverage"]
                    if coverage_data:
                        columns = list(coverage_data.keys())
                        values = list(coverage_data.values())

                        colors = [
                            "green" if v >= 80 else "orange" if v >= 60 else "red"
                            for v in values
                        ]
                        bars = ax1.bar(columns, values, color=colors)
                        ax1.set_title("Column Coverage (%)")
                        ax1.set_ylabel("Coverage Percentage")
                        ax1.set_ylim(0, 100)
                        ax1.tick_params(axis="x", rotation=45)
                        ax1.grid(True, alpha=0.3)

                        # Add value labels
                        for bar, value in zip(bars, values):
                            ax1.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 1,
                                f"{value:.1f}%",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )

                # Uniqueness metrics
                if "uniqueness" in tabular:
                    uniqueness = tabular["uniqueness"]
                    metrics = [
                        "Synthetic Duplicate Ratio",
                        "Original Duplicate Ratio",
                        "Relative Duplication",
                    ]
                    values = [
                        uniqueness.get("synthetic_duplicate_ratio", 0),
                        uniqueness.get("original_duplicate_ratio", 0),
                        uniqueness.get("relative_duplication", 0),
                    ]

                    colors = ["red" if v > 5 else "green" for v in values]
                    bars = ax2.bar(metrics, values, color=colors)
                    ax2.set_title("Uniqueness Metrics (%)")
                    ax2.set_ylabel("Percentage")
                    ax2.tick_params(axis="x", rotation=45)
                    ax2.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, value in zip(bars, values):
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            f"{value:.1f}%",
                            ha="center",
                            va="bottom",
                        )

                # Entropy comparison
                if (
                    "entropy_metrics" in tabular
                    and "dataset_entropy" in tabular["entropy_metrics"]
                ):
                    entropy = tabular["entropy_metrics"]["dataset_entropy"]
                    real_entropy = entropy.get("real", 0)
                    syn_entropy = entropy.get("synthetic", 0)
                    ratio = entropy.get("entropy_ratio", 0)

                    # Entropy values
                    ax3.bar(
                        ["Real Data", "Synthetic Data"],
                        [real_entropy, syn_entropy],
                        color=["#2E86AB", "#A23B72"],
                    )
                    ax3.set_title("Dataset Entropy Comparison")
                    ax3.set_ylabel("Entropy")
                    ax3.grid(True, alpha=0.3)

                    # Entropy ratio
                    ax4.bar(
                        ["Entropy Ratio"],
                        [ratio],
                        color=(
                            "green"
                            if 0.8 <= ratio <= 1.2
                            else "orange"
                            if 0.6 <= ratio <= 1.4
                            else "red"
                        ),
                    )
                    ax4.set_title("Entropy Ratio (Synthetic/Real)")
                    ax4.set_ylabel("Ratio")
                    ax4.axhline(
                        y=1,
                        color="black",
                        linestyle="--",
                        alpha=0.7,
                        label="Ideal (1.0)",
                    )
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)

                plt.tight_layout()
                if save_plots:
                    plot_file = self.output_dir / "diversity_tabular_overview.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    plot_files.append(str(plot_file))
                plt.close()

            # 2. Text Diversity (if available)
            if "text_diversity" in diversity_results:
                text_div = diversity_results["text_diversity"]

                for dataset_type in ["synthetic", "real"]:
                    if dataset_type in text_div:
                        for col_name, col_results in text_div[dataset_type].items():
                            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                                2, 2, figsize=(16, 12)
                            )
                            fig.suptitle(
                                f"Text Diversity Analysis - {dataset_type.title()} Data ({col_name})",
                                fontsize=16,
                                fontweight="bold",
                            )

                            # Lexical diversity
                            if "lexical_diversity" in col_results:
                                lexical = col_results["lexical_diversity"]
                                ngrams = []
                                unique_ratios = []

                                for ng in lexical.keys():
                                    if (
                                        isinstance(lexical[ng], dict)
                                        and "unique_ratio" in lexical[ng]
                                    ):
                                        ngrams.append(ng)
                                        unique_ratios.append(
                                            lexical[ng]["unique_ratio"]
                                        )
                                    elif isinstance(lexical[ng], (int, float)):
                                        # Skip non-dict values like sample_size
                                        continue
                                    else:
                                        logger.warning(
                                            f"lexical[{ng}] is not a dict or missing 'unique_ratio' for column '{col_name}' in dataset '{dataset_type}': {lexical[ng]} (type: {type(lexical[ng])})"
                                        )

                                if ngrams:
                                    ax1.bar(ngrams, unique_ratios, color="#2E86AB")
                                    ax1.set_title("Lexical Diversity (Unique Ratio)")
                                    ax1.set_ylabel("Unique Ratio")
                                    ax1.grid(True, alpha=0.3)
                                else:
                                    ax1.text(
                                        0.5,
                                        0.5,
                                        "No valid lexical diversity data",
                                        ha="center",
                                        va="center",
                                        transform=ax1.transAxes,
                                    )
                                    ax1.set_title("Lexical Diversity (Unique Ratio)")
                                    ax1.set_ylabel("Unique Ratio")
                                    ax1.grid(True, alpha=0.3)

                            # Semantic diversity
                            if "semantic_diversity" in col_results:
                                semantic = col_results["semantic_diversity"]
                                metrics = [
                                    "Total MST Weight",
                                    "Average Edge Weight",
                                    "Distinct Ratio",
                                ]
                                values = [
                                    semantic.get("total_mst_weight", 0),
                                    semantic.get("average_edge_weight", 0),
                                    semantic.get("distinct_ratio", 0),
                                ]

                                ax2.bar(metrics, values, color="#A23B72")
                                ax2.set_title("Semantic Diversity Metrics")
                                ax2.tick_params(axis="x", rotation=45)
                                ax2.grid(True, alpha=0.3)

                            # Sentiment diversity
                            if "sentiment_diversity" in col_results:
                                sentiment = col_results["sentiment_diversity"]
                                sbr = sentiment.get("sentiment_by_rating", None)
                                if sbr is not None and sbr != {}:
                                    if not isinstance(sbr, dict):
                                        logger.error(
                                            f"sentiment_by_rating is not a dict for column '{col_name}' in dataset '{dataset_type}': {sbr} (type: {type(sbr)})"
                                        )
                                        ax3.text(
                                            0.5,
                                            0.5,
                                            f"Error: sentiment_by_rating is not a dict",
                                            ha="center",
                                            va="center",
                                            transform=ax3.transAxes,
                                            color="red",
                                        )
                                        ax3.set_title("Sentiment by Rating")
                                        ax3.set_ylabel("Positive Sentiment %")
                                        ax3.set_xlabel("Rating")
                                        ax3.grid(True, alpha=0.3)
                                    else:
                                        ratings = list(sbr.keys())
                                        values = list(sbr.values())
                                        ax3.bar(ratings, values, color="#F18F01")
                                        ax3.set_title("Sentiment by Rating")
                                        ax3.set_ylabel("Positive Sentiment %")
                                        ax3.set_xlabel("Rating")
                                        ax3.grid(True, alpha=0.3)
                                else:
                                    # No sentiment_by_rating data
                                    ax3.text(
                                        0.5,
                                        0.5,
                                        "No sentiment by rating data",
                                        ha="center",
                                        va="center",
                                        transform=ax3.transAxes,
                                    )
                                    ax3.set_title("Sentiment by Rating")
                                    ax3.set_ylabel("Positive Sentiment %")
                                    ax3.set_xlabel("Rating")
                                    ax3.grid(True, alpha=0.3)

                                if "sentiment_alignment_score" in sentiment:
                                    alignment = sentiment["sentiment_alignment_score"]
                                    ax4.bar(
                                        ["Sentiment Alignment"],
                                        [alignment],
                                        color=(
                                            "green"
                                            if alignment > 0.7
                                            else "orange"
                                            if alignment > 0.5
                                            else "red"
                                        ),
                                    )
                                    ax4.set_title("Sentiment Alignment Score")
                                    ax4.set_ylabel("Score")
                                    ax4.set_ylim(0, 1)
                                    ax4.grid(True, alpha=0.3)
                                else:
                                    # No sentiment alignment score
                                    ax4.text(
                                        0.5,
                                        0.5,
                                        "No sentiment alignment data",
                                        ha="center",
                                        va="center",
                                        transform=ax4.transAxes,
                                    )
                                    ax4.set_title("Sentiment Alignment Score")
                                    ax4.set_ylabel("Score")
                                    ax4.set_ylim(0, 1)
                                    ax4.grid(True, alpha=0.3)

                            plt.tight_layout()
                            if save_plots:
                                plot_file = (
                                    self.output_dir
                                    / f"diversity_text_{dataset_type}_{col_name}.png"
                                )
                                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                                plot_files.append(str(plot_file))
                            plt.close()

        except Exception as e:
            logger.error(f"Error plotting diversity results: {str(e)}")
            import traceback

            traceback.print_exc()

        return plot_files

    def plot_privacy_results(
        self, privacy_results: Dict, save_plots: bool = True
    ) -> List[str]:
        """Generate plots for privacy evaluation results."""
        plot_files = []

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Privacy Risk Analysis", fontsize=16, fontweight="bold")

            # 1. Exact Match Analysis
            if "exact_matches" in privacy_results:
                exact_matches = privacy_results["exact_matches"]
                match_percentage = exact_matches.get("exact_match_percentage", 0)
                risk_level = exact_matches.get("risk_level", "unknown")

                color = "red" if risk_level == "high" else "green"
                ax1.bar(["Exact Match %"], [match_percentage], color=color)
                ax1.set_title("Exact Match Percentage")
                ax1.set_ylabel("Percentage")
                ax1.axhline(
                    y=5,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="High Risk Threshold",
                )
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Add risk level text
                ax1.text(
                    0,
                    match_percentage + 0.5,
                    f"Risk: {risk_level.upper()}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # 2. Membership Inference Attack
            if "membership_inference" in privacy_results:
                mia = privacy_results["membership_inference"]
                auc_score = mia.get("mia_auc_score", 0)
                risk_level = mia.get("risk_level", "unknown")

                color = "red" if risk_level == "high" else "green"
                ax2.bar(["MIA AUC Score"], [auc_score], color=color)
                ax2.set_title("Membership Inference Attack")
                ax2.set_ylabel("AUC Score")
                ax2.set_ylim(0, 1)
                ax2.axhline(
                    y=0.7,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="High Risk Threshold",
                )
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Add risk level text
                ax2.text(
                    0,
                    auc_score + 0.02,
                    f"Risk: {risk_level.upper()}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # 3. Named Entity Analysis (if available)
            if (
                "named_entities" in privacy_results
                and "error" not in privacy_results["named_entities"]
            ):
                entities = privacy_results["named_entities"]

                # Entity density comparison
                if "synthetic" in entities and "original" in entities:
                    syn_density = entities["synthetic"].get("avg_entity_density", 0)
                    orig_density = entities["original"].get("avg_entity_density", 0)

                    ax3.bar(
                        ["Original", "Synthetic"],
                        [orig_density, syn_density],
                        color=["#2E86AB", "#A23B72"],
                    )
                    ax3.set_title("Entity Density Comparison")
                    ax3.set_ylabel("Entity Density")
                    ax3.grid(True, alpha=0.3)

                    # Add risk indicators
                    if syn_density > 0.1:
                        ax3.text(
                            1,
                            syn_density + 0.001,
                            "HIGH RISK",
                            ha="center",
                            va="bottom",
                            color="red",
                            fontweight="bold",
                        )

                # Entity overlap
                if "overlap" in entities:
                    overlap_pct = entities["overlap"].get("overlap_percentage", 0)
                    risk_level = entities["overlap"].get("risk_level", "unknown")

                    color = "red" if risk_level == "high" else "green"
                    ax4.bar(["Entity Overlap %"], [overlap_pct], color=color)
                    ax4.set_title("Entity Overlap Percentage")
                    ax4.set_ylabel("Percentage")
                    ax4.axhline(
                        y=50,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                        label="High Risk Threshold",
                    )
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)

                    # Add risk level text
                    ax4.text(
                        0,
                        overlap_pct + 1,
                        f"Risk: {risk_level.upper()}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            plt.tight_layout()
            if save_plots:
                plot_file = self.output_dir / "privacy_risk_overview.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plot_files.append(str(plot_file))
            plt.close()

            # 4. Anonymeter Results (if available)
            if (
                "anonymeter" in privacy_results
                and "error" not in privacy_results["anonymeter"]
            ):
                anonymeter = privacy_results["anonymeter"]

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(
                    "Anonymeter Privacy Attack Analysis", fontsize=16, fontweight="bold"
                )

                # Attack success rates
                attack_types = []
                attack_rates = []

                for attack_name, attack_data in anonymeter.items():
                    if isinstance(attack_data, dict) and "attack_rate" in attack_data:
                        attack_types.append(attack_name.replace("_", " ").title())
                        attack_rates.append(attack_data["attack_rate"])

                if attack_types:
                    colors = [
                        "red" if rate > 0.5 else "orange" if rate > 0.3 else "green"
                        for rate in attack_rates
                    ]
                    bars = ax1.bar(attack_types, attack_rates, color=colors)
                    ax1.set_title("Attack Success Rates")
                    ax1.set_ylabel("Success Rate")
                    ax1.set_ylim(0, 1)
                    ax1.tick_params(axis="x", rotation=45)
                    ax1.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, rate in zip(bars, attack_rates):
                        ax1.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{rate:.3f}",
                            ha="center",
                            va="bottom",
                        )

                # Risk scores
                risk_scores = []
                risk_names = []

                for attack_name, attack_data in anonymeter.items():
                    if isinstance(attack_data, dict) and "risk" in attack_data:
                        risk_names.append(attack_name.replace("_", " ").title())
                        risk_scores.append(attack_data["risk"])

                if risk_names:
                    colors = [
                        "red" if score > 0.5 else "orange" if score > 0.3 else "green"
                        for score in risk_scores
                    ]
                    bars = ax2.bar(risk_names, risk_scores, color=colors)
                    ax2.set_title("Privacy Risk Scores")
                    ax2.set_ylabel("Risk Score")
                    ax2.set_ylim(0, 1)
                    ax2.tick_params(axis="x", rotation=45)
                    ax2.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, score in zip(bars, risk_scores):
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{score:.3f}",
                            ha="center",
                            va="bottom",
                        )

                # Overall risk
                if "overall_risk" in anonymeter:
                    overall_risk = anonymeter["overall_risk"]
                    risk_score = overall_risk.get("risk_score", 0)
                    risk_level = overall_risk.get("risk_level", "unknown")

                    color = (
                        "red"
                        if risk_level == "high"
                        else "orange"
                        if risk_level == "medium"
                        else "green"
                    )
                    ax3.bar(["Overall Risk"], [risk_score], color=color)
                    ax3.set_title("Overall Privacy Risk")
                    ax3.set_ylabel("Risk Score")
                    ax3.set_ylim(0, 1)
                    ax3.grid(True, alpha=0.3)

                    # Add risk level text
                    ax3.text(
                        0,
                        risk_score + 0.02,
                        f"Risk Level: {risk_level.upper()}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                # Risk level distribution
                risk_levels = ["Low", "Medium", "High"]
                risk_counts = [0, 0, 0]

                for attack_data in anonymeter.values():
                    if isinstance(attack_data, dict) and "risk_level" in attack_data:
                        level = attack_data["risk_level"]
                        if level == "low":
                            risk_counts[0] += 1
                        elif level == "medium":
                            risk_counts[1] += 1
                        elif level == "high":
                            risk_counts[2] += 1

                colors = ["green", "orange", "red"]
                ax4.pie(
                    risk_counts,
                    labels=risk_levels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax4.set_title("Risk Level Distribution")

                plt.tight_layout()
                if save_plots:
                    plot_file = self.output_dir / "privacy_anonymeter_analysis.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    plot_files.append(str(plot_file))
                plt.close()

        except Exception as e:
            logger.error(f"Error plotting privacy results: {str(e)}")

        return plot_files

    def plot_all_results(
        self, results: Dict, save_plots: bool = True
    ) -> Dict[str, List[str]]:
        """Generate all plots for the evaluation results."""
        all_plot_files = {}

        logger.info("Generating fidelity plots...")
        if "fidelity" in results:
            all_plot_files["fidelity"] = self.plot_fidelity_results(
                results["fidelity"], save_plots
            )

        logger.info("Generating utility plots...")
        if "utility" in results:
            all_plot_files["utility"] = self.plot_utility_results(
                results["utility"], save_plots
            )

        logger.info("Generating diversity plots...")
        if "diversity" in results:
            all_plot_files["diversity"] = self.plot_diversity_results(
                results["diversity"], save_plots
            )

        logger.info("Generating privacy plots...")
        if "privacy" in results:
            all_plot_files["privacy"] = self.plot_privacy_results(
                results["privacy"], save_plots
            )

        # Create a summary plot
        if save_plots:
            self._create_summary_plot(results, all_plot_files)

        return all_plot_files

    def _create_summary_plot(self, results: Dict, plot_files: Dict[str, List[str]]):
        """Create a summary plot showing key metrics across all dimensions."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("SynEval Summary Dashboard", fontsize=18, fontweight="bold")

            # Fidelity summary
            if "fidelity" in results:
                fidelity = results["fidelity"]
                if "diagnostic" in fidelity and "quality" in fidelity:
                    diagnostic_score = (
                        fidelity["diagnostic"].get("Overall", {}).get("score", 0)
                    )
                    quality_score = (
                        fidelity["quality"].get("Overall", {}).get("score", 0)
                    )

                    ax1.bar(
                        ["Diagnostic", "Quality"],
                        [diagnostic_score, quality_score],
                        color=["#2E86AB", "#A23B72"],
                    )
                    ax1.set_title("Fidelity Scores")
                    ax1.set_ylabel("Score")
                    ax1.set_ylim(0, 1)
                    ax1.grid(True, alpha=0.3)

            # Utility summary
            if "utility" in results and "error" not in results["utility"]:
                utility = results["utility"]
                if "real_data_model" in utility and "synthetic_data_model" in utility:
                    real_acc = utility["real_data_model"].get("accuracy", 0)
                    syn_acc = utility["synthetic_data_model"].get("accuracy", 0)

                    ax2.bar(
                        ["Real Model", "Synthetic Model"],
                        [real_acc, syn_acc],
                        color=["#2E86AB", "#A23B72"],
                    )
                    ax2.set_title("Utility Performance")
                    ax2.set_ylabel("Accuracy")
                    ax2.set_ylim(0, 1)
                    ax2.grid(True, alpha=0.3)

            # Diversity summary
            if "diversity" in results and "tabular_diversity" in results["diversity"]:
                diversity = results["diversity"]["tabular_diversity"]
                if "uniqueness" in diversity:
                    uniqueness = diversity["uniqueness"]
                    duplicate_ratio = uniqueness.get("synthetic_duplicate_ratio", 0)

                    color = (
                        "green"
                        if duplicate_ratio < 5
                        else "orange"
                        if duplicate_ratio < 10
                        else "red"
                    )
                    ax3.bar(["Duplicate Ratio"], [duplicate_ratio], color=color)
                    ax3.set_title("Diversity - Duplicate Ratio")
                    ax3.set_ylabel("Percentage")
                    ax3.grid(True, alpha=0.3)

            # Privacy summary
            if "privacy" in results:
                privacy = results["privacy"]
                risk_scores = []

                if "exact_matches" in privacy:
                    exact_pct = privacy["exact_matches"].get(
                        "exact_match_percentage", 0
                    )
                    risk_scores.append(exact_pct / 100)  # Normalize to 0-1

                if "membership_inference" in privacy:
                    mia_auc = privacy["membership_inference"].get("mia_auc_score", 0)
                    risk_scores.append(mia_auc)

                if risk_scores:
                    avg_risk = np.mean(risk_scores)
                    color = (
                        "green"
                        if avg_risk < 0.3
                        else "orange"
                        if avg_risk < 0.7
                        else "red"
                    )
                    ax4.bar(["Average Privacy Risk"], [avg_risk], color=color)
                    ax4.set_title("Privacy Risk Summary")
                    ax4.set_ylabel("Risk Score")
                    ax4.set_ylim(0, 1)
                    ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.output_dir / "syneval_summary_dashboard.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plot_files["summary"] = [str(plot_file)]
            plt.close()

        except Exception as e:
            logger.error(f"Error creating summary plot: {str(e)}")
