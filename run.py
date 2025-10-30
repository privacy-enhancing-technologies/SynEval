#!/usr/bin/env python3

import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from fidelity import FidelityEvaluator
from utility import UtilityEvaluator
from diversity import DiversityEvaluator
from privacy import PrivacyEvaluator
import logging

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - environment specific
    torch = None  # type: ignore


def configure_device(
    device_preference: str, force_cpu: bool = False, gpu_memory_fraction: float = 0.8
) -> str:
    """
    Configure the device for computation based on user preferences and system capabilities.

    Args:
        device_preference: User's device preference ('auto', 'cpu', 'cuda')
        force_cpu: Whether to force CPU usage even if GPU is available
        gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)

    Returns:
        str: The configured device ('cpu' or 'cuda')
    """
    if force_cpu or torch is None:
        if device_preference == "cuda" and torch is None:
            print(
                "Warning: PyTorch is required for CUDA execution but is not installed. Falling back to CPU."
            )
        return "cpu"

    if device_preference == "cpu":
        return "cpu"
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            # Set GPU memory fraction
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            return "cuda"
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    else:  # auto
        if torch.cuda.is_available():
            # Set GPU memory fraction
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            return "cuda"
        else:
            return "cpu"


class SynEval:
    def __init__(
        self,
        synthetic_data: pd.DataFrame,
        original_data: pd.DataFrame,
        metadata: Dict,
        device: str = "auto",
    ):
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.metadata = metadata
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Initialize evaluators as None - will be created when needed
        self._fidelity_evaluator = None
        self._diversity_evaluator = None
        self._privacy_evaluator = None
        self._utility_evaluator = None

    def evaluate_fidelity(self, selected_metrics: List[str] = None) -> Dict:
        if self._fidelity_evaluator is None:
            self._fidelity_evaluator = FidelityEvaluator(
                self.synthetic_data,
                self.original_data,
                self.metadata,
                selected_metrics,
                self.device,
            )
        else:
            # Update selected metrics if evaluator already exists
            self._fidelity_evaluator.selected_metrics = (
                selected_metrics
                if selected_metrics
                else self._fidelity_evaluator.available_metrics
            )
        return self._fidelity_evaluator.evaluate()

    def evaluate_utility(
        self,
        input_columns: List[str],
        output_columns: List[str],
        selected_metrics: List[str] = None,
    ) -> Dict:
        # Utility evaluator is created fresh each time since it needs input/output columns
        utility_evaluator = UtilityEvaluator(
            self.synthetic_data,
            self.original_data,
            self.metadata,
            input_columns=input_columns,
            output_columns=output_columns,
            selected_metrics=selected_metrics,
            device=self.device,
        )
        return utility_evaluator.evaluate()

    def evaluate_diversity(self, selected_metrics: List[str] = None) -> Dict:
        if self._diversity_evaluator is None:
            self._diversity_evaluator = DiversityEvaluator(
                self.synthetic_data,
                self.original_data,
                self.metadata,
                selected_metrics=selected_metrics,
                device=self.device,
            )
        else:
            # Update selected metrics if evaluator already exists
            self._diversity_evaluator.selected_metrics = (
                selected_metrics
                if selected_metrics
                else self._diversity_evaluator.available_metrics
            )
        return self._diversity_evaluator.evaluate()

    def evaluate_privacy(self, selected_metrics: List[str] = None) -> Dict:
        if self._privacy_evaluator is None:
            self._privacy_evaluator = PrivacyEvaluator(
                self.synthetic_data,
                self.original_data,
                self.metadata,
                selected_metrics,
                self.device,
            )
        else:
            # Update selected metrics if evaluator already exists
            self._privacy_evaluator.selected_metrics = (
                selected_metrics
                if selected_metrics
                else self._privacy_evaluator.available_metrics
            )
        return self._privacy_evaluator.evaluate()

    def evaluate_correlation(self, correlation_types, **kwargs):
        if self._utility_evaluator is None:
            self._utility_evaluator = UtilityEvaluator(
                self.synthetic_data,
                self.original_data,
                self.metadata,
                input_columns=[],
                output_columns=[],
                device=self.device,
            )
        return self._utility_evaluator.evaluate_correlation(correlation_types, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Evaluation Framework")

    parser.add_argument(
        "--synthetic", required=True, help="Path to synthetic data CSV file"
    )
    parser.add_argument(
        "--original", required=True, help="Path to original data CSV file"
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON file")

    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=["fidelity", "utility", "diversity", "privacy"],
        default=["fidelity", "utility", "diversity", "privacy"],
        help="Evaluation dimensions to run (default: all)",
    )
    parser.add_argument(
        "--dimension",
        nargs="+",
        choices=["fidelity", "utility", "diversity", "privacy"],
        help="Alias for --dimensions (singular form)",
    )

    parser.add_argument(
        "--utility-input", nargs="+", help="Input columns for utility evaluation"
    )
    parser.add_argument(
        "--utility-output", nargs="+", help="Output columns for utility evaluation"
    )

    # Fidelity metrics selection
    parser.add_argument(
        "--fidelity-metrics",
        nargs="+",
        choices=["diagnostic", "quality", "text", "numerical_statistics"],
        help="Specific fidelity metrics to run (default: all)",
    )

    # Diversity metrics selection
    parser.add_argument(
        "--diversity-metrics",
        nargs="+",
        choices=["tabular_diversity", "text_diversity"],
        help="Specific diversity metrics to run (default: all)",
    )

    # Utility metrics selection
    parser.add_argument(
        "--utility-metrics",
        nargs="+",
        choices=["tstr_accuracy", "correlation_analysis"],
        help="Specific utility metrics to run (default: all)",
    )

    # Privacy metrics selection
    parser.add_argument(
        "--privacy-metrics",
        nargs="+",
        choices=[
            "exact_matches",
            "membership_inference",
            "tabular_privacy",
            "text_privacy",
            "anonymeter",
        ],
        help="Specific privacy metrics to run (default: all)",
    )

    # Correlation metrics (legacy support)
    parser.add_argument(
        "--correlation",
        nargs="+",
        choices=[
            "sentiment_rating",
            "keyword_category",
            "numeric_length",
            "semantic_tabular",
            "pii_text_leakage",
        ],
        help="Cross-modality correlation metrics to run (legacy, use --utility-metrics correlation_analysis)",
    )
    parser.add_argument(
        "--correlation-text-col",
        default="review",
        help="Text column for correlation analysis",
    )
    parser.add_argument(
        "--correlation-rating-col",
        default="rating",
        help="Rating column for sentiment correlation",
    )
    parser.add_argument(
        "--correlation-category-col",
        default="category",
        help="Category column for keyword correlation",
    )
    parser.add_argument(
        "--correlation-numeric-col",
        default="price",
        help="Numeric column for length correlation",
    )
    parser.add_argument(
        "--correlation-pii-col",
        default="user_id",
        help="PII column for leakage correlation",
    )
    parser.add_argument(
        "--correlation-top-n",
        type=int,
        default=50,
        help="Top N keywords for keyword-category correlation",
    )

    parser.add_argument(
        "--output",
        default="./evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots for all evaluation metrics and save them to the ./plots directory",
    )

    # Device selection arguments
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for computation (default: auto - automatically detect best available device)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available (overrides --device)",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use (0.0-1.0, default: 0.8)",
    )

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load data
    logger.info(f"Loading synthetic data from {args.synthetic}")
    synthetic_data = pd.read_csv(args.synthetic)

    logger.info(f"Loading original data from {args.original}")
    original_data = pd.read_csv(args.original)

    logger.info(f"Loading metadata from {args.metadata}")
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    # Drop _id if not defined in metadata
    metadata_columns = metadata.get("columns", {})
    for df_name, df in [("synthetic", synthetic_data), ("original", original_data)]:
        if "_id" in df.columns and "_id" not in metadata_columns:
            logger.warning(
                f"Column '_id' found in {df_name} data but not in metadata — removing it"
            )
            df.drop(columns=["_id"], inplace=True)

    # Handle both --dimensions and --dimension arguments
    dimensions = args.dimensions
    if args.dimension:
        dimensions = args.dimension

    # Validate utility args if requested
    if "utility" in dimensions:
        if not args.utility_input or not args.utility_output:
            parser.error(
                "Utility evaluation requires both --utility-input and --utility-output arguments."
            )

    # Configure device
    device = configure_device(args.device, args.force_cpu, args.gpu_memory_fraction)
    logger.info(f"Configured device: {device}")

    # Run evaluation
    evaluator = SynEval(synthetic_data, original_data, metadata, device)
    results = {}

    if "fidelity" in dimensions:
        logger.info("Running fidelity evaluation...")
        try:
            results["fidelity"] = evaluator.evaluate_fidelity(
                selected_metrics=args.fidelity_metrics
            )
            logger.info("Fidelity evaluation completed successfully")
        except ImportError as e:
            logger.warning(f"Skipping fidelity evaluation: {e}")
            results["fidelity"] = {"skipped": True, "reason": str(e)}
        except Exception as e:
            logger.error(f"Error in fidelity evaluation: {str(e)}")
            results["fidelity"] = {"error": str(e)}

    if "utility" in dimensions:
        logger.info("Running utility evaluation...")
        try:
            results["utility"] = evaluator.evaluate_utility(
                input_columns=args.utility_input,
                output_columns=args.utility_output,
                selected_metrics=args.utility_metrics,
            )
            logger.info("Utility evaluation completed successfully")
        except ImportError as e:
            logger.warning(f"Skipping utility evaluation: {e}")
            results["utility"] = {"skipped": True, "reason": str(e)}
        except Exception as e:
            logger.error(f"Error in utility evaluation: {str(e)}")
            results["utility"] = {"error": str(e)}

    if "diversity" in dimensions:
        logger.info("Running diversity evaluation...")
        try:
            results["diversity"] = evaluator.evaluate_diversity(
                selected_metrics=args.diversity_metrics
            )
            logger.info("Diversity evaluation completed successfully")
        except ImportError as e:
            logger.warning(f"Skipping diversity evaluation: {e}")
            results["diversity"] = {"skipped": True, "reason": str(e)}
        except Exception as e:
            logger.error(f"Error in diversity evaluation: {str(e)}")
            results["diversity"] = {
                "error": str(e),
                "tabular_diversity": {},
                "text_diversity": {},
            }

    if "privacy" in dimensions:
        logger.info("Running privacy evaluation...")
        try:
            results["privacy"] = evaluator.evaluate_privacy(
                selected_metrics=args.privacy_metrics
            )
            logger.info("Privacy evaluation completed successfully")
        except ImportError as e:
            logger.warning(f"Skipping privacy evaluation: {e}")
            results["privacy"] = {"skipped": True, "reason": str(e)}
        except Exception as e:
            logger.error(f"Error in privacy evaluation: {str(e)}")
            results["privacy"] = {
                "error": str(e),
                "membership_inference": {},
                "exact_matches": {},
            }

    if args.correlation:
        logger.info(f"Running correlation evaluation: {args.correlation}")
        correlation_kwargs = dict(
            text_col=args.correlation_text_col,
            rating_col=args.correlation_rating_col,
            category_col=args.correlation_category_col,
            numeric_col=args.correlation_numeric_col,
            pii_col=args.correlation_pii_col,
            top_n=args.correlation_top_n,
        )
        # semantic_tabular 需要 text_embeddings，暂留接口
        results["correlation"] = evaluator.evaluate_correlation(
            args.correlation, **correlation_kwargs
        )

    # Convert any non-serializable values (e.g., NaN) before saving
    def safe_json(obj):
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception as e:
            logger.error(f"Error serializing results: {str(e)}")
            return {"error": "Failed to serialize results", "original_error": str(e)}

    # Save results
    logger.info(f"Saving results to {args.output}")
    try:
        safe_results = safe_json(results)
        with open(args.output, "w") as f:
            json.dump(safe_results, f, indent=2)
        logger.info(f"Results successfully saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving results to {args.output}: {str(e)}")
        # Try to save to a backup file
        backup_file = f"{args.output}.backup"
        try:
            with open(backup_file, "w") as f:
                json.dump(safe_results, f, indent=2)
            logger.info(f"Results saved to backup file: {backup_file}")
        except Exception as backup_e:
            logger.error(f"Failed to save backup file: {str(backup_e)}")

    # Display results summary
    print("\n📈 Evaluation Results Summary")
    print("=" * 50)
    display_results_summary(results)

    # Plotting if requested
    if getattr(args, "plot", False):
        logger.info("Generating plots for evaluation results...")
        try:
            from plotting import SynEvalPlotter

            plotter = SynEvalPlotter(output_dir="./plots")
            plotter.plot_all_results(results, save_plots=True)
            logger.info("Plots saved to ./plots directory.")
        except ImportError as e:
            logger.warning(f"Plotting module not available: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")

    logger.info("Evaluation complete!")
    return results


def display_results_summary(results: Dict):
    """Display a summary of evaluation results."""

    for dimension, dimension_results in results.items():
        print(f"\n🔍 {dimension.upper()} EVALUATION")
        print("-" * 30)

        if dimension == "fidelity":
            if "diagnostic" in dimension_results and dimension_results["diagnostic"]:
                diag = dimension_results["diagnostic"]
                print(f"Data Validity: {diag.get('Data Validity', 'N/A')}")
                print(f"Data Structure: {diag.get('Data Structure', 'N/A')}")
                print(f"Overall Score: {diag.get('Overall', {}).get('score', 'N/A')}")

            if "numerical_statistics" in dimension_results:
                num_stats = dimension_results["numerical_statistics"]
                if num_stats:
                    fidelity_scores = [
                        col_data.get("overall_fidelity_score", 0)
                        for col_data in num_stats.values()
                        if isinstance(col_data, dict)
                        and "overall_fidelity_score" in col_data
                    ]
                    if fidelity_scores:
                        avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)
                        # Safe formatting for average fidelity
                        if isinstance(avg_fidelity, (int, float)):
                            print(f"Average Numerical Fidelity: {avg_fidelity:.3f}")
                        else:
                            print(f"Average Numerical Fidelity: {avg_fidelity}")

        elif dimension == "utility":
            # Check for tstr_accuracy results first
            if "tstr_accuracy" in dimension_results:
                tstr_results = dimension_results["tstr_accuracy"]
                if (
                    "real_data_model" in tstr_results
                    and "synthetic_data_model" in tstr_results
                ):
                    real_acc = tstr_results["real_data_model"].get("accuracy", 0)
                    syn_acc = tstr_results["synthetic_data_model"].get("accuracy", 0)

                    # Safe formatting for accuracy scores
                    real_acc_formatted = (
                        f"{real_acc:.3f}"
                        if isinstance(real_acc, (int, float))
                        else str(real_acc)
                    )
                    syn_acc_formatted = (
                        f"{syn_acc:.3f}"
                        if isinstance(syn_acc, (int, float))
                        else str(syn_acc)
                    )

                    print(f"Task Type: {tstr_results.get('task_type', 'N/A')}")
                    print(f"Training Size: {tstr_results.get('training_size', 'N/A')}")
                    print(f"Test Size: {tstr_results.get('test_size', 'N/A')}")
                    print(f"Real Data Model Accuracy: {real_acc_formatted}")
                    print(f"Synthetic Data Model Accuracy: {syn_acc_formatted}")

                    # Safe formatting for performance ratio
                    if (
                        isinstance(real_acc, (int, float))
                        and isinstance(syn_acc, (int, float))
                        and real_acc != 0
                    ):
                        ratio = syn_acc / real_acc
                        print(f"Performance Ratio: {ratio:.3f}")
                    else:
                        print("Performance Ratio: N/A (cannot calculate)")
                else:
                    print(
                        "TSTR accuracy results found but missing model performance data"
                    )
            # Fallback: check for direct model results (legacy format)
            elif (
                "real_data_model" in dimension_results
                and "synthetic_data_model" in dimension_results
            ):
                real_acc = dimension_results["real_data_model"].get("accuracy", 0)
                syn_acc = dimension_results["synthetic_data_model"].get("accuracy", 0)

                # Safe formatting for accuracy scores
                real_acc_formatted = (
                    f"{real_acc:.3f}"
                    if isinstance(real_acc, (int, float))
                    else str(real_acc)
                )
                syn_acc_formatted = (
                    f"{syn_acc:.3f}"
                    if isinstance(syn_acc, (int, float))
                    else str(syn_acc)
                )

                print(f"Real Data Model Accuracy: {real_acc_formatted}")
                print(f"Synthetic Data Model Accuracy: {syn_acc_formatted}")

                # Safe formatting for performance ratio
                if (
                    isinstance(real_acc, (int, float))
                    and isinstance(syn_acc, (int, float))
                    and real_acc != 0
                ):
                    ratio = syn_acc / real_acc
                    print(f"Performance Ratio: {ratio:.3f}")
                else:
                    print("Performance Ratio: N/A (cannot calculate)")
            else:
                print("No utility evaluation results found")

        elif dimension == "diversity":
            if "tabular_diversity" in dimension_results:
                tab = dimension_results["tabular_diversity"]

                # Coverage metrics
                if "coverage" in tab and tab["coverage"]:
                    avg_coverage = sum(tab["coverage"].values()) / len(tab["coverage"])
                    # Safe formatting for average coverage
                    if isinstance(avg_coverage, (int, float)):
                        print(f"Average Tabular Coverage: {avg_coverage:.1f}%")
                    else:
                        print(f"Average Tabular Coverage: {avg_coverage}%")

                # Uniqueness metrics
                if "uniqueness" in tab and tab["uniqueness"]:
                    if "duplicate_ratio" in tab["uniqueness"]:
                        dup_ratio = tab["uniqueness"]["duplicate_ratio"]
                        if isinstance(dup_ratio, (int, float)):
                            print(f"Duplicate Ratio: {dup_ratio:.3f}")
                        else:
                            print(f"Duplicate Ratio: {dup_ratio}")

                # Entropy metrics
                if "entropy_metrics" in tab and tab["entropy_metrics"]:
                    if "dataset_entropy" in tab["entropy_metrics"]:
                        entropy_data = tab["entropy_metrics"]["dataset_entropy"]
                        if "entropy_ratio" in entropy_data:
                            entropy_ratio = entropy_data["entropy_ratio"]
                            if isinstance(entropy_ratio, (int, float)):
                                print(f"Entropy Ratio: {entropy_ratio:.3f}")
                            else:
                                print(f"Entropy Ratio: {entropy_ratio}")

            # Text diversity metrics
            if "text_diversity" in dimension_results:
                text_div = dimension_results["text_diversity"]
                if "synthetic" in text_div and text_div["synthetic"]:
                    # Get first text column results
                    first_col = next(iter(text_div["synthetic"]), None)
                    if first_col:
                        col_results = text_div["synthetic"][first_col]

                        # Lexical diversity
                        if "lexical_diversity" in col_results:
                            lex_div = col_results["lexical_diversity"]
                            if (
                                "1-gram" in lex_div
                                and "unique_ratio" in lex_div["1-gram"]
                            ):
                                unique_ratio = lex_div["1-gram"]["unique_ratio"]
                                if isinstance(unique_ratio, (int, float)):
                                    print(f"Text Lexical Diversity: {unique_ratio:.3f}")
                                else:
                                    print(f"Text Lexical Diversity: {unique_ratio}")

                        # Semantic diversity
                        if "semantic_diversity" in col_results:
                            sem_div = col_results["semantic_diversity"]
                            if "distinct_ratio" in sem_div:
                                distinct_ratio = sem_div["distinct_ratio"]
                                if isinstance(distinct_ratio, (int, float)):
                                    print(
                                        f"Text Semantic Diversity: {distinct_ratio:.3f}"
                                    )
                                else:
                                    print(f"Text Semantic Diversity: {distinct_ratio}")

                        # Sentiment diversity
                        if "sentiment_diversity" in col_results:
                            sent_div = col_results["sentiment_diversity"]
                            if "sentiment_alignment_score" in sent_div:
                                alignment_score = sent_div["sentiment_alignment_score"]
                                if isinstance(alignment_score, (int, float)):
                                    print(
                                        f"Sentiment Alignment Score: {alignment_score:.3f}"
                                    )
                                else:
                                    print(
                                        f"Sentiment Alignment Score: {alignment_score}"
                                    )

        elif dimension == "privacy":
            if "membership_inference" in dimension_results:
                mia = dimension_results["membership_inference"]
                print(f"Membership Inference Risk: {mia.get('risk_level', 'N/A')}")

                # Safe formatting for MIA AUC Score
                mia_auc = mia.get("mia_auc_score", None)
                if isinstance(mia_auc, (int, float)):
                    print(f"MIA AUC Score: {mia_auc:.3f}")
                else:
                    print(f"MIA AUC Score: {mia_auc if mia_auc is not None else 'N/A'}")

                # Additional MIA metrics
                syn_confidence = mia.get("synthetic_confidence", None)
                orig_confidence = mia.get("original_confidence", None)
                if isinstance(syn_confidence, (int, float)) and isinstance(
                    orig_confidence, (int, float)
                ):
                    print(f"Synthetic Confidence: {syn_confidence:.3f}")
                    print(f"Original Confidence: {orig_confidence:.3f}")

            if "exact_matches" in dimension_results:
                exact = dimension_results["exact_matches"]
                print(f"Exact Match Risk: {exact.get('risk_level', 'N/A')}")

                # Exact match percentage
                exact_percentage = exact.get("exact_match_percentage", None)
                if isinstance(exact_percentage, (int, float)):
                    print(f"Exact Match Percentage: {exact_percentage:.2f}%")

            # Anonymeter results
            if "anonymeter" in dimension_results:
                anonymeter = dimension_results["anonymeter"]
                if "overall_risk" in anonymeter:
                    overall_risk = anonymeter["overall_risk"]
                    risk_score = overall_risk.get("risk_score", None)
                    risk_level = overall_risk.get("risk_level", "N/A")
                    if isinstance(risk_score, (int, float)):
                        print(f"Anonymeter Overall Risk Score: {risk_score:.3f}")
                    print(f"Anonymeter Risk Level: {risk_level}")

            # Text privacy results
            if "text_privacy" in dimension_results:
                text_priv = dimension_results["text_privacy"]
                if "ner_statistics" in text_priv:
                    ner_stats = text_priv["ner_statistics"]
                    if "entity_density" in ner_stats:
                        entity_density = ner_stats["entity_density"]
                        if isinstance(entity_density, (int, float)):
                            print(f"Entity Density: {entity_density:.3f}")

            # Tabular privacy results
            if "tabular_privacy" in dimension_results:
                tab_priv = dimension_results["tabular_privacy"]
                if "structured_privacy_metrics" in tab_priv:
                    struct_metrics = tab_priv["structured_privacy_metrics"]
                    for metric_name, metric_data in struct_metrics.items():
                        if isinstance(metric_data, dict):
                            if metric_name == "IMS":
                                ims_syn = metric_data.get("ims_syn_train", 0)
                                ims_train_test = metric_data.get("train_test_ims", 0)
                                passed = metric_data.get("passed", False)
                                if isinstance(ims_syn, (int, float)):
                                    print(f"IMS (Synthetic-Train): {ims_syn:.4f}")
                                if isinstance(ims_train_test, (int, float)):
                                    print(f"IMS (Train-Test): {ims_train_test:.4f}")
                                print(f"IMS Passed: {'Yes' if passed else 'No'}")
                            elif metric_name == "DCR":
                                syn_5pct = metric_data.get("syn_train_5pct", 0)
                                train_5pct = metric_data.get("train_train_5pct", 0)
                                passed = metric_data.get("passed", False)
                                if isinstance(syn_5pct, (int, float)):
                                    print(f"DCR (Synthetic 5%): {syn_5pct:.4f}")
                                if isinstance(train_5pct, (int, float)):
                                    print(f"DCR (Train 5%): {train_5pct:.4f}")
                                print(f"DCR Passed: {'Yes' if passed else 'No'}")
                            elif metric_name == "NNDR":
                                syn_5pct = metric_data.get("syn_train_5pct", 0)
                                train_5pct = metric_data.get("train_train_5pct", 0)
                                passed = metric_data.get("passed", False)
                                if isinstance(syn_5pct, (int, float)):
                                    print(f"NNDR (Synthetic 5%): {syn_5pct:.4f}")
                                if isinstance(train_5pct, (int, float)):
                                    print(f"NNDR (Train 5%): {train_5pct:.4f}")
                                print(f"NNDR Passed: {'Yes' if passed else 'No'}")
                            elif "score" in metric_data:
                                score = metric_data["score"]
                                if isinstance(score, (int, float)):
                                    print(
                                        f"{metric_name.replace('_', ' ').title()}: {score:.3f}"
                                    )

        if "correlation" in results:
            print(f"\n🔗 CROSS-MODALITY CORRELATION")
            print("-" * 30)
            for k, v in results["correlation"].items():
                print(f"{k}: {v}")


if __name__ == "__main__":

    main()
