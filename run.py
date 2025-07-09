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


class SynEval:
    def __init__(self, synthetic_data: pd.DataFrame, original_data: pd.DataFrame, metadata: Dict):
        self.synthetic_data = synthetic_data
        self.original_data = original_data
        self.metadata = metadata
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluators as None - will be created when needed
        self._fidelity_evaluator = None
        self._diversity_evaluator = None
        self._privacy_evaluator = None
        self._utility_evaluator = None

    def evaluate_fidelity(self) -> Dict:
        if self._fidelity_evaluator is None:
            self._fidelity_evaluator = FidelityEvaluator(self.synthetic_data, self.original_data, self.metadata)
        return self._fidelity_evaluator.evaluate()

    def evaluate_utility(self, input_columns: List[str], output_columns: List[str]) -> Dict:
        # Utility evaluator is created fresh each time since it needs input/output columns
        utility_evaluator = UtilityEvaluator(
            self.synthetic_data,
            self.original_data,
            self.metadata,
            input_columns=input_columns,
            output_columns=output_columns
        )
        return utility_evaluator.evaluate()

    def evaluate_diversity(self) -> Dict:
        if self._diversity_evaluator is None:
            self._diversity_evaluator = DiversityEvaluator(self.synthetic_data, self.original_data, self.metadata)
        return self._diversity_evaluator.evaluate()

    def evaluate_privacy(self) -> Dict:
        if self._privacy_evaluator is None:
            self._privacy_evaluator = PrivacyEvaluator(self.synthetic_data, self.original_data, self.metadata)
        return self._privacy_evaluator.evaluate()

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Data Evaluation Framework')

    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--original', required=True, help='Path to original data CSV file')
    parser.add_argument('--metadata', required=True, help='Path to metadata JSON file')

    parser.add_argument('--dimensions', nargs='+', 
                        choices=['fidelity', 'utility', 'diversity', 'privacy'],
                        default=['fidelity', 'utility', 'diversity', 'privacy'],
                        help='Evaluation dimensions to run (default: all)')
    parser.add_argument('--dimension', nargs='+', 
                        choices=['fidelity', 'utility', 'diversity', 'privacy'],
                        help='Alias for --dimensions (singular form)')

    parser.add_argument('--utility-input', nargs='+', help='Input columns for utility evaluation')
    parser.add_argument('--utility-output', nargs='+', help='Output columns for utility evaluation')

    parser.add_argument('--output', default='./evaluation_results.json', help='Path to save evaluation results')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
    parser.add_argument('--plot', action='store_true', help='Generate plots for all evaluation metrics and save them to the ./plots directory')
    
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load data
    logger.info(f"Loading synthetic data from {args.synthetic}")
    synthetic_data = pd.read_csv(args.synthetic)

    logger.info(f"Loading original data from {args.original}")
    original_data = pd.read_csv(args.original)

    logger.info(f"Loading metadata from {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)

    # Drop _id if not defined in metadata
    metadata_columns = metadata.get("columns", {})
    for df_name, df in [('synthetic', synthetic_data), ('original', original_data)]:
        if '_id' in df.columns and '_id' not in metadata_columns:
            logger.warning(f"Column '_id' found in {df_name} data but not in metadata — removing it")
            df.drop(columns=['_id'], inplace=True)

    # Handle both --dimensions and --dimension arguments
    dimensions = args.dimensions
    if args.dimension:
        dimensions = args.dimension
    
    # Validate utility args if requested
    if 'utility' in dimensions:
        if not args.utility_input or not args.utility_output:
            parser.error("Utility evaluation requires both --utility-input and --utility-output arguments.")

    # Run evaluation
    evaluator = SynEval(synthetic_data, original_data, metadata)
    results = {}

    if 'fidelity' in dimensions:
        logger.info("Running fidelity evaluation...")
        results['fidelity'] = evaluator.evaluate_fidelity()

    if 'utility' in dimensions:
        logger.info("Running utility evaluation...")
        results['utility'] = evaluator.evaluate_utility(
            input_columns=args.utility_input,
            output_columns=args.utility_output
        )

    if 'diversity' in dimensions:
        logger.info("Running diversity evaluation...")
        try:
            results['diversity'] = evaluator.evaluate_diversity()
            logger.info("Diversity evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error in diversity evaluation: {str(e)}")
            results['diversity'] = {
                'error': str(e),
                'tabular_diversity': {},
                'text_diversity': {}
            }

    if 'privacy' in dimensions:
        logger.info("Running privacy evaluation...")
        try:
            results['privacy'] = evaluator.evaluate_privacy()
            logger.info("Privacy evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error in privacy evaluation: {str(e)}")
            results['privacy'] = {
                'error': str(e),
                'membership_inference': {},
                'exact_matches': {}
            }

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
        with open(args.output, 'w') as f:
            json.dump(safe_results, f, indent=2)
        logger.info(f"Results successfully saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving results to {args.output}: {str(e)}")
        # Try to save to a backup file
        backup_file = f"{args.output}.backup"
        try:
            with open(backup_file, 'w') as f:
                json.dump(safe_results, f, indent=2)
            logger.info(f"Results saved to backup file: {backup_file}")
        except Exception as backup_e:
            logger.error(f"Failed to save backup file: {str(backup_e)}")

    # Display results summary
    print("\n📈 Evaluation Results Summary")
    print("=" * 50)
    display_results_summary(results)

    # Plotting if requested
    if getattr(args, 'plot', False):
        logger.info("Generating plots for evaluation results...")
        try:
            from plotting import SynEvalPlotter
            plotter = SynEvalPlotter(output_dir='./plots')
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
        
        if dimension == 'fidelity':
            if 'diagnostic' in dimension_results and dimension_results['diagnostic']:
                diag = dimension_results['diagnostic']
                print(f"Data Validity: {diag.get('Data Validity', 'N/A')}")
                print(f"Data Structure: {diag.get('Data Structure', 'N/A')}")
                print(f"Overall Score: {diag.get('Overall', {}).get('score', 'N/A')}")
            
            if 'numerical_statistics' in dimension_results:
                num_stats = dimension_results['numerical_statistics']
                if num_stats:
                    fidelity_scores = [col_data.get('overall_fidelity_score', 0) 
                                     for col_data in num_stats.values() 
                                     if isinstance(col_data, dict) and 'overall_fidelity_score' in col_data]
                    if fidelity_scores:
                        avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)
                        print(f"Average Numerical Fidelity: {avg_fidelity:.3f}")
        
        elif dimension == 'utility':
            if 'real_data_model' in dimension_results and 'synthetic_data_model' in dimension_results:
                real_acc = dimension_results['real_data_model'].get('accuracy', 0)
                syn_acc = dimension_results['synthetic_data_model'].get('accuracy', 0)
                print(f"Real Data Model Accuracy: {real_acc:.3f}")
                print(f"Synthetic Data Model Accuracy: {syn_acc:.3f}")
                print(f"Performance Ratio: {syn_acc/real_acc:.3f}")
        
        elif dimension == 'diversity':
            if 'tabular_diversity' in dimension_results:
                tab = dimension_results['tabular_diversity']
                if 'coverage' in tab:
                    avg_coverage = sum(tab['coverage'].values()) / len(tab['coverage']) if tab['coverage'] else 0
                    print(f"Average Tabular Coverage: {avg_coverage:.1f}%")
        
        elif dimension == 'privacy':
            if 'membership_inference' in dimension_results:
                mia = dimension_results['membership_inference']
                print(f"Membership Inference Risk: {mia.get('risk_level', 'N/A')}")
                print(f"MIA AUC Score: {mia.get('mia_auc_score', 'N/A'):.3f}")
            
            if 'exact_matches' in dimension_results:
                exact = dimension_results['exact_matches']
                print(f"Exact Match Risk: {exact.get('risk_level', 'N/A')}")

if __name__ == "__main__":

    main()