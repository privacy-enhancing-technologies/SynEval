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
        
        self.fidelity_evaluator = FidelityEvaluator(synthetic_data, original_data, metadata)
        self.diversity_evaluator = DiversityEvaluator(synthetic_data, original_data, metadata)
        self.privacy_evaluator = PrivacyEvaluator(synthetic_data, original_data, metadata)

    def evaluate_fidelity(self) -> Dict:
        return self.fidelity_evaluator.evaluate()

    def evaluate_utility(self, input_columns: List[str], output_columns: List[str]) -> Dict:
        utility_evaluator = UtilityEvaluator(
            self.synthetic_data,
            self.original_data,
            self.metadata,
            input_columns=input_columns,
            output_columns=output_columns
        )
        return utility_evaluator.evaluate()

    def evaluate_diversity(self) -> Dict:
        return self.diversity_evaluator.evaluate()

    def evaluate_privacy(self) -> Dict:
        return self.privacy_evaluator.evaluate()

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Data Evaluation Framework')

    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--original', required=True, help='Path to original data CSV file')
    parser.add_argument('--metadata', required=True, help='Path to metadata JSON file')

    parser.add_argument('--dimensions', nargs='+', 
                        choices=['fidelity', 'utility', 'diversity', 'privacy'],
                        default=['fidelity', 'utility', 'diversity', 'privacy'],
                        help='Evaluation dimensions to run (default: all)')

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

    # Validate utility args if requested
    if 'utility' in args.dimensions:
        if not args.utility_input or not args.utility_output:
            parser.error("Utility evaluation requires both --utility-input and --utility-output arguments.")

    # Run evaluation
    evaluator = SynEval(synthetic_data, original_data, metadata)
    results = {}

    if 'fidelity' in args.dimensions:
        logger.info("Running fidelity evaluation...")
        results['fidelity'] = evaluator.evaluate_fidelity()

    if 'utility' in args.dimensions:
        logger.info("Running utility evaluation...")
        results['utility'] = evaluator.evaluate_utility(
            input_columns=args.utility_input,
            output_columns=args.utility_output
        )

    if 'diversity' in args.dimensions:
        logger.info("Running diversity evaluation...")
        results['diversity'] = evaluator.evaluate_diversity()

    if 'privacy' in args.dimensions:
        logger.info("Running privacy evaluation...")
        results['privacy'] = evaluator.evaluate_privacy()

    # Convert any non-serializable values (e.g., NaN) before saving
    def safe_json(obj):
        return json.loads(json.dumps(obj, default=str))

    # Save results
    logger.info(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(safe_json(results), f, indent=2)

    # Plotting if requested
    if getattr(args, 'plot', False):
        logger.info("Generating plots for evaluation results...")
        from plotting import SynEvalPlotter
        plotter = SynEvalPlotter(output_dir='./plots')
        plotter.plot_all_results(results, save_plots=True)
        logger.info("Plots saved to ./plots directory.")

    logger.info("Evaluation complete!")

if __name__ == "__main__":

    main()