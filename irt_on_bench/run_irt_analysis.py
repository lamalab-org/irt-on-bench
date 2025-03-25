#!/usr/bin/env python
"""
Main script to run IRT analysis on language model benchmark data.

This script loads model score data, computes binary matrices,
fits IRT models, and analyzes the results.

Usage:
    python run_irt_analysis.py --data_path=../data/filtered_model_score_dict.pkl --output_path=../results/

Example:
    python run_irt_analysis.py --pymc --check_existing
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from irt_on_bench.data_loader import load_model_scores, create_binary_matrix, save_trace, load_trace
from irt_on_bench.models.metadata import BinaryQuestionMetadata
from irt_on_bench.models.analyzer import BenchmarkAnalyzer
from irt_on_bench.models.irt_models import (
    fit_2pl_pymc, check_model_diagnostics, extract_parameters, extract_parameter_uncertainties
)
from irt_on_bench.fit_analysis import item_fit_statistics, identify_misfitting_items, \
    plot_item_characteristic_curve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run IRT analysis on benchmark data')
    parser.add_argument('--data_path', type=str, default='../data/filtered_model_score_dict.pkl',
                        help='Path to the pickle file containing model scores')
    parser.add_argument('--output_path', type=str, default='../results/',
                        help='Path to save results')
    parser.add_argument('--model', type=str, default='2pl', choices=['2pl', 'rasch'],
                        help='IRT model to fit (default: 2pl)')
    parser.add_argument('--category', type=str, default='overall',
                        help='Category of scores to analyze (default: overall)')
    parser.add_argument('--no-pymc', dest='pymc', action='store_false',
                        help='Disable PyMC for Bayesian IRT model fitting (use classical methods instead)')
    parser.set_defaults(pymc=True)
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='Number of MCMC samples (for PyMC, default: 2000)')
    parser.add_argument('--tune', type=int, default=2000,
                        help='Number of tuning samples (for PyMC, default: 2000)')
    parser.add_argument('--chains', type=int, default=4,
                        help='Number of MCMC chains (for PyMC, default: 4)')
    parser.add_argument('--cores', type=int, default=1,
                        help='Number of CPU cores to use for sampling (default: 1)')
    parser.add_argument('--target_accept', type=float, default=0.8,
                        help='Target acceptance rate for MCMC (default: 0.8)')
    parser.add_argument('--check_existing', action='store_true',
                        help='Check for existing output files and ask before overwriting')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing output files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


def check_output_files(output_path, args):
    """
    Check if output files already exist and ask for confirmation before overwriting.

    Parameters
    ----------
    output_path : str
        Path to the output directory
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    bool
        True if it's safe to proceed with analysis, False otherwise
    """
    # Define expected output files
    expected_files = [
        os.path.join(output_path, 'binary_matrix.csv'),
        os.path.join(output_path, 'extreme_items.csv'),
        os.path.join(output_path, 'misfitting_items.csv'),
        os.path.join(output_path, 'item_statistics.csv'),
        os.path.join(output_path, 'model_abilities.csv'),
        os.path.join(output_path, 'item_characteristic_curves.png')
    ]

    if args.pymc:
        expected_files.append(os.path.join(output_path, 'trace_all_correct_filtered.pkl'))

    # Check if any of the expected files exist
    existing_files = [f for f in expected_files if os.path.exists(f)]

    if existing_files and not args.force:
        logger.warning("The following output files already exist:")
        for f in existing_files:
            logger.warning(f"  - {f}")

        if args.check_existing:
            while True:
                response = input("\nOverwrite these files? [y/N]: ").lower()
                if response in ['y', 'yes']:
                    return True
                elif response in ['', 'n', 'no']:
                    logger.info("Analysis aborted. Use --force to overwrite files without prompting.")
                    return False
                else:
                    print("Please answer 'y' or 'n'.")
        else:
            # If not checking interactively, just log a warning
            logger.warning("Using existing files may lead to inconsistent results.")
            logger.warning("Use --check_existing to confirm overwrite or --force to overwrite without prompting.")

    return True


def main():
    """Main function to run the analysis."""
    # Record start time
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    logger.info(f"Starting IRT analysis with model={args.model}, pymc={args.pymc}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Check if output files already exist
    if args.check_existing and not check_output_files(args.output_path, args):
        sys.exit(0)

    logger.info(f"Loading model scores from {args.data_path}")
    # Load model scores
    model_score_dicts = load_model_scores(args.data_path)

    # Create binary matrix
    binary_array, binary_df, models = create_binary_matrix(
        model_score_dicts,
        category=args.category
    )

    # Print sizes for debugging
    logger.info(f"Number of models: {len(models)}")
    logger.info(f"Binary matrix shape: {binary_array.shape}")

    # Save binary matrix to CSV
    binary_df.to_csv(os.path.join(args.output_path, 'binary_matrix.csv'))

    # Check for existing trace file
    trace_path = os.path.join(args.output_path, 'trace_all_correct_filtered.pkl')
    trace_exists = os.path.exists(trace_path)

    if args.pymc:
        if trace_exists and not args.force:
            logger.info(f"Loading existing trace from {trace_path}")
            trace = load_trace(trace_path)
        else:
            logger.info(f"Fitting PyMC 2PL model with {args.n_samples} samples and {args.chains} chains...")
            # Fit 2PL model using PyMC
            trace = fit_2pl_pymc(
                binary_array.T,  # Transpose to [n_participants, n_items]
                n_samples=args.n_samples,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                cores=args.cores
            )

            # Save trace
            save_trace(trace, trace_path)
            logger.info(f"Saved trace to {trace_path}")

        # Check model diagnostics
        diagnostics = check_model_diagnostics(trace)

        # Extract parameters and uncertainties
        parameters = extract_parameters(trace)
        uncertainties = extract_parameter_uncertainties(trace)

        # Extract the parameters we need for analysis
        difficulties = parameters['difficulties']
        discriminations = parameters['discriminations']
        abilities = parameters['abilities']
        abilities_std = uncertainties['abilities']['std']

    else:
        logger.info(f"Fitting classical {args.model.upper()} model...")
        # Use BenchmarkAnalyzer for classical IRT
        analyzer = BenchmarkAnalyzer()

        # Create question metadata (using BinaryQuestionMetadata for all questions)
        n_questions = binary_array.shape[0]
        for i in range(n_questions):
            analyzer.add_question_metadata(BinaryQuestionMetadata(f"q{i}"))

        # Add model results (using binary_array directly)
        for i, model in enumerate(models):
            # Create a DataFrame with the binary scores for each question
            model_df = pd.DataFrame({
                'all_correct_': binary_array[:, i]
            }, index=[f"q{j}" for j in range(n_questions)])

            analyzer.add_model_results(model, model_df)

        # Fit IRT model
        logger.info("Computing score matrix and fitting IRT model...")
        start_fitting = time.time()
        irt_results = analyzer.fit_irt(model=args.model)
        logger.info(f"IRT model fitting completed in {time.time() - start_fitting:.2f} seconds")

        # Extract parameters
        difficulties = irt_results['difficulties']
        discriminations = irt_results['discriminations']
        abilities = irt_results['abilities']
        abilities_std = np.zeros_like(abilities)  # No uncertainties for classical methods

    # Analyze extreme items
    logger.info("Analyzing extreme items...")
    question_ids = [f"q{i}" for i in range(len(difficulties))]
    extreme_items = BenchmarkAnalyzer.analyze_extreme_items(
        difficulties,
        discriminations,
        question_ids
    )

    # Save extreme items to CSV
    extreme_items.to_csv(os.path.join(args.output_path, 'extreme_items.csv'))
    logger.info(f"Found {len(extreme_items)} extreme items")

    # Calculate item fit statistics
    logger.info("Calculating item fit statistics...")
    item_stats = item_fit_statistics(
        binary_array,  # [n_items, n_participants]
        abilities,
        difficulties,
        discriminations
    )

    # Identify misfitting items
    misfitting_items = identify_misfitting_items(item_stats)
    misfitting_items.to_csv(os.path.join(args.output_path, 'misfitting_items.csv'))
    logger.info(f"Found {len(misfitting_items)} misfitting items")

    # Save all item statistics
    item_stats.to_csv(os.path.join(args.output_path, 'item_statistics.csv'))

    # Save model parameters
    model_abilities = pd.DataFrame({
        'model': models,
        'ability': abilities,
        'std_error': abilities_std
    })
    model_abilities.to_csv(os.path.join(args.output_path, 'model_abilities.csv'))
    logger.info(f"Saved model abilities to {os.path.join(args.output_path, 'model_abilities.csv')}")

    # Plot item characteristic curves for a few interesting items
    logger.info("Plotting item characteristic curves...")
    interesting_items = pd.concat([
        extreme_items.head(5),
        misfitting_items.head(5)
    ]).drop_duplicates(subset=['item_id'])

    fig, axes = plt.subplots(nrows=len(interesting_items), figsize=(10, 4 * len(interesting_items)))

    if len(interesting_items) == 1:
        axes = [axes]

    for i, (_, item) in enumerate(interesting_items.iterrows()):
        plot_item_characteristic_curve(
            item['difficulty'],
            item['discrimination'],
            ax=axes[i]
        )
        axes[i].set_title(f"Item {item['item_id']} (Difficulty: {item['difficulty']:.2f}, "
                          f"Discrimination: {item['discrimination']:.2f})")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'item_characteristic_curves.png'))
    logger.info(
        f"Saved item characteristic curves to {os.path.join(args.output_path, 'item_characteristic_curves.png')}")

    # Record end time and report
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis complete! Total time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()