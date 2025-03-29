#!/usr/bin/env python
"""
IRT Analysis CLI for Benchmarks

This script runs Item Response Theory (IRT) analysis on language model benchmark data.
It loads model score data, computes binary matrices, fits IRT models, and analyzes results.


Examples:
    # Run basic analysis with default parameters
    python cli.py --data_path=../data/filtered_model_score_dict.pkl --output_path=../results/

    # Run with classical IRT methods instead of PyMC
    python cli.py --data_path=../data/filtered_model_score_dict.pkl --pymc=False

    # Run with specific model parameters
    python cli.py --model=rasch --n_samples=5000 --chains=8 --cores=4

    # Show help information
    python cli.py --help
"""

import os
import sys
import time

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from irt_on_bench.analysis import (
    identify_misfitting_items,
    item_fit_statistics,
    plot_item_characteristic_curve,
)
from irt_on_bench.analyzer import BenchmarkAnalyzer
from irt_on_bench.data import (
    create_binary_matrix,
    load_model_scores,
    load_trace,
    save_trace,
)
from irt_on_bench.metadata import BinaryQuestionMetadata
from irt_on_bench.models import (
    extract_parameter_uncertainties,
    extract_parameters,
    fit_2pl_pymc,
)
from irt_on_bench.utils import enable_logging


class IRTAnalysis:
    """
    IRT Analysis CLI for language model benchmarks.

    This class provides methods to run Item Response Theory analysis on benchmark data,
    fit IRT models, and analyze results.
    """

    def __init__(self):
        """Initialize the IRT Analysis CLI."""
        # Set default values for attributes
        self.seed = 42
        np.random.seed(self.seed)

    def run(
        self,
        data_path="../data/filtered_model_score_dict.pkl",
        output_path="../results/",
        model="2pl",
        category="overall",
        pymc=True,
        n_samples=2000,
        tune=2000,
        chains=4,
        cores=1,
        target_accept=0.8,
        check_existing=False,
        force=False,
        seed=42,
    ):
        """
        Run IRT analysis on benchmark data.

        Args:
            data_path (str): Path to the pickle file containing model scores.
            output_path (str): Path to save results.
            model (str): IRT model to fit ('2pl' or 'rasch').
            category (str): Category of scores to analyze.
            pymc (bool): Use PyMC for Bayesian IRT model fitting.
            n_samples (int): Number of MCMC samples (for PyMC).
            tune (int): Number of tuning samples (for PyMC).
            chains (int): Number of MCMC chains (for PyMC).
            cores (int): Number of CPU cores to use for sampling.
            target_accept (float): Target acceptance rate for MCMC.
            check_existing (bool): Check for existing output files and ask before overwriting.
            force (bool): Force overwrite existing output files.
            seed (int): Random seed for reproducibility.

        Returns:
            None
        """
        # Record start time
        start_time = time.time()

        # Set random seed
        if seed != self.seed:
            self.seed = seed
            np.random.seed(self.seed)

        logger.info(f"Starting IRT analysis with model={model}, pymc={pymc}")

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Check if output files already exist
        if check_existing and not self._check_output_files(output_path, force):
            sys.exit(0)

        logger.info(f"Loading model scores from {data_path}")
        # Load model scores
        model_score_dicts = load_model_scores(data_path)

        # Create binary matrix
        binary_array, binary_df, models = create_binary_matrix(
            model_score_dicts, category=category
        )

        # Print sizes for debugging
        logger.info(f"Number of models: {len(models)}")
        logger.info(f"Binary matrix shape: {binary_array.shape}")

        # Save binary matrix to CSV
        binary_df.to_csv(os.path.join(output_path, "binary_matrix.csv"))

        # Check for existing trace file
        trace_path = os.path.join(output_path, "trace_all_correct_filtered.pkl")
        trace_exists = os.path.exists(trace_path)

        if pymc:
            if trace_exists and not force:
                logger.info(f"Loading existing trace from {trace_path}")
                trace = load_trace(trace_path)
            else:
                logger.info(
                    f"Fitting PyMC 2PL model with {n_samples} samples and {chains} chains..."
                )
                # Fit 2PL model using PyMC
                trace = fit_2pl_pymc(
                    binary_array.T,  # Transpose to [n_participants, n_items]
                    n_samples=n_samples,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    cores=cores,
                )

                # Save trace
                save_trace(trace, trace_path)
                logger.info(f"Saved trace to {trace_path}")

            # Extract parameters and uncertainties
            parameters = extract_parameters(trace)
            uncertainties = extract_parameter_uncertainties(trace)

            # Extract the parameters we need for analysis
            difficulties = parameters["difficulties"]
            discriminations = parameters["discriminations"]
            abilities = parameters["abilities"]
            abilities_std = uncertainties["abilities"]["std"]

        else:
            logger.info(f"Fitting classical {model.upper()} model...")
            # Use BenchmarkAnalyzer for classical IRT
            analyzer = BenchmarkAnalyzer()

            # Create question metadata
            n_questions = binary_array.shape[0]
            for i in range(n_questions):
                analyzer.add_question_metadata(BinaryQuestionMetadata(f"q{i}"))

            # Add model results
            for i, model_name in enumerate(models):
                # Create a DataFrame with the binary scores for each question
                model_df = pd.DataFrame(
                    {"all_correct_": binary_array[:, i]},
                    index=[f"q{j}" for j in range(n_questions)],
                )

                analyzer.add_model_results(model_name, model_df)

            # Fit IRT model
            logger.info("Computing score matrix and fitting IRT model...")
            start_fitting = time.time()
            irt_results = analyzer.fit_irt(model=model)
            logger.info(
                f"IRT model fitting completed in {time.time() - start_fitting:.2f} seconds"
            )

            # Extract parameters
            difficulties = irt_results["difficulties"]
            discriminations = irt_results["discriminations"]
            abilities = irt_results["abilities"]
            abilities_std = np.zeros_like(
                abilities
            )  # No uncertainties for classical methods

        # Analyze items and save results
        self._analyze_and_save_results(
            output_path,
            binary_array,
            difficulties,
            discriminations,
            abilities,
            abilities_std,
            models,
        )

        # Record end time and report
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis complete! Total time: {elapsed_time:.2f} seconds")

        return {
            "models_analyzed": len(models),
            "items_analyzed": binary_array.shape[0],
            "elapsed_time": elapsed_time,
        }

    def _check_output_files(self, output_path, force=False):
        """
        Check if output files already exist and ask for confirmation before overwriting.

        Args:
            output_path (str): Path to the output directory
            force (bool): Force overwrite existing output files

        Returns:
            bool: True if it's safe to proceed with analysis, False otherwise
        """
        # Define expected output files
        expected_files = [
            os.path.join(output_path, "binary_matrix.csv"),
            os.path.join(output_path, "extreme_items.csv"),
            os.path.join(output_path, "misfitting_items.csv"),
            os.path.join(output_path, "item_statistics.csv"),
            os.path.join(output_path, "model_abilities.csv"),
            os.path.join(output_path, "item_characteristic_curves.png"),
            os.path.join(output_path, "trace_all_correct_filtered.pkl"),
        ]

        # Check if any of the expected files exist
        existing_files = [f for f in expected_files if os.path.exists(f)]

        if existing_files and not force:
            logger.warning("The following output files already exist:")
            for f in existing_files:
                logger.warning(f"  - {f}")

            while True:
                response = input("\nOverwrite these files? [y/N]: ").lower()
                if response in ["y", "yes"]:
                    return True
                elif response in ["", "n", "no"]:
                    logger.info(
                        "Analysis aborted. Use --force to overwrite files without prompting."
                    )
                    return False
                else:
                    print("Please answer 'y' or 'n'.")

        return True

    def _analyze_and_save_results(
        self,
        output_path,
        binary_array,
        difficulties,
        discriminations,
        abilities,
        abilities_std,
        models,
    ):
        """
        Analyze IRT results and save outputs.

        Args:
            output_path (str): Path to save results
            binary_array (np.ndarray): Binary response matrix
            difficulties (np.ndarray): Item difficulties
            discriminations (np.ndarray): Item discriminations
            abilities (np.ndarray): Model abilities
            abilities_std (np.ndarray): Standard errors of abilities
            models (list): list of model names
        """
        # Analyze extreme items
        logger.info("Analyzing extreme items...")
        question_ids = [f"q{i}" for i in range(len(difficulties))]
        extreme_items = BenchmarkAnalyzer.analyze_extreme_items(
            difficulties, discriminations, question_ids
        )

        # Save extreme items to CSV
        extreme_items.to_csv(os.path.join(output_path, "extreme_items.csv"))
        logger.info(f"Found {len(extreme_items)} extreme items")

        # Calculate item fit statistics
        logger.info("Calculating item fit statistics...")
        item_stats = item_fit_statistics(
            binary_array,  # [n_items, n_participants]
            abilities,
            difficulties,
            discriminations,
        )

        # Identify misfitting items
        misfitting_items = identify_misfitting_items(item_stats)
        misfitting_items.to_csv(os.path.join(output_path, "misfitting_items.csv"))
        logger.info(f"Found {len(misfitting_items)} misfitting items")

        # Save all item statistics
        item_stats.to_csv(os.path.join(output_path, "item_statistics.csv"))

        # Save model parameters
        model_abilities = pd.DataFrame(
            {"model": models, "ability": abilities, "std_error": abilities_std}
        )
        model_abilities.to_csv(os.path.join(output_path, "model_abilities.csv"))
        logger.info(
            f"Saved model abilities to {os.path.join(output_path, 'model_abilities.csv')}"
        )

        # Plot item characteristic curves for interesting items
        self._plot_item_curves(
            output_path, extreme_items, misfitting_items, difficulties, discriminations
        )

    def _plot_item_curves(
        self,
        output_path,
        extreme_items,
        misfitting_items,
        difficulties,
        discriminations,
    ):
        """
        Plot item characteristic curves for interesting items.

        Args:
            output_path (str): Path to save results
            extreme_items (pd.DataFrame): DataFrame of extreme items
            misfitting_items (pd.DataFrame): DataFrame of misfitting items
            difficulties (np.ndarray): Item difficulties
            discriminations (np.ndarray): Item discriminations
        """
        logger.info("Plotting item characteristic curves...")
        interesting_items = pd.concat(
            [extreme_items.head(5), misfitting_items.head(5)]
        ).drop_duplicates(subset=["item_id"])

        fig, axes = plt.subplots(
            nrows=len(interesting_items), figsize=(10, 4 * len(interesting_items))
        )

        if len(interesting_items) == 1:
            axes = [axes]

        for i, (_, item) in enumerate(interesting_items.iterrows()):
            plot_item_characteristic_curve(
                item["difficulty"], item["discrimination"], ax=axes[i]
            )
            axes[i].set_title(
                f"Item {item['item_id']} (Difficulty: {item['difficulty']:.2f}, "
                f"Discrimination: {item['discrimination']:.2f})"
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "item_characteristic_curves.png"))
        logger.info(
            f"Saved item characteristic curves to {os.path.join(output_path, 'item_characteristic_curves.png')}"
        )

def setup_analyzer_from_scores(
    scores_path: str, 
    metric_name: str,
    question_ids: list[str] | None = None,
    metadata_dict: dict[str, dict] | None = None
) -> tuple[BenchmarkAnalyzer, list[str]]:
    """
    Creates and sets up a BenchmarkAnalyzer from model scores file.
    
    Args:
        scores_path: Path to the scores pickle file
        question_ids: Optional list of question IDs. If None, will generate IDs like "q0", "q1", etc.
        metadata_dict: Optional dictionary of question metadata {question_id: {metadata_key: value}}
    
    Returns:
        Tuple of (configured analyzer, list of model names)
    """
    # Load model scores
    model_scores = load_model_scores(scores_path)
    
    # Create binary matrix
    binary_array, binary_df, models = create_binary_matrix(model_scores)
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer()
    
    # Generate question IDs if not provided
    if question_ids is None:
        question_ids = [f"q{i}" for i in range(binary_array.shape[0])]
    
    # Add question metadata
    for i, q_id in enumerate(question_ids):
        # Create metadata object with basic ID
        metadata = BinaryQuestionMetadata(q_id)
        
        # Add any additional metadata if provided
        if metadata_dict and q_id in metadata_dict:
            for key, value in metadata_dict[q_id].items():
                setattr(metadata, key, value)
        
        analyzer.add_question_metadata(metadata)
    
    # Add model results
    for i, model in enumerate(models):
        model_df = pd.DataFrame({
            metric_name : binary_array[:, i]
        }, index=question_ids)
        analyzer.add_model_results(model, model_df)
    
    return analyzer, models


def main():
    """Main function to create and run IRT Analysis CLI."""
    enable_logging()
    fire.Fire(IRTAnalysis().run, name="IRT Analysis CLI")


if __name__ == "__main__":
    main()
