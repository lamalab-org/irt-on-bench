#!/usr/bin/env python
"""
Combined script for analyzing item difficulty parameters and comparing them to human scores.

This script:
1. Analyzes item difficulty parameters from a PyMC trace and compares with human labels
2. Creates a scatter plot comparing item difficulties with mean human scores

Usage:
    python combined_difficulty_analysis.py --trace_path=../data/trace_all_correct_filtered.pkl
                                         --difficulty_json=../data/difficulty_dict.json
                                         --human_file=../data/humans_as_models_scores_combined.pkl
                                         --filtered_file=../data/filtered_model_score_dict.pkl
                                         --output_dir=pdf
"""

import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.constants import golden

from irt_on_bench.data import load_trace

# Plotting constants
ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden


def range_frame(ax, x, y, pad=0.1):
    """
    Create a range frame for better visualization.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to modify
    x : array-like
        x-values to set boundaries
    y : array-like
        y-values to set boundaries
    pad : float, optional
        Padding around the data (default: 0.1)
    """
    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)

    ax.set_ylim(y_min - pad * (y_max - y_min),
                y_max + pad * (y_max - y_min))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_bounds(y_min, y_max)
    ax.spines['bottom'].set_bounds(x_min, x_max)


def plot_difficulty_swarm(df, output_path):
    """
    Create a swarm plot of item difficulties by human-assigned categories.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'difficulty' and 'difficulty_level' columns
    output_path : str
        Path to save the figure
    """
    plt.figure(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH / 2), facecolor='white')

    # Create swarm plot with much smaller points
    ax = sns.swarmplot(data=df, x='difficulty_level', y='difficulty', color="#6A040F", size=1)

    # Apply range frame
    x = df['difficulty_level'].unique()
    y = df['difficulty'].values
    range_frame(ax, x, y)

    # Customize labels
    ax.set_xticklabels(['Easy', 'Intermediate', 'Hard'])

    plt.xlabel('Human Assigned')
    plt.ylabel('Computed Difficulty')
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')

    print(f"Figure saved to {output_path}")


def prepare_difficulty_dataframe(trace, difficulty_dict):
    """
    Prepare a DataFrame with difficulty values and human labels.

    Parameters
    ----------
    trace : arviz.InferenceData
        PyMC trace with posterior samples
    difficulty_dict : dict
        Dictionary mapping difficulty levels to item indices

    Returns
    -------
    pandas.DataFrame
        DataFrame with difficulty values and human-assigned levels
    """
    # Extract difficulty values from trace
    difficulty_vector = trace.posterior['difficulties'].mean(dim=['chain', 'draw']).values

    # Create DataFrame with difficulty values
    difficulty_df = pd.DataFrame(difficulty_vector, columns=['difficulty'])

    # Create mapping dictionary with default value (2 = 'hard')
    level_map = {i: 2 for i in range(len(difficulty_vector))}

    # Map easy indices to 0
    for i in difficulty_dict['easy']:
        level_map[i] = 0

    # Map intermediate indices to 1
    for i in difficulty_dict['intermediate']:
        level_map[i] = 1

    # Add difficulty level column
    difficulty_df['difficulty_level'] = difficulty_df.index.map(level_map)

    return difficulty_df, difficulty_vector


def plot_difficulty_vs_mean_scores(human_file, filtered_file, difficulty_vector, output_path):
    """
    Create a scatter plot comparing item difficulties with mean human scores.

    Parameters
    ----------
    human_file : str
        Path to the human scores pickle file
    filtered_file : str
        Path to the filtered model scores pickle file
    difficulty_vector : np.ndarray
        Array of item difficulty parameters
    output_path : str
        Path to save the figure
    """
    # Load files
    with open(human_file, 'rb') as f:
        human_data = pickle.load(f)

    with open(filtered_file, 'rb') as f:
        filtered_data = pickle.load(f)

    # Get scores for each human
    all_scores = []
    for human_id in human_data['raw_scores'].keys():
        human_df = human_data['raw_scores'][human_id]
        scores = pd.DataFrame(index=human_df.index)
        scores['all_correct'] = human_df['all_correct_'].astype(int)
        all_scores.append(scores)

    combined_scores = pd.concat(all_scores, axis=1)
    mean_scores = combined_scores.mean(axis=1)

    # Get Mistral data and find matching indices
    mistral_df = pd.DataFrame(filtered_data['overall']['Mistral-Large-2'])
    human_indices = mean_scores.index
    mistral_indices = mistral_df['Unnamed: 0'].values
    matching_indices = [idx for idx, val in enumerate(human_indices) if val in mistral_indices]

    # Get matching difficulties and scores
    matched_difficulties = difficulty_vector[matching_indices]
    matched_scores = mean_scores.iloc[matching_indices]

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, 1.2 * ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Remove grid
    ax.grid(False)

    # Create scatter plot with improved styling
    ax.scatter(matched_difficulties, matched_scores, alpha=0.5, color='#6A040F')

    # Add trend line
    z = np.polyfit(matched_difficulties, matched_scores, 1)
    p = np.poly1d(z)
    ax.plot(matched_difficulties, p(matched_difficulties), "--", color='#6A040F', alpha=0.8)

    # Calculate and add correlation
    correlation = np.corrcoef(matched_difficulties, matched_scores)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add margins to create space between points and axes
    ax.margins(x=0.1, y=0.1)

    # Customize labels
    ax.set_xlabel('Question Difficulty (IRT)')
    ax.set_ylabel('Average Human Score')

    # Apply range frame
    range_frame(ax, matched_difficulties, matched_scores)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, format='pdf')

    print(f"Figure saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze item difficulties and compare with human scores')
    parser.add_argument('--trace_path', type=str,
                        default='../results/trace_all_correct_filtered.pkl',
                        help='Path to the PyMC trace pickle file')
    parser.add_argument('--difficulty_json', type=str,
                        default='../data/difficulty_dict.json',
                        help='Path to the JSON file with human difficulty labels')
    parser.add_argument('--human_file', type=str,
                        default='../data/humans_as_models_scores_combined.pkl',
                        help='Path to the human scores pickle file')
    parser.add_argument('--filtered_file', type=str,
                        default='../data/filtered_model_score_dict.pkl',
                        help='Path to the filtered model scores pickle file')
    parser.add_argument('--output_dir', type=str,
                        default='pdf',
                        help='Directory to save output figures')
    parser.add_argument('--style', type=str,
                        default='lamalab.mplstyle',
                        help='Matplotlib style file to use')

    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output paths
    difficulty_swarm_path = os.path.join(args.output_dir, 'difficulty_swarm.pdf')
    difficulty_scores_path = os.path.join(args.output_dir, 'difficulty_human_scores.pdf')

    # Set matplotlib style if file exists
    if os.path.exists(args.style):
        plt.style.use(args.style)

    print(f"Loading trace from {args.trace_path}")
    # Load trace using our module
    trace = load_trace(args.trace_path)

    print(f"Loading difficulty labels from {args.difficulty_json}")
    # Load difficulty dictionary
    with open(args.difficulty_json, 'r') as f:
        difficulty_dict = json.load(f)

    # Prepare DataFrame with difficulty values and human labels
    difficulty_df, difficulty_vector = prepare_difficulty_dataframe(trace, difficulty_dict)

    # Print summary statistics
    print("\nSummary statistics by difficulty level:")
    print(difficulty_df.groupby('difficulty_level')['difficulty'].describe())

    # Plot and save swarm plot
    print("\nCreating difficulty swarm plot...")
    plot_difficulty_swarm(difficulty_df, difficulty_swarm_path)

    # Plot and save difficulty vs human scores plot
    print("\nCreating difficulty vs human scores plot...")
    plot_difficulty_vs_mean_scores(
        args.human_file,
        args.filtered_file,
        difficulty_vector,
        difficulty_scores_path
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()