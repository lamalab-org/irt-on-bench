#!/usr/bin/env python
"""
Compare language model performance using classic scores and IRT ability estimates.

This script loads model scores and IRT analysis results to create a comparison
visualization that shows both traditional scoring and IRT-based evaluations
with uncertainty estimates.

Usage:
    python model_comparison.py --trace_path=../data/trace_all_correct_filtered.pkl
                             --scores_path=../data/filtered_model_score_dict.pkl
                             --output=model_comparison.pdf
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy.constants import golden

from irt_on_bench.data import load_model_scores, load_trace

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


def get_parameter_estimates(trace):
    """
    Extract point estimates and uncertainty from a PyMC trace.

    Parameters
    ----------
    trace : arviz.InferenceData
        PyMC trace with posterior samples

    Returns
    -------
    dict
        Dictionary containing mean, standard deviation, and HDI
        for abilities, discriminations, and difficulties
    """
    estimates = {
        'abilities': {
            'mean': trace.posterior['abilities'].mean(dim=['chain', 'draw']).values,
            'std': trace.posterior['abilities'].std(dim=['chain', 'draw']).values,
            'hdi': az.hdi(trace, var_names=['abilities'])
        },
        'discriminations': {
            'mean': trace.posterior['discriminations'].mean(dim=['chain', 'draw']).values,
            'std': trace.posterior['discriminations'].std(dim=['chain', 'draw']).values,
            'hdi': az.hdi(trace, var_names=['discriminations'])
        },
        'difficulties': {
            'mean': trace.posterior['difficulties'].mean(dim=['chain', 'draw']).values,
            'std': trace.posterior['difficulties'].std(dim=['chain', 'draw']).values,
            'hdi': az.hdi(trace, var_names=['difficulties'])
        }
    }
    return estimates


def calculate_model_scores(model_score_dicts, category='overall'):
    """
    Calculate average scores for each model.

    Parameters
    ----------
    model_score_dicts : dict
        Dictionary containing model scores
    category : str, optional
        Category of scores to use (default: 'overall')

    Returns
    -------
    pandas.DataFrame
        DataFrame with average scores for each model
    """
    model_list = list(model_score_dicts[category].keys())

    model_score_list = []
    for model in model_list:
        score = (sum(model_score_dicts[category][model]['all_correct_']) /
                 len(model_score_dicts[category][model]['all_correct_']))
        model_score_list.append(score)

    return pd.DataFrame(model_score_list, index=model_list, columns=['score'])


def plot_model_comparison(df, abilities_mean, abilities_std=None,
                          models_to_plot=None, colors=None, output_path="model_comparison.pdf"):
    """
    Create a comparison plot of traditional scores vs. IRT ability estimates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with traditional scores
    abilities_mean : array-like
        Mean ability estimates from IRT
    abilities_std : array-like, optional
        Standard deviation of ability estimates
    models_to_plot : list, optional
        List of model names to include in the plot
    colors : list, optional
        List of colors for the models
    output_path : str, optional
        Path to save the figure (default: "model_comparison.pdf")
    """
    fig, ax = plt.subplots(figsize=(1.5 * ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))

    # Normalize both the average scores and abilities to 0-1 scale
    df_norm = (df - df.min()) / (df.max() - df.min())
    abilities_norm = (abilities_mean - abilities_mean.min()) / (abilities_mean.max() - abilities_mean.min())

    # Normalize the standard deviations using the same scale factor as abilities
    if abilities_std is not None:
        scale_factor = 1 / (abilities_mean.max() - abilities_mean.min())
        abilities_std_norm = abilities_std * scale_factor

    if models_to_plot is None:
        models_to_plot = df.index

    if colors is None:
        colors = ['#03071E', '#3A3B73', '#6A040F', '#B20404', '#D63909',
                  '#E27F07', '#F9BB0B', '#FFDA00']

    x = np.array([0.1, 0.9])

    for i, model in enumerate(models_to_plot):
        idx = df.index.get_loc(model)
        y = np.array([df_norm.loc[model, 'score'], abilities_norm[idx]])

        # Determine label for the legend
        if model == 'o1':
            label = 'o1-preview'
        elif model == 'Claude-2-Zero-T':
            label = 'Claude-2'
        elif model == 'GPT-3.5 Turbo Zero-T':
            label = 'GPT-3.5 Turbo'
        else:
            label = model

        # Plot the line
        plt.plot(x, y, '-', label=label, color=colors[i % len(colors)], linewidth=1.5)

        # Add error bar for IRT score if std is available
        if abilities_std is not None:
            plt.errorbar(0.9, abilities_norm[idx], yerr=abilities_std_norm[idx],
                         fmt='none', color=colors[i % len(colors)], capsize=0, linewidth=1)

    plt.xticks([0.1, 0.9], ['Current\nimplementation', 'IRT'])
    plt.ylabel('Normalized Score')

    # Set axis limits
    if abilities_std is not None:
        ymin = min(0, np.min(abilities_norm - abilities_std_norm) - 0.05)
        ymax = max(1, np.max(abilities_norm + abilities_std_norm) + 0.05)
    else:
        ymin = 0
        ymax = 1
    plt.ylim(ymin, ymax)
    plt.xlim(0, 1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Apply range frame
    y_values = np.concatenate([
        df_norm.loc[models_to_plot, 'score'],
        abilities_norm[[df.index.get_loc(m) for m in models_to_plot]]
    ])
    range_frame(ax, x, y_values)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')

    print(f"Figure saved to {output_path}")
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare model performance metrics')
    parser.add_argument('--trace_path', type=str,
                        default='../results/trace_all_correct_filtered.pkl',
                        help='Path to the PyMC trace pickle file')
    parser.add_argument('--scores_path', type=str,
                        default='../data/filtered_model_score_dict.pkl',
                        help='Path to the model scores pickle file')
    parser.add_argument('--output', type=str,
                        default='pdf/model_comparison.pdf',
                        help='Path to save the output figure')
    parser.add_argument('--style', type=str,
                        default='lamalab.mplstyle',
                        help='Matplotlib style file to use')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['o1', 'Claude-3.5 (Sonnet)', 'Mistral-Large-2',
                                 'GPT-4', 'Claude-2-Zero-T', 'Gemini-Pro',
                                 'GPT-3.5 Turbo Zero-T', 'Galatica-120b'],
                        help='Models to include in the comparison plot')

    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    args = parse_args()

    # Set matplotlib style if file exists
    if os.path.exists(args.style):
        plt.style.use(args.style)

    print(f"Loading trace from {args.trace_path}")
    # Load trace
    trace = load_trace(args.trace_path)

    print(f"Loading model scores from {args.scores_path}")
    # Load model scores
    model_score_dicts = load_model_scores(args.scores_path)

    # Calculate model scores
    model_score_df = calculate_model_scores(model_score_dicts)

    # Get parameter estimates
    print("Calculating parameter estimates...")
    estimates = get_parameter_estimates(trace)
    abilities_mean = estimates['abilities']['mean']
    abilities_std = estimates['abilities']['std']

    # Print basic statistics
    print("\nModel abilities (IRT estimates):")
    for model in args.models:
        if model in model_score_df.index:
            idx = model_score_df.index.get_loc(model)
            print(f"{model}: {abilities_mean[idx]:.4f} ± {abilities_std[idx]:.4f}")

    # Create comparison plot
    print("\nCreating comparison plot...")
    plot_model_comparison(
        model_score_df,
        abilities_mean,
        abilities_std,
        models_to_plot=args.models,
        output_path=args.output
    )


if __name__ == "__main__":
    main()