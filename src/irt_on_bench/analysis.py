"""
Diagnostic utilities for IRT analysis.

This module provides functions for checking the quality of fitted IRT models
and identifying problematic items or models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_fit_statistics(
    observed: np.ndarray, expected: np.ndarray
) -> dict[str, float]:
    """
    Calculate fit statistics for comparing observed and expected responses.

    Args:
        observed (np.ndarray): Observed binary responses with shape [n_items, n_participants].
        expected (np.ndarray): Expected response probabilities with shape [n_items, n_participants].

    Returns:
        dict[str, float]: dictionary containing:
            - 'rmse': Root mean squared error.
            - 'mae': Mean absolute error.
            - 'accuracy': Classification accuracy (using 0.5 threshold).
    """
    # Calculate residuals
    residuals = observed - expected

    # Root mean squared error
    rmse = np.sqrt(np.mean(residuals**2))

    # Mean absolute error
    mae = np.mean(np.abs(residuals))

    # Classification accuracy (using 0.5 threshold)
    predicted = (expected >= 0.5).astype(int)
    accuracy = np.mean(observed == predicted)

    return {"rmse": rmse, "mae": mae, "accuracy": accuracy}


def item_fit_statistics(
    response_matrix: np.ndarray,
    abilities: np.ndarray,
    difficulties: np.ndarray,
    discriminations: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate item fit statistics for a fitted IRT model.

    Args:
        response_matrix (np.ndarray): Binary response matrix with shape [n_items, n_participants].
        abilities (np.ndarray): Estimated ability parameters for each participant.
        difficulties (np.ndarray): Item difficulty parameters.
        discriminations (np.ndarray): Item discrimination parameters.

    Returns:
        pd.DataFrame: DataFrame containing fit statistics for each item:
            - 'item_id': Item identifier.
            - 'discrimination': Item discrimination parameter.
            - 'difficulty': Item difficulty parameter.
            - 'p_value': Proportion of correct responses.
            - 'point_biserial': Point-biserial correlation.
            - 'infit': Infit statistic.
            - 'outfit': Outfit statistic.
    """
    n_items, n_participants = response_matrix.shape

    # Initialize output DataFrame
    item_stats = pd.DataFrame(
        {
            "item_id": np.arange(n_items),
            "discrimination": discriminations,
            "difficulty": difficulties,
            "p_value": np.mean(response_matrix, axis=1),
        }
    )

    # Calculate expected probabilities
    expected_probs = np.zeros((n_items, n_participants))
    for i in range(n_items):
        for j in range(n_participants):
            z = discriminations[i] * (abilities[j] - difficulties[i])
            expected_probs[i, j] = 1 / (1 + np.exp(-z))

    # Calculate point-biserial correlation
    point_biserial = np.zeros(n_items)
    for i in range(n_items):
        point_biserial[i] = np.corrcoef(response_matrix[i, :], abilities)[0, 1]

    item_stats["point_biserial"] = point_biserial

    # Calculate infit and outfit statistics (simplified version)
    infit = np.zeros(n_items)
    outfit = np.zeros(n_items)

    for i in range(n_items):
        # Calculate residuals
        residuals = response_matrix[i, :] - expected_probs[i, :]

        # Standardized residuals
        std_residuals = residuals / np.sqrt(
            expected_probs[i, :] * (1 - expected_probs[i, :])
        )

        # Infit: weighted mean squared standardized residuals
        weights = expected_probs[i, :] * (1 - expected_probs[i, :])
        infit[i] = np.sum(weights * std_residuals**2) / np.sum(weights)

        # Outfit: unweighted mean squared standardized residuals
        outfit[i] = np.mean(std_residuals**2)

    item_stats["infit"] = infit
    item_stats["outfit"] = outfit

    return item_stats


def identify_misfitting_items(
    item_stats: pd.DataFrame,
    infit_threshold: float = 1.3,
    outfit_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Identify items that do not fit well with the IRT model.

    Args:
        item_stats (pd.DataFrame): DataFrame containing item fit statistics.
        infit_threshold (float, optional): Threshold for flagging items with high infit (default: 1.3).
        outfit_threshold (float, optional): Threshold for flagging items with high outfit (default: 1.5).

    Returns:
        pd.DataFrame: DataFrame containing only the misfitting items.

    """
    misfitting = (
        (item_stats["infit"] > infit_threshold)
        | (item_stats["outfit"] > outfit_threshold)
        | (item_stats["point_biserial"] < 0.2)
    )

    return item_stats[misfitting].copy()


def plot_item_characteristic_curve(
    difficulty: float,
    discrimination: float,
    ability_range: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot the item characteristic curve for a given item.

    Args:
        difficulty (float): Item difficulty parameter.
        discrimination (float): Item discrimination parameter.
        ability_range (Optional[tuple[float, float]], optional): Range of ability values to plot (default: (-3, 3)).
        ax (Optional[plt.Axes], optional): Matplotlib axes to plot on (default: create new axes).

    Returns:
        plt.Axes: Matplotlib axes containing the plot.
    """
    if ability_range is None:
        ability_range = (-3, 3)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Generate ability values
    abilities = np.linspace(ability_range[0], ability_range[1], 100)

    # Calculate probabilities
    probs = np.zeros_like(abilities)
    for i, ability in enumerate(abilities):
        z = discrimination * (ability - difficulty)
        probs[i] = 1 / (1 + np.exp(-z))

    # Plot the curve
    ax.plot(abilities, probs)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=difficulty, color="red", linestyle="--", alpha=0.5)

    # Add labels and title
    ax.set_xlabel("Ability")
    ax.set_ylabel("Probability of Correct Response")
    ax.set_title(
        f"Item Characteristic Curve (Difficulty: {difficulty:.2f}, Discrimination: {discrimination:.2f})"
    )

    # Set axis limits
    ax.set_xlim(ability_range)
    ax.set_ylim(0, 1)

    return ax
