"""
Data loading utilities for IRT analysis.

This module provides functions to load model scores and benchmark data
from pickle files or other formats for further analysis.
"""

import pickle
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def load_model_scores(file_path: str) -> dict[str, Any]:
    """
    Args:
        file_path (str): Path to the pickle file containing model scores

    Returns:
        dict[str, Any]: dictionary containing model score data
    """
    try:
        with open(file_path, "rb") as f:
            model_score_dicts = pickle.load(f)

        if not isinstance(model_score_dicts, dict):
            raise ValueError(
                f"Loaded object is not a dictionary: {type(model_score_dicts)}"
            )

        if "overall" not in model_score_dicts:
            logger.warning(
                "Loaded scores dictionary does not contain 'overall' category"
            )

        return model_score_dicts

    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error loading model scores from {file_path}: {e}")
        raise
    except (pickle.PickleError, ValueError) as e:
        logger.error(f"Error unpickling model scores from {file_path}: {e}")
        raise


def create_binary_matrix(
    model_score_dicts: dict[str, Any],
    category: str = "overall",
    column: str = "all_correct_",
    reference_model: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list]:
    """
    Args:
        model_score_dicts (dict[str, Any]): dictionary containing model score data
        category (str, optional): Category of scores to use (default: 'overall')
        column (str, optional): Column name containing the binary scores (default: 'all_correct_')
        reference_model (str | None): Model to use as reference for dimensions. If None, the first model is used.

    Returns:
        tuple[np.ndarray, pd.DataFrame, list]:
            Binary matrix as NumPy array,
            Same matrix as pandas DataFrame,
            List of model names
    """
    if category not in model_score_dicts:
        raise ValueError(f"Category '{category}' not found in model_score_dicts")

    models = list(model_score_dicts[category].keys())
    if not models:
        raise ValueError(f"No models found in category '{category}'")

    if reference_model is None:
        reference_model = models[0]
    elif reference_model not in models:
        raise ValueError(
            f"Reference model '{reference_model}' not found in models list"
        )

    if column not in model_score_dicts[category][reference_model]:
        raise ValueError(
            f"Column '{column}' not found in reference model '{reference_model}'"
        )

    n_questions = len(model_score_dicts[category][reference_model][column])
    binary_matrix = np.zeros((n_questions, len(models)))

    for i, model in enumerate(models):
        if column not in model_score_dicts[category][model]:
            logger.warning(
                f"Column '{column}' not found in model '{model}', filling with zeros"
            )
            continue

        if len(model_score_dicts[category][model][column]) != n_questions:
            logger.warning(
                f"Model '{model}' has {len(model_score_dicts[category][model][column])} questions, "
                f"but reference model has {n_questions}. Using available data."
            )
            available = min(
                n_questions, len(model_score_dicts[category][model][column])
            )
            binary_matrix[:available, i] = model_score_dicts[category][model][
                column
            ].values[:available]
        else:
            binary_matrix[:, i] = model_score_dicts[category][model][column].values

    binary_df = pd.DataFrame(binary_matrix, columns=models)

    logger.info(f"Created binary matrix with shape {binary_matrix.shape}")
    return binary_matrix, binary_df, models


def save_trace(trace: Any, file_path: str) -> None:
    """
    Args:
        trace (Any): PyMC trace object to save
        file_path (str): Path where to save the trace
    """
    try:
        with open(file_path, "wb") as f:
            pickle.dump(trace, f)
        logger.info(f"Successfully saved trace to {file_path}")
    except (IOError, pickle.PickleError) as e:
        logger.error(f"Error saving trace to {file_path}: {e}")
        raise


def load_trace(file_path: str) -> Any:
    """
    Args:
        file_path (str): Path to the pickle file containing the trace

    Returns:
        Any: PyMC trace object
    """
    try:
        with open(file_path, "rb") as f:
            trace = pickle.load(f)
        logger.info(f"Successfully loaded trace from {file_path}")
        return trace
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error loading trace from {file_path}: {e}")
        raise
    except (pickle.PickleError, ValueError) as e:
        logger.error(f"Error unpickling trace from {file_path}: {e}")
        raise


def load_json_data(file_path: str) -> dict[str, Any]:
    """
    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict[str, Any]: dictionary containing the loaded data
    """
    import json

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise
