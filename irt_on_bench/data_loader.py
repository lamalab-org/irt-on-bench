"""
Data loading utilities for IRT analysis.

This module provides functions to load model scores and benchmark data
from pickle files or other formats for further analysis.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_scores(file_path: str) -> Dict[str, Any]:
    """
    Load model score dictionaries from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the pickle file containing model scores

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model score data

    Example
    -------
    >>> model_scores = load_model_scores('../data/filtered_model_score_dict.pkl')
    >>> print(list(model_scores['overall'].keys()))  # List available models
    """
    try:
        with open(file_path, 'rb') as f:
            model_score_dicts = pickle.load(f)

        # Check if the loaded object is a dictionary
        if not isinstance(model_score_dicts, dict):
            raise ValueError(f"Loaded object is not a dictionary: {type(model_score_dicts)}")

        # Basic validation
        if 'overall' not in model_score_dicts:
            logger.warning("Loaded scores dictionary does not contain 'overall' category")

        return model_score_dicts

    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error loading model scores from {file_path}: {e}")
        raise
    except (pickle.PickleError, ValueError) as e:
        logger.error(f"Error unpickling model scores from {file_path}: {e}")
        raise


def create_binary_matrix(model_score_dicts: Dict[str, Any],
                         category: str = 'overall',
                         column: str = 'all_correct_',
                         reference_model: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame, list]:
    """
    Create a binary matrix from model score dictionaries.

    Parameters
    ----------
    model_score_dicts : Dict[str, Any]
        Dictionary containing model score data
    category : str, optional
        Category of scores to use (default: 'overall')
    column : str, optional
        Column name containing the binary scores (default: 'all_correct_')
    reference_model : Optional[str], optional
        Model to use as reference for dimensions. If None, the first model is used.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, list]
        Binary matrix as NumPy array,
        Same matrix as pandas DataFrame,
        List of model names

    Example
    -------
    >>> binary_array, binary_df, models = create_binary_matrix(model_scores)
    >>> print(f"Matrix shape: {binary_array.shape}")
    """
    # Validate inputs
    if category not in model_score_dicts:
        raise ValueError(f"Category '{category}' not found in model_score_dicts")

    # Extract model names
    models = list(model_score_dicts[category].keys())
    if not models:
        raise ValueError(f"No models found in category '{category}'")

    # Create a reference model to get dimensions
    if reference_model is None:
        reference_model = models[0]  # Use first model as reference
    elif reference_model not in models:
        raise ValueError(f"Reference model '{reference_model}' not found in models list")

    # Check if the column exists in the reference model
    if column not in model_score_dicts[category][reference_model]:
        raise ValueError(f"Column '{column}' not found in reference model '{reference_model}'")

    # Create matrix with correct dimensions
    n_questions = len(model_score_dicts[category][reference_model][column])
    binary_matrix = np.zeros((n_questions, len(models)))

    # Fill the matrix with binary scores
    for i, model in enumerate(models):
        # Skip models that don't have the required column
        if column not in model_score_dicts[category][model]:
            logger.warning(f"Column '{column}' not found in model '{model}', filling with zeros")
            continue

        # Check if the dimensions match
        if len(model_score_dicts[category][model][column]) != n_questions:
            logger.warning(f"Model '{model}' has {len(model_score_dicts[category][model][column])} questions, "
                           f"but reference model has {n_questions}. Using available data.")
            # Fill with available data
            available = min(n_questions, len(model_score_dicts[category][model][column]))
            binary_matrix[:available, i] = model_score_dicts[category][model][column].values[:available]
        else:
            # Fill the matrix with binary scores
            binary_matrix[:, i] = model_score_dicts[category][model][column].values

    # Convert to pandas DataFrame for better visualization
    binary_df = pd.DataFrame(binary_matrix, columns=models)

    logger.info(f"Created binary matrix with shape {binary_matrix.shape}")
    return binary_matrix, binary_df, models


def save_trace(trace: Any, file_path: str) -> None:
    """
    Save PyMC trace object to a pickle file.

    Parameters
    ----------
    trace : Any
        PyMC trace object to save
    file_path : str
        Path where to save the trace
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(trace, f)
        logger.info(f"Successfully saved trace to {file_path}")
    except (IOError, pickle.PickleError) as e:
        logger.error(f"Error saving trace to {file_path}: {e}")
        raise


def load_trace(file_path: str) -> Any:
    """
    Load PyMC trace object from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the pickle file containing the trace

    Returns
    -------
    Any
        PyMC trace object
    """
    try:
        with open(file_path, 'rb') as f:
            trace = pickle.load(f)
        logger.info(f"Successfully loaded trace from {file_path}")
        return trace
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error loading trace from {file_path}: {e}")
        raise
    except (pickle.PickleError, ValueError) as e:
        logger.error(f"Error unpickling trace from {file_path}: {e}")
        raise


def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the loaded data
    """
    import json
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise