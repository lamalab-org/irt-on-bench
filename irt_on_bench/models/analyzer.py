"""
Benchmark analyzer for IRT analysis.

This module provides the BenchmarkAnalyzer class for computing score matrices
and fitting IRT models to language model evaluation data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from irt_on_bench.models.metadata import QuestionMetadata
from irt_on_bench.models.irt_models import (twopl_mml, rasch_mml, ability_mle)


class BenchmarkAnalyzer:
    """
    Class for analyzing benchmark results using IRT models.

    This class provides methods for:
    - Adding model results and question metadata
    - Computing score matrices
    - Fitting IRT models
    - Analyzing extreme items

    Attributes
    ----------
    model_dataframes : Dict[str, pd.DataFrame]
        Dictionary mapping model IDs to their results DataFrames
    question_metadata : Dict[str, QuestionMetadata]
        Dictionary mapping question IDs to their metadata
    score_matrix : Optional[np.ndarray]
        Computed score matrix (None until computed)
    model_ids : Optional[List[str]]
        List of model IDs (None until score matrix is computed)
    """

    def __init__(self):
        """Initialize a new BenchmarkAnalyzer instance."""
        self.model_dataframes: Dict[str, pd.DataFrame] = {}
        self.question_metadata: Dict[str, QuestionMetadata] = {}
        self.score_matrix: Optional[np.ndarray] = None
        self.model_ids: Optional[List[str]] = None

    def add_model_results(self, model_id: str, results_df: pd.DataFrame) -> None:
        """
        Add a model's results DataFrame.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model
        results_df : pd.DataFrame
            DataFrame containing the model's results
        """
        self.model_dataframes[model_id] = results_df

    def add_question_metadata(self, metadata: QuestionMetadata) -> None:
        """
        Add metadata for a question.

        Parameters
        ----------
        metadata : QuestionMetadata
            Metadata object for the question
        """
        self.question_metadata[metadata.question_id] = metadata

    def compute_score_matrix(self) -> np.ndarray:
        """
        Compute score matrix using metadata-specific scoring.

        Returns
        -------
        np.ndarray
            Computed score matrix with dimensions [n_models, n_questions]

        Raises
        ------
        ValueError
            If model results or question metadata are missing
        """
        if not self.model_dataframes or not self.question_metadata:
            raise ValueError("Need both model results and question metadata")

        self.model_ids = list(self.model_dataframes.keys())
        question_ids = list(self.question_metadata.keys())

        # Initialize score matrix
        self.score_matrix = np.full(
            (len(self.model_ids), len(question_ids)),
            np.nan
        )

        # Compute scores
        for model_idx, model_id in enumerate(self.model_ids):
            df = self.model_dataframes[model_id]
            for q_idx, q_id in enumerate(question_ids):
                if q_id not in df.index:
                    continue

                metadata = self.question_metadata[q_id]
                row = df.loc[q_id]

                self.score_matrix[model_idx, q_idx] = metadata.compute_score(row)

        return self.score_matrix

    def fit_irt(self, model: str = '2pl') -> Dict[str, Any]:
        """
        Fit an IRT model to the score matrix.

        Parameters
        ----------
        model : str, optional
            IRT model to fit: '2pl' for 2-parameter logistic or 'rasch' for Rasch model
            (default: '2pl')

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'difficulties': Item difficulty parameters
            - 'discriminations': Item discrimination parameters
            - 'abilities': Model ability parameters
            - 'binary_matrix': Binary score matrix used for fitting

        Notes
        -----
        The score matrix is transposed to have shape [n_items, n_participants]
        as required by the IRT fitting functions.
        """
        if self.score_matrix is None:
            self.compute_score_matrix()

        # Convert to binary matrix and transpose for IRT models
        # (needs to be [n_items, n_participants])
        binary_matrix = (self.score_matrix >= 0.5).astype(int).T

        if model == '2pl':
            # Fit 2-parameter logistic model
            results = twopl_mml(binary_matrix)
            difficulties = results['Difficulty']
            discriminations = results['Discrimination']
        else:  # rasch
            # Fit Rasch model (1-parameter logistic)
            results = rasch_mml(binary_matrix)
            difficulties = results['Difficulty']
            discriminations = np.ones_like(difficulties)

        # Estimate abilities using maximum likelihood estimation
        abilities = ability_mle(
            binary_matrix,
            difficulties,
            discriminations,
            no_estimate=np.nan
        )

        return {
            'difficulties': difficulties,
            'discriminations': discriminations,
            'abilities': abilities,
            'binary_matrix': binary_matrix
        }

    @staticmethod
    def analyze_extreme_items(difficulties: np.ndarray,
                              discriminations: np.ndarray,
                              question_ids: List[str],
                              threshold: float = 0.95) -> pd.DataFrame:
        """
        Identify items with extreme parameters.

        Parameters
        ----------
        difficulties : np.ndarray
            Array of item difficulties
        discriminations : np.ndarray
            Array of item discriminations
        question_ids : List[str]
            List of question IDs corresponding to the items
        threshold : float, optional
            Threshold for identifying extreme values (default: 0.95)

        Returns
        -------
        pd.DataFrame
            DataFrame of items with extreme parameter values, containing:
            - question_id: Question identifier
            - difficulty: Item difficulty parameter
            - discrimination: Item discrimination parameter
            - is_extreme: Boolean indicating if the item has extreme parameters
        """
        # Create DataFrame with question IDs and parameters
        extreme_items = pd.DataFrame({
            'question_id': question_ids,
            'difficulty': difficulties,
            'discrimination': discriminations
        })

        # Find items with extreme values
        extreme_items['is_extreme'] = (
                (discriminations > threshold * 5.0) |  # High discrimination
                (difficulties > threshold * 6.0) |  # Very difficult
                (difficulties < -4.0)  # Very easy
        )

        # Return only the extreme items
        return extreme_items[extreme_items['is_extreme']]