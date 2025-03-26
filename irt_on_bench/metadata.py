"""
Question metadata classes for IRT analysis.

This module provides classes for representing question metadata and computing
scores using different scoring methods.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ScoringType(Enum):
    """
    Enum for different types of scoring methods.

    Attributes
    ----------
    BINARY : str
        Binary scoring (correct/incorrect)
    PARTIAL : str
        Partial credit scoring
    """
    BINARY = 'binary'
    PARTIAL = 'partial'


@dataclass
class QuestionMetadata:
    """
    Base class for question metadata.

    This class provides a common interface for different types of questions
    with their specific scoring methods.

    Parameters
    ----------
    question_id : str
        Unique identifier for the question
    scoring_type : ScoringType
        Type of scoring used for this question
    """
    question_id: str
    scoring_type: ScoringType

    def compute_score(self, response: Any) -> float:
        """
        Base method for computing scores.

        Parameters
        ----------
        response : Any
            Response data to score

        Returns
        -------
        float
            Computed score

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement compute_score")


@dataclass
class BinaryQuestionMetadata(QuestionMetadata):
    """
    Metadata for binary scored questions.

    This class implements scoring for binary (correct/incorrect) questions.

    Parameters
    ----------
    question_id : str
        Unique identifier for the question
    """

    def __init__(self, question_id: str):
        """
        Initialize a binary question metadata object.

        Parameters
        ----------
        question_id : str
            Unique identifier for the question
        """
        super().__init__(question_id, ScoringType.BINARY)

    def compute_score(self, row: Any, column: str = 'all_correct_') -> float:
        """
        Use the all_correct column directly for binary scoring.

        Parameters
        ----------
        row : Any
            Row containing the response data
        column : str, optional
            Column name containing the binary score (default: 'all_correct_')

        Returns
        -------
        float
            Binary score (0.0 or 1.0)
        """
        if column not in row:
            raise ValueError(f"Column '{column}' not found in row")

        return float(row[column])


@dataclass
class PartialCreditQuestionMetadata(QuestionMetadata):
    """
    Metadata for questions with partial credit scoring.

    Parameters
    ----------
    question_id : str
        Unique identifier for the question
    primary_column : str
        Column name containing the primary score
    partial_columns : Optional[list], optional
        List of column names for partial credit components (default: None)
    weights : Optional[list], optional
        List of weights for each column (default: None, equal weights)
    """
    primary_column: str
    partial_columns: Optional[list] = None
    weights: Optional[list] = None

    def __init__(self, question_id: str, primary_column: str,
                 partial_columns: Optional[list] = None,
                 weights: Optional[list] = None):
        """
        Initialize a partial credit question metadata object.

        Parameters
        ----------
        question_id : str
            Unique identifier for the question
        primary_column : str
            Column name containing the primary score
        partial_columns : Optional[list], optional
            List of column names for partial credit components (default: None)
        weights : Optional[list], optional
            List of weights for each column (default: None, equal weights)
        """
        super().__init__(question_id, ScoringType.PARTIAL)
        self.primary_column = primary_column
        self.partial_columns = partial_columns or []

        # Validate and normalize weights
        if weights is not None:
            if len(weights) != len(self.partial_columns) + 1:
                raise ValueError("Number of weights must match number of columns (primary + partial)")

            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Equal weights if not specified
            n_cols = len(self.partial_columns) + 1  # +1 for primary column
            self.weights = [1.0 / n_cols] * n_cols

    def compute_score(self, row: Any) -> float:
        """
        Compute weighted score using primary and partial credit columns.

        Parameters
        ----------
        row : Any
            Row containing the response data

        Returns
        -------
        float
            Weighted score between 0.0 and 1.0
        """
        # Check if primary column exists
        if self.primary_column not in row:
            raise ValueError(f"Primary column '{self.primary_column}' not found in row")

        # Get primary score
        primary_score = float(row[self.primary_column])

        # If no partial columns, return primary score
        if not self.partial_columns:
            return primary_score

        # Collect all scores
        scores = [primary_score]
        for col in self.partial_columns:
            if col not in row:
                raise ValueError(f"Partial column '{col}' not found in row")
            scores.append(float(row[col]))

        # Compute weighted sum
        weighted_score = sum(s * w for s, w in zip(scores, self.weights))

        return weighted_score