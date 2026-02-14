"""
Base Factor Abstract Class

All factor implementations inherit from this class to ensure consistent interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional
try:
    from src.utils.logging_config import get_logger
except ModuleNotFoundError:
    import logging
    get_logger = logging.getLogger

logger = get_logger(__name__)


class BaseFactor(ABC):
    """
    Abstract base class for all factor calculations.

    All factors should:
    1. Calculate a score for each ETF
    2. Normalize scores to z-score (mean=0, std=1)
    3. Handle missing data gracefully
    4. Be deterministic and testable
    """

    def __init__(self, name: str, lookback_period: int):
        """
        Initialize factor.

        Args:
            name: Factor name (e.g., 'momentum', 'quality')
            lookback_period: Number of days of historical data needed
        """
        self.name = name
        self.lookback_period = lookback_period
        logger.debug(f"Initialized {name} factor with lookback={lookback_period}")

    @abstractmethod
    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate factor scores for each ETF.

        Args:
            prices: DataFrame with ETF prices, columns=tickers, index=dates
            **kwargs: Additional data (volumes, fundamentals, etc.)

        Returns:
            pd.Series: Factor scores, normalized to z-score
                      Index = ticker symbols
                      Values = normalized scores
        """
        pass

    def validate_data(self, prices: pd.DataFrame) -> None:
        """
        Validate that input data is sufficient for calculation.

        Raises:
            ValueError: If data is insufficient
        """
        if len(prices) < self.lookback_period:
            raise ValueError(
                f"{self.name} factor requires {self.lookback_period} days of data, "
                f"but only {len(prices)} days provided"
            )

        if prices.isnull().any().any():
            null_cols = prices.columns[prices.isnull().any()].tolist()
            logger.warning(f"{self.name}: Found NaN values in {len(null_cols)} ETFs")

    def normalize(self, scores: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        Normalize scores to standard distribution.

        Args:
            scores: Raw factor scores
            method: Normalization method ('zscore', 'rank', 'minmax')

        Returns:
            pd.Series: Normalized scores
        """
        # Remove NaN values
        valid_scores = scores.dropna()

        if len(valid_scores) == 0:
            logger.error(f"{self.name}: All scores are NaN!")
            return scores

        if method == 'zscore':
            # Z-score: (x - mean) / std
            mean = valid_scores.mean()
            std = valid_scores.std()

            if std == 0:
                logger.warning(f"{self.name}: Zero std deviation, returning zeros")
                return pd.Series(0, index=scores.index)

            normalized = (scores - mean) / std

        elif method == 'rank':
            # Rank percentile (0-1)
            normalized = scores.rank(pct=True)

        elif method == 'minmax':
            # Min-max scaling (0-1)
            min_val = valid_scores.min()
            max_val = valid_scores.max()

            if max_val == min_val:
                logger.warning(f"{self.name}: All values identical, returning 0.5")
                return pd.Series(0.5, index=scores.index)

            normalized = (scores - min_val) / (max_val - min_val)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def winsorize(self, scores: pd.Series, lower: float = 0.01,
                  upper: float = 0.99) -> pd.Series:
        """
        Winsorize scores to remove extreme outliers.

        Args:
            scores: Input scores
            lower: Lower percentile to clip (default 1%)
            upper: Upper percentile to clip (default 99%)

        Returns:
            pd.Series: Winsorized scores
        """
        valid_scores = scores.dropna()

        if len(valid_scores) == 0:
            return scores

        lower_bound = valid_scores.quantile(lower)
        upper_bound = valid_scores.quantile(upper)

        winsorized = scores.clip(lower=lower_bound, upper=upper_bound)

        num_clipped = ((scores < lower_bound) | (scores > upper_bound)).sum()
        if num_clipped > 0:
            logger.debug(f"{self.name}: Winsorized {num_clipped} outliers")

        return winsorized

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', lookback={self.lookback_period})"
