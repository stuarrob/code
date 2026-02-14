"""
Value Factor for ETFs

Traditional value metrics (P/E, P/B) don't apply to ETFs.
For ETFs, "value" means:
- Low expense ratio (lower cost = better value)
- Low tracking error (efficiency)
- Low bid-ask spread (liquidity) [optional]

Reference: ETF value is about cost-efficiency, not fundamental valuation
"""

import pandas as pd
import numpy as np
from .base_factor import BaseFactor
try:
    from src.utils.logging_config import get_logger
except ModuleNotFoundError:
    import logging
    get_logger = logging.getLogger

logger = get_logger(__name__)


class ValueFactor(BaseFactor):
    """
    Value factor for ETFs based on cost-efficiency.

    For ETFs, value = low costs + high efficiency
    """

    def __init__(self):
        """Initialize value factor."""
        super().__init__("value", lookback_period=60)  # Minimal lookback needed

    def calculate(self,
                  prices: pd.DataFrame,
                  expense_ratios: pd.Series = None,
                  benchmarks: pd.DataFrame = None,
                  **kwargs) -> pd.Series:
        """
        Calculate value scores for ETFs.

        Args:
            prices: DataFrame of ETF prices
            expense_ratios: Series of annual expense ratios (e.g., 0.03 = 0.03%)
            benchmarks: DataFrame of benchmark prices (for tracking error)

        Returns:
            pd.Series: Normalized value scores (higher = better value)
        """
        if expense_ratios is None:
            raise ValueError("ValueFactor requires expense_ratios parameter")

        # Ensure expense ratios match price tickers
        expense_ratios = expense_ratios.reindex(prices.columns)

        # Calculate expense ratio score (lower ER = higher score)
        er_scores = -1 * expense_ratios  # Negative because lower is better

        # If benchmarks provided, calculate tracking error
        if benchmarks is not None:
            te_scores = self._calculate_tracking_error_scores(prices, benchmarks)

            # Combine: 60% expense ratio, 40% tracking error
            value_score = 0.6 * self.normalize(er_scores) + 0.4 * self.normalize(te_scores)

            logger.info(
                f"Value factor: Using expense ratio (60%) + tracking error (40%)"
            )
        else:
            # Only expense ratio
            value_score = self.normalize(er_scores)

            logger.info(
                f"Value factor: Using expense ratio only"
            )

        return value_score

    def _calculate_tracking_error_scores(self,
                                        prices: pd.DataFrame,
                                        benchmarks: pd.DataFrame) -> pd.Series:
        """
        Calculate tracking error scores.

        Lower tracking error = better (more efficient)
        """
        # Calculate returns
        etf_returns = prices.pct_change().dropna()
        bench_returns = benchmarks.pct_change().dropna()

        # Align indices
        common_dates = etf_returns.index.intersection(bench_returns.index)
        etf_returns = etf_returns.loc[common_dates]
        bench_returns = bench_returns.loc[common_dates]

        # Tracking difference
        tracking_diff = etf_returns.sub(bench_returns, axis=0)

        # Tracking error = std of tracking difference (annualized)
        tracking_error = tracking_diff.std() * np.sqrt(252)

        # Score = -1 * tracking error (lower TE = higher score)
        te_scores = -1 * tracking_error

        return te_scores


class SimplifiedValueFactor(ValueFactor):
    """
    Simplified value factor using only expense ratios.

    Use this when benchmark/tracking error data is unavailable.
    """

    def __init__(self):
        super().__init__()
        self.name = "simplified_value"

    def calculate(self, prices: pd.DataFrame,
                  expense_ratios: pd.Series, **kwargs) -> pd.Series:
        """Calculate value scores using expense ratios only."""
        # Force ignore benchmarks
        return super().calculate(prices, expense_ratios, benchmarks=None)
