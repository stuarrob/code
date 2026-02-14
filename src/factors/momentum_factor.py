"""
Momentum Factor

Based on AQR Capital Management research:
- Use 12-month momentum (252 trading days)
- Skip most recent month (21 days) to avoid short-term reversal
- Winsorize outliers to reduce noise

Reference: "Fact, Fiction and Momentum Investing" - AQR (2014)
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


class MomentumFactor(BaseFactor):
    """
    Time-series momentum factor for ETFs.

    Calculates momentum as price change over lookback period,
    skipping the most recent period to avoid short-term mean reversion.
    """

    def __init__(self,
                 lookback: int = 252,
                 skip_recent: int = 21,
                 winsorize_pct: float = 0.01):
        """
        Initialize momentum factor.

        Args:
            lookback: Days to look back for momentum calculation (default: 252 = 12 months)
            skip_recent: Days to skip at end to avoid reversal (default: 21 = 1 month)
            winsorize_pct: Percentile for winsorization (default: 0.01 = 1%)
        """
        super().__init__("momentum", lookback)
        self.skip_recent = skip_recent
        self.winsorize_pct = winsorize_pct

        logger.info(
            f"Momentum factor initialized: lookback={lookback}d, "
            f"skip_recent={skip_recent}d, winsorize={winsorize_pct}"
        )

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate momentum scores.

        Momentum = (Price[t-skip] / Price[t-lookback]) - 1

        Args:
            prices: DataFrame of ETF prices (columns=tickers, index=dates)

        Returns:
            pd.Series: Normalized momentum scores
        """
        # Validate data
        self.validate_data(prices)

        # Get relevant price points
        # End price: skip_recent days ago (to avoid reversal)
        end_idx = -self.skip_recent if self.skip_recent > 0 else len(prices)
        end_prices = prices.iloc[end_idx]

        # Start price: lookback days ago
        start_prices = prices.iloc[-self.lookback_period]

        # Calculate raw momentum
        momentum = (end_prices / start_prices) - 1

        # Handle division by zero or invalid values
        momentum = momentum.replace([np.inf, -np.inf], np.nan)

        # Log statistics
        valid_momentum = momentum.dropna()
        if len(valid_momentum) > 0:
            logger.debug(
                f"Momentum stats: mean={valid_momentum.mean():.3f}, "
                f"std={valid_momentum.std():.3f}, "
                f"min={valid_momentum.min():.3f}, "
                f"max={valid_momentum.max():.3f}"
            )

        # Winsorize extreme values
        momentum = self.winsorize(
            momentum,
            lower=self.winsorize_pct,
            upper=1 - self.winsorize_pct
        )

        # Normalize to z-score
        normalized = self.normalize(momentum, method='zscore')

        return normalized

    def calculate_rolling(self,
                         prices: pd.DataFrame,
                         window: int = 21) -> pd.DataFrame:
        """
        Calculate rolling momentum scores (useful for stability analysis).

        Args:
            prices: DataFrame of ETF prices
            window: Window for rolling calculation (default: 21 days)

        Returns:
            pd.DataFrame: Rolling momentum scores over time
        """
        if len(prices) < self.lookback_period + window:
            raise ValueError(
                f"Need at least {self.lookback_period + window} days for rolling calc"
            )

        rolling_scores = []
        dates = []

        for i in range(self.lookback_period, len(prices) - self.skip_recent, window):
            # Get price window for this calculation
            price_window = prices.iloc[:i+self.skip_recent]

            # Calculate momentum for this point in time
            scores = self.calculate(price_window)

            rolling_scores.append(scores)
            dates.append(prices.index[i])

        return pd.DataFrame(rolling_scores, index=dates)


class DualMomentumFactor(MomentumFactor):
    """
    Dual Momentum: Combines relative and absolute momentum.

    Relative: Rank ETFs by momentum (which is best)
    Absolute: Filter out ETFs with negative momentum (trending down)

    Based on Gary Antonacci's research.
    """

    def __init__(self,
                 lookback: int = 252,
                 skip_recent: int = 21,
                 absolute_threshold: float = 0.0):
        """
        Initialize dual momentum.

        Args:
            lookback: Days for momentum calculation
            skip_recent: Days to skip to avoid reversal
            absolute_threshold: Minimum momentum to pass filter (default: 0 = positive trend)
        """
        super().__init__(lookback, skip_recent)
        self.absolute_threshold = absolute_threshold
        self.name = "dual_momentum"

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate dual momentum scores.

        Returns:
            pd.Series: Scores where negative momentum ETFs are set to NaN
        """
        # Get relative momentum scores
        momentum_scores = super().calculate(prices)

        # Calculate raw momentum for absolute filter
        end_idx = -self.skip_recent if self.skip_recent > 0 else len(prices)
        end_prices = prices.iloc[end_idx]
        start_prices = prices.iloc[-self.lookback_period]
        raw_momentum = (end_prices / start_prices) - 1

        # Apply absolute momentum filter
        # Set scores to NaN for ETFs below threshold
        filtered_scores = momentum_scores.copy()
        failed_filter = raw_momentum < self.absolute_threshold
        filtered_scores[failed_filter] = np.nan

        num_filtered = failed_filter.sum()
        logger.info(
            f"Dual momentum: {num_filtered}/{len(raw_momentum)} ETFs "
            f"filtered due to negative/weak trend"
        )

        return filtered_scores
