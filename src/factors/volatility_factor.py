"""
Low Volatility Factor

Based on the "low-volatility anomaly":
- Low volatility stocks/ETFs have historically outperformed high volatility
- Contradicts CAPM (higher risk should = higher return)
- Documented across 50+ years, all markets

Reference: "Betting Against Beta" - AQR (2014)
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


class VolatilityFactor(BaseFactor):
    """
    Low volatility factor for ETFs.

    Lower volatility = higher score (better risk-adjusted returns)
    """

    def __init__(self, lookback: int = 60):
        """
        Initialize volatility factor.

        Args:
            lookback: Days for volatility calculation (default: 60 = ~3 months)
        """
        super().__init__("low_volatility", lookback)

        logger.info(f"Low volatility factor initialized: lookback={lookback}d")

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate low volatility scores.

        Args:
            prices: DataFrame of ETF prices

        Returns:
            pd.Series: Normalized scores (higher = lower volatility = better)
        """
        self.validate_data(prices)

        # Get price window
        price_window = prices.iloc[-self.lookback_period:]
        returns = price_window.pct_change().dropna()

        # Calculate realized volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Log statistics
        logger.debug(
            f"Volatility stats: mean={volatility.mean():.3f}, "
            f"min={volatility.min():.3f}, max={volatility.max():.3f}"
        )

        # Score = -1 * volatility (lower vol = higher score)
        vol_scores = -1 * volatility

        # Normalize
        normalized = self.normalize(vol_scores, method='zscore')

        return normalized

    def calculate_with_beta(self,
                           prices: pd.DataFrame,
                           market_prices: pd.Series) -> pd.Series:
        """
        Calculate low-beta scores (relative to market).

        Low beta = less sensitive to market moves = defensive

        Args:
            prices: ETF prices
            market_prices: Market index prices (e.g., SPY)

        Returns:
            pd.Series: Normalized low-beta scores
        """
        self.validate_data(prices)

        price_window = prices.iloc[-self.lookback_period:]
        market_window = market_prices.iloc[-self.lookback_period:]

        # Calculate returns
        etf_returns = price_window.pct_change().dropna()
        market_returns = market_window.pct_change().dropna()

        # Align indices
        common_dates = etf_returns.index.intersection(market_returns.index)
        etf_returns = etf_returns.loc[common_dates]
        market_returns = market_returns.loc[common_dates]

        # Calculate beta for each ETF
        betas = {}
        for ticker in etf_returns.columns:
            # Beta = Cov(ETF, Market) / Var(Market)
            covariance = etf_returns[ticker].cov(market_returns)
            market_variance = market_returns.var()

            beta = covariance / market_variance
            betas[ticker] = beta

        betas = pd.Series(betas)

        # Score = -1 * beta (lower beta = higher score)
        beta_scores = -1 * betas

        logger.info(
            f"Beta calculation: mean={betas.mean():.2f}, "
            f"range=[{betas.min():.2f}, {betas.max():.2f}]"
        )

        # Normalize
        normalized = self.normalize(beta_scores, method='zscore')

        return normalized
