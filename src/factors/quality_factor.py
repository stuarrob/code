"""
Quality Factor

Based on AQR research: Quality = Profitability + Safety + Growth

For ETFs, quality metrics include:
- Sharpe ratio (return per unit risk)
- Maximum drawdown (resilience in crashes)
- Return stability (consistency of returns)

Reference: "Quality Minus Junk" - AQR (2013)
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


class QualityFactor(BaseFactor):
    """
    Quality factor for ETFs.

    Combines multiple quality metrics:
    1. Sharpe ratio - return efficiency
    2. Drawdown resilience - downside protection
    3. Return stability - predictability
    """

    def __init__(self,
                 lookback: int = 252,
                 risk_free_rate: float = 0.04):
        """
        Initialize quality factor.

        Args:
            lookback: Days for calculation (default: 252 = 1 year)
            risk_free_rate: Annual risk-free rate (default: 4%)
        """
        super().__init__("quality", lookback)
        self.risk_free_rate = risk_free_rate

        logger.info(
            f"Quality factor initialized: lookback={lookback}d, "
            f"risk_free_rate={risk_free_rate:.2%}"
        )

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate quality scores.

        Quality = weighted average of:
        - Sharpe ratio (40%)
        - Drawdown resilience (30%)
        - Return stability (30%)

        Args:
            prices: DataFrame of ETF prices

        Returns:
            pd.Series: Normalized quality scores
        """
        self.validate_data(prices)

        # Get price window for calculation
        price_window = prices.iloc[-self.lookback_period:]
        returns = price_window.pct_change().dropna()

        # Calculate component metrics
        sharpe_scores = self._calculate_sharpe(returns)
        drawdown_scores = self._calculate_drawdown_resilience(price_window)
        stability_scores = self._calculate_return_stability(returns)

        # Log statistics
        logger.debug(
            f"Quality components - "
            f"Sharpe: μ={sharpe_scores.mean():.2f}, "
            f"Drawdown: μ={drawdown_scores.mean():.2f}, "
            f"Stability: μ={stability_scores.mean():.2f}"
        )

        # Normalize each component
        sharpe_norm = self.normalize(sharpe_scores, method='zscore')
        drawdown_norm = self.normalize(drawdown_scores, method='zscore')
        stability_norm = self.normalize(stability_scores, method='zscore')

        # Weighted combination
        quality_score = (
            0.40 * sharpe_norm +
            0.30 * drawdown_norm +
            0.30 * stability_norm
        )

        return quality_score

    def _calculate_sharpe(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate Sharpe ratio for each ETF.

        Sharpe = (Mean Return - Risk Free Rate) / Volatility
        """
        # Annualize returns and volatility
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)

        # Calculate Sharpe
        sharpe = (annual_return - self.risk_free_rate) / annual_vol

        # Handle division by zero
        sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

        return sharpe

    def _calculate_drawdown_resilience(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate drawdown resilience (inverse of max drawdown).

        Lower drawdown = higher resilience = higher quality
        """
        # Calculate cumulative returns
        cumulative = (1 + prices.pct_change()).cumprod()

        # Running maximum
        running_max = cumulative.expanding().max()

        # Drawdown
        drawdown = (cumulative - running_max) / running_max

        # Maximum drawdown (most negative value)
        max_drawdown = drawdown.min()

        # Resilience = -1 * max_drawdown (so lower drawdown = higher score)
        resilience = -1 * max_drawdown

        return resilience

    def _calculate_return_stability(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate return stability (consistency).

        More stable returns = higher quality
        Measured as inverse of return volatility
        """
        # Rolling volatility
        volatility = returns.std() * np.sqrt(252)

        # Stability = -1 * volatility (lower vol = higher stability)
        stability = -1 * volatility

        return stability

    def get_component_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Get individual component scores (useful for analysis).

        Returns:
            pd.DataFrame: Columns = [sharpe, drawdown_resilience, return_stability]
        """
        self.validate_data(prices)

        price_window = prices.iloc[-self.lookback_period:]
        returns = price_window.pct_change().dropna()

        components = pd.DataFrame({
            'sharpe': self._calculate_sharpe(returns),
            'drawdown_resilience': self._calculate_drawdown_resilience(price_window),
            'return_stability': self._calculate_return_stability(returns)
        })

        return components


class DefensiveQualityFactor(QualityFactor):
    """
    Defensive quality factor - emphasizes downside protection.

    Adjusts weights to favor drawdown resilience and stability
    over pure Sharpe ratio.
    """

    def __init__(self, lookback: int = 252, risk_free_rate: float = 0.04):
        super().__init__(lookback, risk_free_rate)
        self.name = "defensive_quality"

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate defensive quality scores.

        Defensive Quality:
        - Sharpe ratio (20%) - less emphasis
        - Drawdown resilience (50%) - high emphasis
        - Return stability (30%) - moderate emphasis
        """
        self.validate_data(prices)

        price_window = prices.iloc[-self.lookback_period:]
        returns = price_window.pct_change().dropna()

        # Calculate components
        sharpe_scores = self._calculate_sharpe(returns)
        drawdown_scores = self._calculate_drawdown_resilience(price_window)
        stability_scores = self._calculate_return_stability(returns)

        # Normalize
        sharpe_norm = self.normalize(sharpe_scores, method='zscore')
        drawdown_norm = self.normalize(drawdown_scores, method='zscore')
        stability_norm = self.normalize(stability_scores, method='zscore')

        # Defensive weighting (emphasis on protection)
        quality_score = (
            0.20 * sharpe_norm +
            0.50 * drawdown_norm +
            0.30 * stability_norm
        )

        logger.info(
            f"Defensive quality: Emphasizing drawdown protection (50%) "
            f"over Sharpe (20%)"
        )

        return quality_score
