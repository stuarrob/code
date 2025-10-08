"""
Simple Momentum Signal Generator

Focuses on "run with winners, sell losers" with minimal parameters.
Academic research shows 12-month momentum is most robust.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MomentumSignalGenerator:
    """
    Generate simple momentum signals.

    Philosophy: "Run with winners, sell losers"
    - Winners = positive price momentum
    - Losers = negative price momentum or below stop-loss
    """

    def __init__(
        self,
        momentum_period: int = 126,      # 6-month momentum (most robust)
        rel_strength_enabled: bool = True,  # Compare vs SPY
        trend_filter_enabled: bool = True   # Only buy uptrends
    ):
        """
        Initialize momentum signal generator.

        Parameters
        ----------
        momentum_period : int
            Lookback period for momentum (default 126 days = 6 months)
        rel_strength_enabled : bool
            Include relative strength vs SPY
        trend_filter_enabled : bool
            Filter for price > 200-day SMA
        """
        self.momentum_period = momentum_period
        self.rel_strength_enabled = rel_strength_enabled
        self.trend_filter_enabled = trend_filter_enabled

        logger.info(
            f"Momentum Signal Generator initialized:\n"
            f"  Momentum period: {momentum_period} days\n"
            f"  Relative strength: {rel_strength_enabled}\n"
            f"  Trend filter: {trend_filter_enabled}"
        )

    def generate_signals(
        self,
        prices: pd.DataFrame,
        spy_prices: pd.Series = None
    ) -> pd.Series:
        """
        Generate momentum signals for all ETFs.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical prices for ETFs (columns = tickers)
        spy_prices : pd.Series, optional
            SPY prices for relative strength calculation

        Returns
        -------
        pd.Series
            Momentum signal for each ETF (higher = stronger winner)
        """
        signals = {}

        for ticker in prices.columns:
            price = prices[ticker].dropna()

            # Need at least momentum_period of data
            if len(price) < self.momentum_period:
                continue

            # Calculate momentum (simple price change)
            current_price = price.iloc[-1]
            past_price = price.iloc[-self.momentum_period]
            momentum = (current_price / past_price) - 1

            signal = momentum

            # Add relative strength vs SPY if enabled
            if self.rel_strength_enabled and spy_prices is not None:
                spy_current = spy_prices.iloc[-1]
                spy_past = spy_prices.iloc[-self.momentum_period]
                spy_momentum = (spy_current / spy_past) - 1
                rel_strength = momentum - spy_momentum

                # Combine: 70% absolute momentum, 30% relative strength
                signal = 0.70 * momentum + 0.30 * rel_strength

            # Apply trend filter (only positive signals for uptrends)
            if self.trend_filter_enabled:
                sma_200 = price.rolling(200).mean().iloc[-1] if len(price) >= 200 else price.mean()
                if current_price < sma_200:
                    # Downtrend: reduce signal (but don't zero - allow selling)
                    signal = signal * 0.5

            signals[ticker] = signal

        return pd.Series(signals)

    def get_winner_losers(
        self,
        prices: pd.DataFrame,
        spy_prices: pd.Series = None,
        top_n: int = 20,
        avoid_losers: bool = True
    ) -> dict:
        """
        Identify clear winners and losers.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical prices
        spy_prices : pd.Series, optional
            SPY prices
        top_n : int
            Number of top winners to select
        avoid_losers : bool
            Exclude ETFs with negative momentum

        Returns
        -------
        dict
            {
                'winners': list of tickers,
                'losers': list of tickers,
                'signals': pd.Series of all signals
            }
        """
        signals = self.generate_signals(prices, spy_prices)

        # Sort by signal strength
        sorted_signals = signals.sort_values(ascending=False)

        # Winners: top N with positive momentum
        if avoid_losers:
            positive_signals = sorted_signals[sorted_signals > 0]
            winners = positive_signals.head(top_n).index.tolist()
        else:
            winners = sorted_signals.head(top_n).index.tolist()

        # Losers: negative momentum
        losers = sorted_signals[sorted_signals < 0].index.tolist()

        logger.info(
            f"Winner/Loser Analysis:\n"
            f"  Winners (top {top_n}): {len(winners)}\n"
            f"  Losers (negative momentum): {len(losers)}\n"
            f"  Neutral: {len(signals) - len(winners) - len(losers)}"
        )

        return {
            'winners': winners,
            'losers': losers,
            'signals': signals
        }
