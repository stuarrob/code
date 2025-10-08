"""
Performance Metrics Calculator

Calculates comprehensive performance metrics for portfolio backtesting:
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- CAGR
- Volatility
- Win Rate
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate performance metrics for portfolio returns."""

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize performance metrics calculator.

        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate (default 4%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series
        portfolio_values : pd.Series
            Portfolio value series

        Returns
        -------
        dict
            Dictionary of all metrics
        """
        metrics = {}

        # Basic metrics
        metrics['total_return'] = self._total_return(portfolio_values)
        metrics['cagr'] = self._cagr(portfolio_values, returns)
        metrics['volatility'] = self._volatility(returns)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self._sharpe_ratio(returns)
        metrics['sortino_ratio'] = self._sortino_ratio(returns)
        metrics['calmar_ratio'] = self._calmar_ratio(returns, portfolio_values)

        # Drawdown metrics
        metrics['max_drawdown'] = self._max_drawdown(portfolio_values)
        metrics['avg_drawdown'] = self._avg_drawdown(portfolio_values)
        metrics['max_drawdown_duration'] = self._max_drawdown_duration(portfolio_values)

        # Distribution metrics
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        metrics['win_rate'] = (returns > 0).sum() / len(returns)

        # Return statistics
        metrics['best_day'] = returns.max()
        metrics['worst_day'] = returns.min()
        metrics['avg_daily_return'] = returns.mean()
        metrics['median_daily_return'] = returns.median()

        return metrics

    def _total_return(self, portfolio_values: pd.Series) -> float:
        """Calculate total return."""
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    def _cagr(self, portfolio_values: pd.Series, returns: pd.Series) -> float:
        """
        Calculate Compound Annual Growth Rate.

        CAGR = (Ending Value / Beginning Value)^(1/years) - 1
        """
        total_return = self._total_return(portfolio_values)
        years = len(returns) / 252  # Assume 252 trading days per year

        if years == 0:
            return 0.0

        cagr = (1 + total_return) ** (1 / years) - 1
        return cagr

    def _volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(252)

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe Ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        excess_returns = returns - (self.risk_free_rate / 252)

        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino Ratio.

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        Only penalizes downside volatility.
        """
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return sortino

    def _calmar_ratio(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series
    ) -> float:
        """
        Calculate Calmar Ratio.

        Calmar = CAGR / Max Drawdown
        Measures return relative to worst drawdown.
        """
        cagr = self._cagr(portfolio_values, returns)
        max_dd = self._max_drawdown(portfolio_values)

        if max_dd == 0:
            return 0.0

        return cagr / abs(max_dd)

    def _max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Max DD = (Trough Value - Peak Value) / Peak Value
        """
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        return drawdown.min()

    def _avg_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate average drawdown."""
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        # Only include negative drawdowns
        negative_dd = drawdown[drawdown < 0]
        if len(negative_dd) == 0:
            return 0.0
        return negative_dd.mean()

    def _max_drawdown_duration(self, portfolio_values: pd.Series) -> int:
        """
        Calculate maximum drawdown duration in days.

        Returns the longest period between new highs.
        """
        cumulative_max = portfolio_values.expanding().max()
        is_new_high = portfolio_values >= cumulative_max

        # Find periods between new highs
        new_high_indices = np.where(is_new_high)[0]

        if len(new_high_indices) <= 1:
            return len(portfolio_values)

        max_duration = np.diff(new_high_indices).max()
        return int(max_duration)

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics over specified window.

        Parameters
        ----------
        returns : pd.Series
            Daily returns
        window : int
            Rolling window size (default 252 days = 1 year)

        Returns
        -------
        pd.DataFrame
            Rolling metrics
        """
        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling Sharpe
        excess_returns = returns - (self.risk_free_rate / 252)
        rolling_metrics['sharpe'] = (
            np.sqrt(252) * excess_returns.rolling(window).mean() /
            excess_returns.rolling(window).std()
        )

        # Rolling volatility
        rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling return
        rolling_metrics['return'] = returns.rolling(window).mean() * 252

        return rolling_metrics

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Pretty print performance metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics from calculate_all_metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)

        print("\n📈 RETURN METRICS")
        print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
        print(f"  CAGR:                {metrics['cagr']*100:>8.2f}%")
        print(f"  Volatility:          {metrics['volatility']*100:>8.2f}%")

        print("\n⚖️  RISK-ADJUSTED METRICS")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>8.2f}")

        print("\n📉 DRAWDOWN METRICS")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  Avg Drawdown:        {metrics['avg_drawdown']*100:>8.2f}%")
        print(f"  Max DD Duration:     {metrics['max_drawdown_duration']:>8.0f} days")

        print("\n📊 DISTRIBUTION METRICS")
        print(f"  Win Rate:            {metrics['win_rate']*100:>8.2f}%")
        print(f"  Skewness:            {metrics['skewness']:>8.2f}")
        print(f"  Kurtosis:            {metrics['kurtosis']:>8.2f}")

        print("\n📅 DAILY RETURN STATISTICS")
        print(f"  Average:             {metrics['avg_daily_return']*100:>8.4f}%")
        print(f"  Median:              {metrics['median_daily_return']*100:>8.4f}%")
        print(f"  Best Day:            {metrics['best_day']*100:>8.2f}%")
        print(f"  Worst Day:           {metrics['worst_day']*100:>8.2f}%")

        print("="*60 + "\n")
