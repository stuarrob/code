"""
Performance Metrics for Backtesting

Calculates comprehensive performance metrics including:
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown
- CAGR
- Volatility
- Win Rate
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for backtesting.

    Metrics include:
    - Total Return
    - CAGR (Compound Annual Growth Rate)
    - Volatility (annualized)
    - Sharpe Ratio
    - Sortino Ratio
    - Calmar Ratio
    - Max Drawdown
    - Win Rate
    - Best/Worst periods
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate for Sharpe/Sortino calculations (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(self,
                             portfolio_values: pd.Series,
                             returns: Optional[pd.Series] = None,
                             benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate all performance metrics.

        Parameters
        ----------
        portfolio_values : pd.Series
            Time series of portfolio values
        returns : pd.Series, optional
            Time series of returns (calculated if not provided)
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison

        Returns
        -------
        dict
            Dictionary of performance metrics
        """
        if returns is None:
            returns = portfolio_values.pct_change().dropna()

        # Ensure we have enough data
        if len(returns) < 2:
            logger.warning("Insufficient data for metrics calculation")
            return self._empty_metrics()

        # Calculate metrics
        metrics = {}

        # Basic metrics
        metrics['total_return'] = self.total_return(portfolio_values)
        metrics['cagr'] = self.cagr(portfolio_values)
        metrics['volatility'] = self.volatility(returns)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self.sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.sortino_ratio(returns)
        metrics['calmar_ratio'] = self.calmar_ratio(portfolio_values, returns)

        # Drawdown metrics
        drawdown_info = self.drawdown_metrics(portfolio_values)
        metrics.update(drawdown_info)

        # Win rate
        metrics['win_rate'] = self.win_rate(returns)
        metrics['avg_win'] = self.average_win(returns)
        metrics['avg_loss'] = self.average_loss(returns)

        # Best/worst periods
        metrics['best_day'] = returns.max()
        metrics['worst_day'] = returns.min()
        metrics['best_month'] = self.best_month(returns)
        metrics['worst_month'] = self.worst_month(returns)

        # Additional stats
        metrics['positive_days'] = (returns > 0).sum()
        metrics['negative_days'] = (returns < 0).sum()
        metrics['total_days'] = len(returns)

        # Benchmark comparison
        if benchmark_returns is not None:
            metrics['alpha'] = self.alpha(returns, benchmark_returns)
            metrics['beta'] = self.beta(returns, benchmark_returns)
            metrics['information_ratio'] = self.information_ratio(returns, benchmark_returns)

        return metrics

    def total_return(self, portfolio_values: pd.Series) -> float:
        """Calculate total return."""
        if len(portfolio_values) < 2:
            return 0.0
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0

    def cagr(self, portfolio_values: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(portfolio_values) < 2:
            return 0.0

        total_return = self.total_return(portfolio_values)
        years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25

        if years <= 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1.0

    def volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year)

    def sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Volatility
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)

        if excess_returns.std() == 0:
            return 0.0

        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    def sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio.

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        Only penalizes downside volatility.
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std() * np.sqrt(periods_per_year)

        return (excess_returns.mean() * periods_per_year) / downside_std

    def calmar_ratio(self, portfolio_values: pd.Series, returns: pd.Series) -> float:
        """
        Calculate Calmar Ratio.

        Calmar = CAGR / Max Drawdown
        """
        max_dd = self.max_drawdown(portfolio_values)

        if max_dd == 0:
            return 0.0

        return self.cagr(portfolio_values) / abs(max_dd)

    def max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) < 2:
            return 0.0

        # Calculate running maximum
        running_max = portfolio_values.expanding().max()

        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max

        return drawdown.min()

    def drawdown_metrics(self, portfolio_values: pd.Series) -> Dict:
        """
        Calculate comprehensive drawdown metrics.

        Returns
        -------
        dict
            max_drawdown, max_drawdown_duration, current_drawdown
        """
        if len(portfolio_values) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'current_drawdown': 0.0
            }

        # Calculate running maximum
        running_max = portfolio_values.expanding().max()

        # Calculate drawdown series
        drawdown = (portfolio_values - running_max) / running_max

        # Max drawdown
        max_dd = drawdown.min()

        # Current drawdown
        current_dd = drawdown.iloc[-1]

        # Max drawdown duration
        is_underwater = drawdown < 0
        underwater_periods = is_underwater.astype(int).groupby(
            (is_underwater != is_underwater.shift()).cumsum()
        ).sum()

        max_dd_duration = underwater_periods.max() if len(underwater_periods) > 0 else 0

        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_dd_duration,
            'current_drawdown': current_dd
        }

    def win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive periods)."""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)

    def average_win(self, returns: pd.Series) -> float:
        """Calculate average winning return."""
        wins = returns[returns > 0]
        return wins.mean() if len(wins) > 0 else 0.0

    def average_loss(self, returns: pd.Series) -> float:
        """Calculate average losing return."""
        losses = returns[returns < 0]
        return losses.mean() if len(losses) > 0 else 0.0

    def best_month(self, returns: pd.Series) -> float:
        """Calculate best monthly return."""
        if len(returns) < 20:
            return 0.0

        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly_returns.max() if len(monthly_returns) > 0 else 0.0

    def worst_month(self, returns: pd.Series) -> float:
        """Calculate worst monthly return."""
        if len(returns) < 20:
            return 0.0

        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly_returns.min() if len(monthly_returns) > 0 else 0.0

    def alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Jensen's alpha vs benchmark."""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        # Align series
        aligned_returns, aligned_bench = returns.align(benchmark_returns, join='inner')

        if len(aligned_returns) < 2:
            return 0.0

        # Calculate beta first
        beta = self.beta(aligned_returns, aligned_bench)

        # Alpha = Portfolio Return - (Risk Free + Beta * (Benchmark Return - Risk Free))
        portfolio_return = aligned_returns.mean() * 252
        benchmark_return = aligned_bench.mean() * 252

        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))

        return alpha

    def beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta vs benchmark."""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        # Align series
        aligned_returns, aligned_bench = returns.align(benchmark_returns, join='inner')

        if len(aligned_returns) < 2:
            return 0.0

        # Covariance / Variance
        covariance = aligned_returns.cov(aligned_bench)
        benchmark_variance = aligned_bench.var()

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio.

        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        # Align series
        aligned_returns, aligned_bench = returns.align(benchmark_returns, join='inner')

        if len(aligned_returns) < 2:
            return 0.0

        # Excess returns
        excess_returns = aligned_returns - aligned_bench

        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        return (excess_returns.mean() * 252) / tracking_error

    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'current_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'best_day': 0.0,
            'worst_day': 0.0,
            'best_month': 0.0,
            'worst_month': 0.0,
            'positive_days': 0,
            'negative_days': 0,
            'total_days': 0
        }

    def format_metrics(self, metrics: Dict) -> str:
        """Format metrics for display."""
        lines = []
        lines.append("=" * 60)
        lines.append("PERFORMANCE METRICS")
        lines.append("=" * 60)

        lines.append(f"\nReturns:")
        lines.append(f"  Total Return:        {metrics['total_return']:>10.2%}")
        lines.append(f"  CAGR:                {metrics['cagr']:>10.2%}")
        lines.append(f"  Volatility:          {metrics['volatility']:>10.2%}")

        lines.append(f"\nRisk-Adjusted:")
        lines.append(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        lines.append(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        lines.append(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

        lines.append(f"\nDrawdown:")
        lines.append(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        lines.append(f"  Max DD Duration:     {metrics['max_drawdown_duration']:>10.0f} days")
        lines.append(f"  Current Drawdown:    {metrics['current_drawdown']:>10.2%}")

        lines.append(f"\nWin/Loss:")
        lines.append(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        lines.append(f"  Average Win:         {metrics['avg_win']:>10.2%}")
        lines.append(f"  Average Loss:        {metrics['avg_loss']:>10.2%}")

        lines.append(f"\nBest/Worst:")
        lines.append(f"  Best Day:            {metrics['best_day']:>10.2%}")
        lines.append(f"  Worst Day:           {metrics['worst_day']:>10.2%}")
        lines.append(f"  Best Month:          {metrics['best_month']:>10.2%}")
        lines.append(f"  Worst Month:         {metrics['worst_month']:>10.2%}")

        if 'alpha' in metrics:
            lines.append(f"\nBenchmark Comparison:")
            lines.append(f"  Alpha:               {metrics['alpha']:>10.2%}")
            lines.append(f"  Beta:                {metrics['beta']:>10.2f}")
            lines.append(f"  Information Ratio:   {metrics['information_ratio']:>10.2f}")

        lines.append("=" * 60)

        return "\n".join(lines)
