"""
Backtesting Visualizations

Create illustrative graphics for backtest results:
- Portfolio value over time
- Drawdown chart
- Rolling metrics
- Turnover and rebalancing impact
- Contribution growth tracking
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BacktestVisualizer:
    """Create visualizations for backtest results."""

    def __init__(self, output_dir: str = "results/backtests"):
        """
        Initialize visualizer.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_portfolio_value(
        self,
        portfolio_values: pd.Series,
        benchmark: pd.Series = None,
        rebalance_dates: List[pd.Timestamp] = None,
        title: str = "Portfolio Value Over Time",
        filename: str = "portfolio_value.png"
    ) -> str:
        """
        Plot portfolio value over time with optional benchmark.

        Parameters
        ----------
        portfolio_values : pd.Series
            Portfolio values over time
        benchmark : pd.Series, optional
            Benchmark values for comparison
        rebalance_dates : list, optional
            Dates of rebalancing events
        title : str
            Plot title
        filename : str
            Output filename

        Returns
        -------
        str
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot portfolio
        ax.plot(portfolio_values.index, portfolio_values.values,
                label='Portfolio', linewidth=2, color='#2E86AB')

        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to same starting value
            benchmark_normalized = benchmark / benchmark.iloc[0] * portfolio_values.iloc[0]
            ax.plot(benchmark.index, benchmark_normalized.values,
                   label='Benchmark (SPY)', linewidth=2, color='#A23B72', alpha=0.7, linestyle='--')

        # Mark rebalance events
        if rebalance_dates:
            for date in rebalance_dates:
                if date in portfolio_values.index:
                    ax.axvline(date, color='gray', alpha=0.3, linestyle=':', linewidth=1)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_drawdown(
        self,
        portfolio_values: pd.Series,
        title: str = "Portfolio Drawdown",
        filename: str = "drawdown.png"
    ) -> str:
        """
        Plot drawdown over time.

        Parameters
        ----------
        portfolio_values : pd.Series
            Portfolio values
        title : str
            Plot title
        filename : str
            Output filename

        Returns
        -------
        str
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Calculate drawdown
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max

        # Plot
        ax.fill_between(drawdown.index, drawdown.values, 0,
                        color='#C73E1D', alpha=0.6)
        ax.plot(drawdown.index, drawdown.values, color='#8B0000', linewidth=1.5)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
        title: str = "Rolling Performance Metrics",
        filename: str = "rolling_metrics.png"
    ) -> str:
        """
        Plot rolling Sharpe, volatility, and returns.

        Parameters
        ----------
        returns : pd.Series
            Daily returns
        window : int
            Rolling window size (default 252 = 1 year)
        title : str
            Plot title
        filename : str
            Output filename

        Returns
        -------
        str
            Path to saved plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Rolling Sharpe
        excess_returns = returns - (0.04 / 252)  # Assume 4% risk-free rate
        rolling_sharpe = (
            np.sqrt(252) * excess_returns.rolling(window).mean() /
            excess_returns.rolling(window).std()
        )
        axes[0].plot(rolling_sharpe.index, rolling_sharpe.values,
                    color='#2E86AB', linewidth=2)
        axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[0].set_title('Rolling Sharpe Ratio (1-Year Window)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        axes[1].plot(rolling_vol.index, rolling_vol.values,
                    color='#F18F01', linewidth=2)
        axes[1].set_ylabel('Volatility', fontsize=11)
        axes[1].set_title('Rolling Volatility (Annualized)', fontsize=12, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
        axes[1].grid(True, alpha=0.3)

        # Rolling returns
        rolling_ret = returns.rolling(window).mean() * 252
        axes[2].plot(rolling_ret.index, rolling_ret.values,
                    color='#06A77D', linewidth=2)
        axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Annual Return', fontsize=11)
        axes[2].set_xlabel('Date', fontsize=11)
        axes[2].set_title('Rolling Annual Return', fontsize=12, fontweight='bold')
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
        axes[2].grid(True, alpha=0.3)

        # Format x-axes
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_rebalancing_impact(
        self,
        rebalance_events: List,
        title: str = "Rebalancing Impact",
        filename: str = "rebalancing_impact.png"
    ) -> str:
        """
        Plot turnover and costs from rebalancing.

        Parameters
        ----------
        rebalance_events : list
            List of RebalanceEvent objects
        title : str
            Plot title
        filename : str
            Output filename

        Returns
        -------
        str
            Path to saved plot
        """
        if not rebalance_events:
            return None

        dates = [e.date for e in rebalance_events]
        turnovers = [e.turnover * 100 for e in rebalance_events]
        costs = [e.cost for e in rebalance_events]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Turnover
        axes[0].bar(dates, turnovers, color='#2E86AB', alpha=0.7)
        axes[0].set_ylabel('Turnover (%)', fontsize=11)
        axes[0].set_title('Portfolio Turnover per Rebalance', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Costs
        axes[1].bar(dates, costs, color='#C73E1D', alpha=0.7)
        axes[1].set_ylabel('Cost ($)', fontsize=11)
        axes[1].set_xlabel('Date', fontsize=11)
        axes[1].set_title('Transaction Costs per Rebalance', fontsize=12, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        axes[1].grid(True, alpha=0.3, axis='y')

        # Format x-axes
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_contribution_growth(
        self,
        portfolio_df: pd.DataFrame,
        initial_capital: float,
        monthly_contribution: float,
        annual_contribution: float,
        title: str = "Portfolio Growth with Contributions",
        filename: str = "contribution_growth.png"
    ) -> str:
        """
        Plot portfolio growth showing contributions vs investment gains.

        Parameters
        ----------
        portfolio_df : pd.DataFrame
            Portfolio history with 'value' column
        initial_capital : float
            Initial investment
        monthly_contribution : float
            Monthly contribution amount
        annual_contribution : float
            Annual contribution amount
        title : str
            Plot title
        filename : str
            Output filename

        Returns
        -------
        str
            Path to saved plot
        """
        # Calculate cumulative contributions
        dates = portfolio_df.index
        contributions = [initial_capital]
        cumulative = initial_capital

        for i in range(1, len(dates)):
            monthly_added = monthly_contribution if dates[i].month != dates[i-1].month else 0
            annual_added = annual_contribution if dates[i].year != dates[i-1].year else 0
            cumulative += monthly_added + annual_added
            contributions.append(cumulative)

        contributions_series = pd.Series(contributions, index=dates)
        investment_gains = portfolio_df['value'] - contributions_series

        fig, ax = plt.subplots(figsize=(14, 7))

        # Stacked area chart
        ax.fill_between(dates, 0, contributions_series.values,
                        label='Contributions', color='#A8DADC', alpha=0.8)
        ax.fill_between(dates, contributions_series.values, portfolio_df['value'].values,
                        label='Investment Gains', color='#06A77D', alpha=0.8)

        # Portfolio value line
        ax.plot(dates, portfolio_df['value'].values,
               color='#1D3557', linewidth=2, label='Total Value')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def create_all_plots(
        self,
        results: Dict,
        initial_capital: float = 1_000_000,
        monthly_contribution: float = 0,
        annual_contribution: float = 0,
        benchmark: pd.Series = None
    ) -> Dict[str, str]:
        """
        Create all standard backtest plots.

        Parameters
        ----------
        results : dict
            Backtest results from BacktestEngine.run()
        initial_capital : float
            Initial investment
        monthly_contribution : float
            Monthly contribution
        annual_contribution : float
            Annual contribution
        benchmark : pd.Series, optional
            Benchmark series

        Returns
        -------
        dict
            Paths to all generated plots
        """
        portfolio_values = results['portfolio_values']['value']
        returns = results['returns']
        rebalance_events = results['rebalance_events']
        rebalance_dates = [e.date for e in rebalance_events]

        plots = {}

        # Portfolio value
        plots['portfolio_value'] = self.plot_portfolio_value(
            portfolio_values,
            benchmark=benchmark,
            rebalance_dates=rebalance_dates
        )

        # Drawdown
        plots['drawdown'] = self.plot_drawdown(portfolio_values)

        # Rolling metrics
        plots['rolling_metrics'] = self.plot_rolling_metrics(returns)

        # Rebalancing impact
        if rebalance_events:
            plots['rebalancing'] = self.plot_rebalancing_impact(rebalance_events)

        # Contribution growth (if contributions enabled)
        if monthly_contribution > 0 or annual_contribution > 0:
            plots['contribution_growth'] = self.plot_contribution_growth(
                results['portfolio_values'],
                initial_capital,
                monthly_contribution,
                annual_contribution
            )

        return plots
