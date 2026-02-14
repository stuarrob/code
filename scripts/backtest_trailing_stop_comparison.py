#!/usr/bin/env python3
"""
Trailing Stop Strategy Comparison Backtest

Compares different stop-loss strategies:
1. Fixed entry stop (-12% from entry price) - CURRENT
2. Trailing stop (X% from peak price) - NEW

Tests multiple trailing stop percentages to find optimal value.

Usage:
    python scripts/backtest_trailing_stop_comparison.py
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils.dataframe import dataframe_to_rows

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.factors import (
    FactorIntegrator,
    MomentumFactor,
    QualityFactor,
    SimplifiedValueFactor,
    VolatilityFactor,
)

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Quieter for comparison runs
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = Path.home() / "trading"


@dataclass
class Position:
    """Track individual position with stop-loss state"""
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    peak_price: float = field(init=False)
    peak_date: pd.Timestamp = field(init=False)

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.peak_date = self.entry_date

    def update_peak(self, current_price: float, current_date: pd.Timestamp):
        if current_price > self.peak_price:
            self.peak_price = current_price
            self.peak_date = current_date

    def current_gain(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price

    def drawdown_from_peak(self, current_price: float) -> float:
        return (current_price - self.peak_price) / self.peak_price

    def days_held(self, current_date: pd.Timestamp) -> int:
        return (current_date - self.entry_date).days


class StopLossStrategy:
    """Configurable stop-loss strategy for backtesting"""

    def __init__(
        self,
        name: str,
        initial_capital: float = 100000,
        monthly_addition: float = 5000,
        num_positions: int = 20,
        # Stop-loss parameters
        use_entry_stop: bool = True,
        entry_stop_pct: float = 0.12,  # -12% from entry
        use_trailing_stop: bool = False,
        trailing_stop_pct: float = 0.10,  # -10% from peak
        trailing_activation_pct: float = 0.0,  # Activate immediately (0) or after X% gain
    ):
        self.name = name
        self.initial_capital = initial_capital
        self.monthly_addition = monthly_addition
        self.num_positions = num_positions

        # Stop-loss config
        self.use_entry_stop = use_entry_stop
        self.entry_stop_pct = entry_stop_pct
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activation_pct = trailing_activation_pct

        # State
        self.positions: Dict[str, Position] = {}
        self.cash = 0
        self.total_contributions = initial_capital
        self.reserve = 0

        # History
        self.portfolio_history = []
        self.trades_history = []
        self.stop_loss_history = []

    def reset(self):
        """Reset state for new run"""
        self.positions = {}
        self.cash = 0
        self.total_contributions = self.initial_capital
        self.reserve = 0
        self.portfolio_history = []
        self.trades_history = []
        self.stop_loss_history = []

    def check_stop_losses(self, prices: pd.DataFrame, date: pd.Timestamp) -> List[Tuple]:
        """Check all positions for stop-loss triggers"""
        to_sell = []
        current_prices = prices.loc[date]

        for ticker, position in self.positions.items():
            if ticker not in current_prices.index or pd.isna(current_prices[ticker]):
                continue

            current_price = current_prices[ticker]
            position.update_peak(current_price, date)

            loss_from_entry = position.current_gain(current_price)
            loss_from_peak = position.drawdown_from_peak(current_price)
            gain = position.current_gain(current_price)

            # Check entry stop (fixed stop from entry price)
            if self.use_entry_stop and loss_from_entry < -self.entry_stop_pct:
                to_sell.append((ticker, f"Entry stop (-{self.entry_stop_pct:.0%})", current_price, loss_from_entry))
                continue

            # Check trailing stop (from peak price)
            if self.use_trailing_stop:
                # Only activate trailing stop if gain threshold met (or threshold is 0)
                if gain >= self.trailing_activation_pct:
                    if loss_from_peak < -self.trailing_stop_pct:
                        to_sell.append((ticker, f"Trail stop (-{self.trailing_stop_pct:.0%} from peak)", current_price, loss_from_entry))

        return to_sell

    def execute_sells(self, to_sell: List[Tuple], date: pd.Timestamp):
        """Execute stop-loss sells"""
        for ticker, reason, price, gain in to_sell:
            position = self.positions[ticker]
            value = position.shares * price
            self.cash += value

            self.trades_history.append({
                "date": date,
                "ticker": ticker,
                "action": "SELL",
                "shares": position.shares,
                "price": price,
                "value": value,
                "reason": reason,
                "gain": gain,
                "days_held": position.days_held(date),
            })

            self.stop_loss_history.append({
                "date": date,
                "ticker": ticker,
                "entry_price": position.entry_price,
                "exit_price": price,
                "peak_price": position.peak_price,
                "gain": gain,
                "reason": reason,
            })

            del self.positions[ticker]

    def fill_positions(self, factor_scores: pd.Series, prices: pd.DataFrame, date: pd.Timestamp):
        """Buy new positions to fill empty slots"""
        current_tickers = set(self.positions.keys())
        num_empty = self.num_positions - len(self.positions)

        if num_empty <= 0 or self.cash <= 0:
            return

        total_value = self.get_portfolio_value(prices, date) + self.cash
        target_value = total_value / self.num_positions

        valid_scores = factor_scores.dropna()
        candidates = valid_scores[~valid_scores.index.isin(current_tickers)].sort_values(ascending=False)
        current_prices = prices.loc[date]

        bought = 0
        for ticker in candidates.index:
            if bought >= num_empty:
                break
            if ticker not in current_prices.index or pd.isna(current_prices[ticker]):
                continue

            price = current_prices[ticker]
            if price <= 0:
                continue

            shares = int(min(target_value, self.cash) / price)
            if shares <= 0:
                continue

            cost = shares * price
            self.positions[ticker] = Position(ticker, shares, price, date)
            self.cash -= cost
            bought += 1

            self.trades_history.append({
                "date": date,
                "ticker": ticker,
                "action": "BUY",
                "shares": shares,
                "price": price,
                "value": cost,
                "reason": "Fill slot",
                "gain": None,
                "days_held": None,
            })

    def get_portfolio_value(self, prices: pd.DataFrame, date: pd.Timestamp) -> float:
        if not self.positions:
            return 0.0
        current_prices = prices.loc[date]
        total = 0.0
        for ticker, pos in self.positions.items():
            if ticker in current_prices.index and not pd.isna(current_prices[ticker]):
                total += pos.shares * current_prices[ticker]
        return total


def calculate_factors(prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Calculate factor scores at given date"""
    historical = prices.loc[:date]
    if len(historical) < 252:
        return pd.Series(dtype=float)

    calculators = {
        "momentum": MomentumFactor(lookback=252, skip_recent=21),
        "quality": QualityFactor(lookback=252),
        "value": SimplifiedValueFactor(),
        "volatility": VolatilityFactor(lookback=60),
    }
    integrator = FactorIntegrator(factor_weights={"momentum": 0.25, "quality": 0.25, "value": 0.25, "volatility": 0.25})

    scores_dict = {}
    for name, calc in calculators.items():
        try:
            if name == "value":
                expense = pd.Series(np.random.uniform(0.0005, 0.01, len(historical.columns)), index=historical.columns)
                scores = calc.calculate(historical, expense)
            else:
                scores = calc.calculate(historical)
            scores_dict[name] = scores
        except:
            scores_dict[name] = pd.Series(dtype=float)

    return integrator.integrate(pd.DataFrame(scores_dict))


def run_backtest(strategy: StopLossStrategy, prices: pd.DataFrame, start_date: str, end_date: str) -> Dict:
    """Run backtest for a single strategy"""
    strategy.reset()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if prices.index.tz is not None:
        if start.tz is None:
            start = start.tz_localize(prices.index.tz)
        if end.tz is None:
            end = end.tz_localize(prices.index.tz)

    trading_days = prices.index[(prices.index >= start) & (prices.index <= end)]

    # Get weekly dates
    check_dates = []
    current_week = None
    for date in trading_days:
        week_key = (date.isocalendar()[0], date.isocalendar()[1])
        if week_key != current_week:
            check_dates.append(date)
            current_week = week_key

    # Capital deployment: 59% immediate, rest in reserve
    immediate = strategy.initial_capital * 0.59
    strategy.reserve = strategy.initial_capital - immediate
    strategy.cash = immediate

    # Track for monthly contributions and reserve deployment
    last_month = None
    last_reserve_deploy = None
    factor_scores = None
    weeks_since_factor_calc = 0

    for i, date in enumerate(check_dates):
        current_month = date.month

        # Monthly contribution
        if last_month is not None and current_month != last_month:
            strategy.cash += strategy.monthly_addition
            strategy.total_contributions += strategy.monthly_addition
        last_month = current_month

        # Deploy from reserve monthly
        if strategy.reserve > 0:
            if last_reserve_deploy is None or (date - last_reserve_deploy).days >= 28:
                deploy = min(10000, strategy.reserve)
                strategy.cash += deploy
                strategy.reserve -= deploy
                last_reserve_deploy = date

        # Check stop-losses
        to_sell = strategy.check_stop_losses(prices, date)
        if to_sell:
            strategy.execute_sells(to_sell, date)

        # Recalculate factors every 2 weeks
        if weeks_since_factor_calc >= 2 or factor_scores is None:
            factor_scores = calculate_factors(prices, date)
            weeks_since_factor_calc = 0
        weeks_since_factor_calc += 1

        # Fill empty positions
        if factor_scores is not None and len(factor_scores) > 0:
            strategy.fill_positions(factor_scores, prices, date)

        # Record portfolio state
        portfolio_value = strategy.get_portfolio_value(prices, date)
        total_value = portfolio_value + strategy.cash + strategy.reserve

        strategy.portfolio_history.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "cash": strategy.cash,
            "reserve": strategy.reserve,
            "total_value": total_value,
            "num_positions": len(strategy.positions),
            "contributions": strategy.total_contributions,
        })

    # Calculate metrics
    df = pd.DataFrame(strategy.portfolio_history)
    if len(df) < 2:
        return {"error": "Insufficient data"}

    # Calculate returns
    df["return"] = df["total_value"].pct_change()
    total_return = (df["total_value"].iloc[-1] - strategy.initial_capital) / strategy.initial_capital

    # Annualized return
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Risk metrics
    returns = df["return"].dropna()
    volatility = returns.std() * np.sqrt(52)  # Weekly to annual
    risk_free_rate = 0.04  # Assume 4% risk-free

    # Sharpe Ratio
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0

    # Sortino Ratio (only penalize downside volatility)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(52) if len(downside_returns) > 0 else 0
    sortino = (cagr - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0

    # Drawdown
    cummax = df["total_value"].cummax()
    drawdown = (df["total_value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar Ratio (CAGR / Max Drawdown)
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Trade stats
    sells = [t for t in strategy.trades_history if t["action"] == "SELL"]
    buys = [t for t in strategy.trades_history if t["action"] == "BUY"]
    num_stop_losses = len(sells)
    avg_stop_loss = np.mean([t["gain"] for t in sells]) if sells else 0
    win_rate = len([t for t in sells if t["gain"] > 0]) / len(sells) if sells else 0

    # Turnover metrics (to detect whipsawing)
    total_trades = len(strategy.trades_history)
    weeks_in_period = len(df)
    turnover_per_week = total_trades / weeks_in_period if weeks_in_period > 0 else 0

    # Composite Score (research-based)
    # Based on academic literature:
    # - Sharpe and Sortino are complementary (overall vs downside risk)
    # - Calmar captures drawdown risk which is crucial for practitioners
    # - Turnover penalty to avoid whipsawing (transaction cost drag)
    #
    # Score = 0.30 * norm_Sharpe + 0.30 * norm_Sortino + 0.25 * norm_Calmar + 0.15 * (1 - norm_turnover)
    # We'll calculate this after all strategies are run (needs normalization)

    return {
        "strategy": strategy.name,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "num_stop_losses": num_stop_losses,
        "avg_stop_loss": avg_stop_loss,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "turnover_per_week": turnover_per_week,
        "final_value": df["total_value"].iloc[-1],
        "portfolio_history": df,
        "trades": strategy.trades_history,
        "stop_losses": strategy.stop_loss_history,
    }


def main():
    print("=" * 80)
    print("TRAILING STOP STRATEGY COMPARISON")
    print("=" * 80)

    # Load prices
    prices_file = DATA_DIR / "processed" / "etf_prices_filtered.parquet"
    prices = pd.read_parquet(prices_file)
    print(f"Loaded {len(prices.columns)} ETFs, {len(prices)} days")

    start_date = "2025-01-01"
    end_date = "2025-12-11"
    print(f"Period: {start_date} to {end_date}")

    # Define strategies to compare
    strategies = [
        # Current strategy: Entry stop only
        StopLossStrategy(
            name="Entry Stop -12% (Current)",
            use_entry_stop=True,
            entry_stop_pct=0.12,
            use_trailing_stop=False,
        ),
        # Trailing stop variations
        StopLossStrategy(
            name="Trail -8% (from peak)",
            use_entry_stop=False,
            use_trailing_stop=True,
            trailing_stop_pct=0.08,
            trailing_activation_pct=0.0,
        ),
        StopLossStrategy(
            name="Trail -10% (from peak)",
            use_entry_stop=False,
            use_trailing_stop=True,
            trailing_stop_pct=0.10,
            trailing_activation_pct=0.0,
        ),
        StopLossStrategy(
            name="Trail -12% (from peak)",
            use_entry_stop=False,
            use_trailing_stop=True,
            trailing_stop_pct=0.12,
            trailing_activation_pct=0.0,
        ),
        StopLossStrategy(
            name="Trail -15% (from peak)",
            use_entry_stop=False,
            use_trailing_stop=True,
            trailing_stop_pct=0.15,
            trailing_activation_pct=0.0,
        ),
        # Hybrid: Entry stop + trailing
        StopLossStrategy(
            name="Hybrid: Entry -12% + Trail -8%",
            use_entry_stop=True,
            entry_stop_pct=0.12,
            use_trailing_stop=True,
            trailing_stop_pct=0.08,
            trailing_activation_pct=0.05,  # Activate trail after 5% gain
        ),
        StopLossStrategy(
            name="Hybrid: Entry -12% + Trail -10%",
            use_entry_stop=True,
            entry_stop_pct=0.12,
            use_trailing_stop=True,
            trailing_stop_pct=0.10,
            trailing_activation_pct=0.05,
        ),
    ]

    # Run backtests
    results = []
    print("\nRunning backtests...")
    for strategy in strategies:
        print(f"  Testing: {strategy.name}...")
        result = run_backtest(strategy, prices, start_date, end_date)
        results.append(result)
        print(f"    Return: {result['total_return']:.2%}, Sharpe: {result['sharpe']:.2f}, Sortino: {result['sortino']:.2f}, Max DD: {result['max_drawdown']:.2%}")

    # Calculate Composite Score (normalized across strategies)
    # Based on research: Sharpe + Sortino + Calmar - Turnover penalty
    sharpes = np.array([r["sharpe"] for r in results])
    sortinos = np.array([r["sortino"] for r in results])
    calmars = np.array([r["calmar"] for r in results])
    turnovers = np.array([r["turnover_per_week"] for r in results])

    # Normalize each metric to 0-1 scale
    def normalize(arr):
        if arr.max() == arr.min():
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / (arr.max() - arr.min())

    norm_sharpe = normalize(sharpes)
    norm_sortino = normalize(sortinos)
    norm_calmar = normalize(calmars)
    norm_turnover = normalize(turnovers)  # Higher turnover = worse

    # Composite Score:
    # - 30% Sharpe (overall risk-adjusted return)
    # - 30% Sortino (downside risk focus)
    # - 25% Calmar (drawdown-adjusted return)
    # - 15% Turnover penalty (lower is better to avoid whipsawing)
    composite_scores = (
        0.30 * norm_sharpe +
        0.30 * norm_sortino +
        0.25 * norm_calmar +
        0.15 * (1 - norm_turnover)  # Invert turnover so lower is better
    )

    # Add composite score to results
    for i, r in enumerate(results):
        r["composite_score"] = composite_scores[i]

    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        "Strategy": r["strategy"],
        "Total Return": r["total_return"],
        "CAGR": r["cagr"],
        "Sharpe": r["sharpe"],
        "Sortino": r["sortino"],
        "Calmar": r["calmar"],
        "Max Drawdown": r["max_drawdown"],
        "Volatility": r["volatility"],
        "Stop-Losses": r["num_stop_losses"],
        "Total Trades": r["total_trades"],
        "Turnover/Week": r["turnover_per_week"],
        "Composite Score": r["composite_score"],
        "Final Value": r["final_value"],
    } for r in results])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Find best strategy
    best_sharpe_idx = summary_df["Sharpe"].idxmax()
    best_sortino_idx = summary_df["Sortino"].idxmax()
    best_calmar_idx = summary_df["Calmar"].idxmax()
    best_composite_idx = summary_df["Composite Score"].idxmax()
    best_return_idx = summary_df["Total Return"].idxmax()
    best_dd_idx = summary_df["Max Drawdown"].idxmax()  # Least negative

    print("\n" + "-" * 40)
    print("BEST PERFORMERS:")
    print("-" * 40)
    print(f" Best Sharpe Ratio:    {summary_df.loc[best_sharpe_idx, 'Strategy']} ({summary_df.loc[best_sharpe_idx, 'Sharpe']:.2f})")
    print(f" Best Sortino Ratio:   {summary_df.loc[best_sortino_idx, 'Strategy']} ({summary_df.loc[best_sortino_idx, 'Sortino']:.2f})")
    print(f" Best Calmar Ratio:    {summary_df.loc[best_calmar_idx, 'Strategy']} ({summary_df.loc[best_calmar_idx, 'Calmar']:.2f})")
    print(f" Best Composite Score: {summary_df.loc[best_composite_idx, 'Strategy']} ({summary_df.loc[best_composite_idx, 'Composite Score']:.3f})")
    print(f" Best Return:          {summary_df.loc[best_return_idx, 'Strategy']} ({summary_df.loc[best_return_idx, 'Total Return']:.2%})")
    print(f" Best Max DD:          {summary_df.loc[best_dd_idx, 'Strategy']} ({summary_df.loc[best_dd_idx, 'Max Drawdown']:.2%})")

    print("\n" + "-" * 40)
    print("COMPOSITE SCORE METHODOLOGY:")
    print("-" * 40)
    print("  30% Sharpe Ratio (overall risk-adjusted return)")
    print("  30% Sortino Ratio (downside risk focus)")
    print("  25% Calmar Ratio (drawdown-adjusted return)")
    print("  15% Turnover Penalty (lower trades = less whipsawing)")

    # Create visualizations
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle("Trailing Stop Strategy Comparison - Full Analysis\n2025 YTD", fontsize=14, fontweight="bold")

    strategy_names = [r["strategy"] for r in results]

    # 1. Portfolio Value Over Time
    ax1 = axes[0, 0]
    for result in results:
        df = result["portfolio_history"]
        ax1.plot(df["date"], df["total_value"], label=result["strategy"], linewidth=1.5)
    ax1.axhline(y=100000, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("Portfolio Value Over Time", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}k"))

    # 2. Composite Score (THE KEY METRIC)
    ax2 = axes[0, 1]
    composite_scores = [r["composite_score"] for r in results]
    colors_composite = plt.cm.RdYlGn(np.array(composite_scores))
    bars2 = ax2.barh(range(len(strategy_names)), composite_scores, color=colors_composite, alpha=0.8, edgecolor="black")
    ax2.set_yticks(range(len(strategy_names)))
    ax2.set_yticklabels(strategy_names, fontsize=9)
    ax2.set_xlabel("Composite Score (0-1)")
    ax2.set_title("COMPOSITE SCORE (30% Sharpe + 30% Sortino + 25% Calmar + 15% Low Turnover)", fontweight="bold", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars2, composite_scores):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    # Highlight best
    best_idx = np.argmax(composite_scores)
    bars2[best_idx].set_edgecolor("gold")
    bars2[best_idx].set_linewidth(3)

    # 3. Sharpe Ratio Comparison
    ax3 = axes[1, 0]
    sharpes_plot = [r["sharpe"] for r in results]
    colors_bar = ["#2E7D32" if s > 1.0 else "#FF8F00" if s > 0.5 else "#C62828" for s in sharpes_plot]
    bars3 = ax3.barh(range(len(strategy_names)), sharpes_plot, color=colors_bar, alpha=0.7)
    ax3.set_yticks(range(len(strategy_names)))
    ax3.set_yticklabels(strategy_names, fontsize=9)
    ax3.set_xlabel("Sharpe Ratio")
    ax3.set_title("Sharpe Ratio (Overall Risk-Adjusted Return)", fontweight="bold")
    ax3.axvline(x=1.0, color="green", linestyle="--", alpha=0.5)
    ax3.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars3, sharpes_plot):
        ax3.text(val + 0.05, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=9)

    # 4. Sortino Ratio Comparison
    ax4 = axes[1, 1]
    sortinos_plot = [r["sortino"] for r in results]
    colors_sortino = ["#2E7D32" if s > 2.0 else "#FF8F00" if s > 1.0 else "#C62828" for s in sortinos_plot]
    bars4 = ax4.barh(range(len(strategy_names)), sortinos_plot, color=colors_sortino, alpha=0.7)
    ax4.set_yticks(range(len(strategy_names)))
    ax4.set_yticklabels(strategy_names, fontsize=9)
    ax4.set_xlabel("Sortino Ratio")
    ax4.set_title("Sortino Ratio (Downside Risk Focus)", fontweight="bold")
    ax4.axvline(x=2.0, color="green", linestyle="--", alpha=0.5)
    ax4.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars4, sortinos_plot):
        ax4.text(val + 0.05, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=9)

    # 5. Calmar Ratio (Return / Max Drawdown)
    ax5 = axes[2, 0]
    calmars_plot = [r["calmar"] for r in results]
    colors_calmar = ["#2E7D32" if c > 3.0 else "#FF8F00" if c > 1.0 else "#C62828" for c in calmars_plot]
    bars5 = ax5.barh(range(len(strategy_names)), calmars_plot, color=colors_calmar, alpha=0.7)
    ax5.set_yticks(range(len(strategy_names)))
    ax5.set_yticklabels(strategy_names, fontsize=9)
    ax5.set_xlabel("Calmar Ratio")
    ax5.set_title("Calmar Ratio (CAGR / Max Drawdown)", fontweight="bold")
    ax5.axvline(x=3.0, color="green", linestyle="--", alpha=0.5)
    ax5.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars5, calmars_plot):
        ax5.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=9)

    # 6. Turnover / Trades (lower is better - avoids whipsawing)
    ax6 = axes[2, 1]
    trades_plot = [r["total_trades"] for r in results]
    # Color: Green for low trades, red for high (inverse)
    max_trades = max(trades_plot)
    colors_trades = [plt.cm.RdYlGn(1 - t/max_trades) for t in trades_plot]
    bars6 = ax6.barh(range(len(strategy_names)), trades_plot, color=colors_trades, alpha=0.7)
    ax6.set_yticks(range(len(strategy_names)))
    ax6.set_yticklabels(strategy_names, fontsize=9)
    ax6.set_xlabel("Total Trades")
    ax6.set_title("Total Trades (Lower = Less Whipsawing)", fontweight="bold")
    ax6.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars6, trades_plot):
        ax6.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "trailing_stop_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'trailing_stop_comparison.png'}")

    # Save to Excel
    excel_file = RESULTS_DIR / f"trailing_stop_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Portfolio values for each strategy
        for i, result in enumerate(results):
            # Clean sheet name - remove invalid characters and limit length
            sheet_name = result["strategy"].replace(":", "").replace("-", "").replace("%", "pct")[:31]
            sheet_name = f"S{i+1}_{sheet_name[:28]}"
            result["portfolio_history"].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved: {excel_file}")

    # Save latest copy
    latest_file = RESULTS_DIR / "trailing_stop_comparison_latest.xlsx"
    with pd.ExcelWriter(latest_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for i, result in enumerate(results):
            sheet_name = result["strategy"].replace(":", "").replace("-", "").replace("%", "pct")[:31]
            sheet_name = f"S{i+1}_{sheet_name[:28]}"
            result["portfolio_history"].to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved: {latest_file}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    return results, summary_df


if __name__ == "__main__":
    main()
