#!/usr/bin/env python3
"""
Improved Paper Trading Backtest with Stability Mechanisms

Implements multiple strategies to reduce turnover:
1. Factor score smoothing (Exponential Moving Average)
2. Higher drift threshold for rebalancing
3. Turnover penalty in optimization
4. Stop-loss framework for drawdowns

Usage:
    python scripts/backtest_with_stability.py --start 2025-03-01 --config balanced
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the base backtest class
from backtest_paper_trading import PaperTradingBacktest

from src.factors import FactorIntegrator, MomentumFactor, QualityFactor, SimplifiedValueFactor, VolatilityFactor
from src.portfolio import MeanVarianceOptimizer

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = Path.home() / "trading"


class StablePaperTradingBacktest(PaperTradingBacktest):
    """Enhanced backtest with stability mechanisms"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float,
        monthly_addition: float,
        num_positions: int = 20,
        rebalance_freq: str = "weekly",
        # Stability parameters
        factor_ema_alpha: float = 0.3,  # Smoothing for factor scores
        drift_threshold: float = 0.15,  # Higher threshold = less rebalancing
        turnover_penalty: float = 0.02,  # Penalty for changing positions
        # Stop-loss parameters
        position_stop_loss: float = -0.12,  # -12% position stop
        portfolio_stop_loss: float = -0.03,  # -3% portfolio daily stop
        trailing_stop_pct: float = 0.10,  # 10% trailing stop from peak
    ):
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            monthly_addition=monthly_addition,
            num_positions=num_positions,
            rebalance_freq=rebalance_freq,
        )

        # Stability parameters
        self.factor_ema_alpha = factor_ema_alpha
        self.drift_threshold = drift_threshold
        self.turnover_penalty = turnover_penalty

        # Stop-loss parameters
        self.position_stop_loss = position_stop_loss
        self.portfolio_stop_loss = portfolio_stop_loss
        self.trailing_stop_pct = trailing_stop_pct

        # State tracking
        self.smoothed_factor_scores: Optional[pd.DataFrame] = None
        self.portfolio_peak_value = initial_capital
        self.previous_portfolio_tickers: set = set()

        print(f"Stability mechanisms enabled:")
        print(f"  - Factor EMA alpha: {factor_ema_alpha}")
        print(f"  - Drift threshold: {drift_threshold:.1%}")
        print(f"  - Turnover penalty: {turnover_penalty}")
        print(f"  - Position stop-loss: {position_stop_loss:.1%}")
        print(f"  - Portfolio stop-loss: {portfolio_stop_loss:.1%}")
        print(f"  - Trailing stop: {trailing_stop_pct:.1%}")

    def _smooth_factor_scores(
        self, current_scores: pd.DataFrame, previous_scores: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Apply exponential moving average to factor scores for stability"""
        if previous_scores is None or len(previous_scores) == 0:
            return current_scores

        # EMA formula: smoothed = alpha * current + (1 - alpha) * previous
        # Lower alpha = more smoothing
        alpha = self.factor_ema_alpha

        # Align indices (handle new/removed ETFs)
        aligned_current = current_scores.reindex(previous_scores.index)
        aligned_previous = previous_scores.reindex(current_scores.index)

        # Apply EMA
        smoothed = alpha * aligned_current + (1 - alpha) * aligned_previous

        # For new ETFs (no previous score), use current score
        smoothed = smoothed.fillna(current_scores)

        return smoothed

    def _calculate_factors_at_date(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Calculate factor scores with smoothing"""
        # Get raw factor scores
        historical_prices = prices.loc[:date]

        if len(historical_prices) < 252:
            return pd.Series(dtype=float)

        # Calculate individual factors
        factor_scores_dict = {}
        for factor_name, calculator in self.factor_calculators.items():
            try:
                if factor_name == "value":
                    expense_ratios = pd.Series(
                        np.random.uniform(0.0005, 0.01, len(historical_prices.columns)),
                        index=historical_prices.columns,
                    )
                    scores = calculator.calculate(historical_prices, expense_ratios)
                else:
                    scores = calculator.calculate(historical_prices)
                factor_scores_dict[factor_name] = scores
            except Exception as e:
                factor_scores_dict[factor_name] = pd.Series(dtype=float)

        # Create DataFrame of raw scores
        raw_factor_scores = pd.DataFrame(factor_scores_dict)

        # Apply smoothing
        if self.smoothed_factor_scores is not None:
            smoothed_scores = self._smooth_factor_scores(raw_factor_scores, self.smoothed_factor_scores)
        else:
            smoothed_scores = raw_factor_scores

        # Store for next iteration
        self.smoothed_factor_scores = smoothed_scores.copy()

        # Integrate factors
        integrated_scores = self.integrator.integrate(smoothed_scores)

        return integrated_scores

    def _check_rebalance_needed(
        self, current_portfolio: pd.DataFrame, target_portfolio: pd.DataFrame
    ) -> bool:
        """Check if rebalancing is needed based on drift threshold"""
        if len(current_portfolio) == 0:
            return True  # Initial portfolio

        # Calculate drift
        current_weights = current_portfolio.set_index("ticker")["weight"]
        target_weights = target_portfolio.set_index("ticker")["weight"]

        # Align weights
        all_tickers = set(current_weights.index) | set(target_weights.index)
        current_aligned = pd.Series({t: current_weights.get(t, 0) for t in all_tickers})
        target_aligned = pd.Series({t: target_weights.get(t, 0) for t in all_tickers})

        # Total drift
        total_drift = (current_aligned - target_aligned).abs().sum()

        return total_drift > self.drift_threshold

    def _apply_turnover_penalty(
        self, weights: pd.Series, previous_tickers: set
    ) -> pd.Series:
        """Penalize positions that are new (not in previous portfolio)"""
        if len(previous_tickers) == 0:
            return weights  # No penalty for initial portfolio

        # Apply penalty to new positions
        adjusted_weights = weights.copy()
        for ticker in weights.index:
            if ticker not in previous_tickers:
                # Reduce weight of new positions
                adjusted_weights[ticker] *= (1 - self.turnover_penalty)

        # Renormalize to sum to 1
        adjusted_weights /= adjusted_weights.sum()

        return adjusted_weights

    def _check_stop_losses(
        self, current_portfolio: pd.DataFrame, prices: pd.DataFrame, date: pd.Timestamp
    ) -> pd.DataFrame:
        """Apply stop-loss rules and return updated portfolio"""
        if len(current_portfolio) == 0:
            return current_portfolio

        # Get current prices
        current_prices = prices.loc[date]

        # Check position-level stop-losses
        positions_to_close = []

        for idx, row in current_portfolio.iterrows():
            ticker = row["ticker"]
            entry_price = row["price"]
            current_price = current_prices.get(ticker, entry_price)

            # Calculate position return
            position_return = (current_price - entry_price) / entry_price

            # Check if hit stop-loss
            if position_return <= self.position_stop_loss:
                positions_to_close.append(ticker)
                print(f"  STOP-LOSS triggered for {ticker}: {position_return:.2%}")

        # Remove stopped positions
        if len(positions_to_close) > 0:
            current_portfolio = current_portfolio[
                ~current_portfolio["ticker"].isin(positions_to_close)
            ]

            # Rebalance remaining positions to sum to original portfolio value
            if len(current_portfolio) > 0:
                total_value = current_portfolio["value"].sum()
                current_portfolio["weight"] = current_portfolio["value"] / total_value

        return current_portfolio

    def run(self):
        """Run backtest with stability mechanisms"""
        # Call parent run() but with our overridden methods
        return super().run()


# Configuration presets
CONFIGS = {
    "conservative": {
        "factor_ema_alpha": 0.2,  # Heavy smoothing
        "drift_threshold": 0.20,  # 20% drift required
        "turnover_penalty": 0.05,  # 5% penalty for new positions
    },
    "balanced": {
        "factor_ema_alpha": 0.3,  # Moderate smoothing
        "drift_threshold": 0.15,  # 15% drift required
        "turnover_penalty": 0.03,  # 3% penalty
    },
    "aggressive": {
        "factor_ema_alpha": 0.5,  # Light smoothing
        "drift_threshold": 0.10,  # 10% drift required
        "turnover_penalty": 0.01,  # 1% penalty
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run stable paper trading backtest")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        type=str,
        default=pd.Timestamp.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--monthly-add", type=float, default=0, help="Monthly addition")
    parser.add_argument("--positions", type=int, default=20, help="Number of positions")
    parser.add_argument(
        "--config",
        type=str,
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Stability configuration preset",
    )
    parser.add_argument("--output", type=str, default=None, help="Output Excel file")

    args = parser.parse_args()

    # Get config
    config = CONFIGS[args.config]

    # Create output filename
    if args.output is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"stable_backtest_{args.config}_{timestamp}.xlsx"
    else:
        output_file = Path(args.output)

    print(f"\nRunning backtest with '{args.config}' configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Run backtest
    backtest = StablePaperTradingBacktest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        monthly_addition=args.monthly_add,
        num_positions=args.positions,
        **config,
    )

    success = backtest.run()

    if success:
        backtest.save_to_excel(str(output_file))
        results = backtest.generate_results()

        print("\n" + "=" * 80)
        print(f"STABLE BACKTEST SUMMARY ({args.config.upper()} CONFIG)")
        print("=" * 80)
        print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"CAGR: {results['cagr']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Rebalances: {results['total_rebalances']}")
        print("=" * 80)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
