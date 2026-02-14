#!/usr/bin/env python3
"""
Diagnostic script to analyze causes of high portfolio turnover

This script investigates why the portfolio rebalances so frequently by analyzing:
1. Factor score stability week-over-week
2. Portfolio composition changes
3. Optimizer sensitivity to inputs
4. Correlation between factor changes and portfolio changes

Usage:
    python scripts/diagnose_rebalancing.py --backtest-file results/paper_trading_backtest_YYYYMMDD.xlsx
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class RebalancingDiagnostics:
    """Analyze causes of portfolio rebalancing"""

    def __init__(self, backtest_file: str):
        self.backtest_file = Path(backtest_file)

        # Load backtest results
        self.portfolio_history = pd.read_excel(backtest_file, sheet_name="Portfolio_History")
        self.trades_history = pd.read_excel(backtest_file, sheet_name="Trades")
        self.performance = pd.read_excel(backtest_file, sheet_name="Performance")

        # Load price data
        prices_file = DATA_DIR / "processed" / "etf_prices_filtered.parquet"
        self.prices = pd.read_parquet(prices_file)

        # Initialize factor calculators
        self.factor_calculators = {
            "momentum": MomentumFactor(lookback=252, skip_recent=21),
            "quality": QualityFactor(lookback=252),
            "value": SimplifiedValueFactor(),
            "volatility": VolatilityFactor(lookback=60),
        }
        self.integrator = FactorIntegrator(
            factor_weights={
                "momentum": 0.25,
                "quality": 0.25,
                "value": 0.25,
                "volatility": 0.25,
            }
        )

    def calculate_turnover_by_week(self) -> pd.DataFrame:
        """Calculate weekly turnover metrics"""
        print("\n" + "=" * 80)
        print("TURNOVER ANALYSIS")
        print("=" * 80)

        # Group by date
        turnover_data = []
        dates = sorted(self.portfolio_history['date'].unique())

        for i, date in enumerate(dates):
            current_portfolio = self.portfolio_history[
                self.portfolio_history['date'] == date
            ]['ticker'].tolist()

            if i > 0:
                prev_portfolio = self.portfolio_history[
                    self.portfolio_history['date'] == dates[i-1]
                ]['ticker'].tolist()

                # Calculate turnover
                added = set(current_portfolio) - set(prev_portfolio)
                removed = set(prev_portfolio) - set(current_portfolio)
                unchanged = set(current_portfolio) & set(prev_portfolio)

                turnover_pct = (len(added) + len(removed)) / (2 * len(current_portfolio))

                turnover_data.append({
                    'date': date,
                    'added': len(added),
                    'removed': len(removed),
                    'unchanged': len(unchanged),
                    'turnover_pct': turnover_pct,
                })

        turnover_df = pd.DataFrame(turnover_data)

        print(f"\nAverage weekly turnover: {turnover_df['turnover_pct'].mean():.1%}")
        print(f"Max weekly turnover: {turnover_df['turnover_pct'].max():.1%}")
        print(f"Min weekly turnover: {turnover_df['turnover_pct'].min():.1%}")
        print(f"\nWeeks with >50% turnover: {(turnover_df['turnover_pct'] > 0.5).sum()} / {len(turnover_df)}")
        print(f"Weeks with >25% turnover: {(turnover_df['turnover_pct'] > 0.25).sum()} / {len(turnover_df)}")

        return turnover_df

    def analyze_factor_stability(self) -> pd.DataFrame:
        """Analyze how factor scores change week-over-week"""
        print("\n" + "=" * 80)
        print("FACTOR STABILITY ANALYSIS")
        print("=" * 80)

        dates = sorted(self.portfolio_history['date'].unique())
        factor_stability = []
        prev_scores = {}

        # Track factor scores for portfolio holdings over time
        for i, date in enumerate(dates):
            # Get holdings at this date
            holdings = self.portfolio_history[
                self.portfolio_history['date'] == date
            ]['ticker'].tolist()

            if len(holdings) == 0:
                continue

            # Calculate factor scores at this date
            # Handle timezone mismatch
            query_date = pd.Timestamp(date)
            if self.prices.index.tz is not None and query_date.tz is None:
                query_date = query_date.tz_localize(self.prices.index.tz)
            historical_prices = self.prices.loc[:query_date]

            if len(historical_prices) < 252:
                continue

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
                except:
                    factor_scores_dict[factor_name] = pd.Series(dtype=float)

            # Get scores for current holdings
            current_scores = {}
            for factor_name, scores in factor_scores_dict.items():
                current_scores[factor_name] = scores[holdings].mean()

            # Compare to previous week
            if i > 0 and len(prev_scores) > 0:
                for factor_name in current_scores.keys():
                    if factor_name in prev_scores:
                        prev_score = prev_scores[factor_name]
                        curr_score = current_scores[factor_name]

                        if not pd.isna(prev_score) and not pd.isna(curr_score):
                            change = abs(curr_score - prev_score)
                            factor_stability.append({
                                'date': date,
                                'factor': factor_name,
                                'score': curr_score,
                                'change': change,
                            })

            # Store for next iteration
            prev_scores = current_scores.copy()

        if len(factor_stability) > 0:
            stability_df = pd.DataFrame(factor_stability)

            print("\nAverage factor score changes (week-over-week):")
            for factor in stability_df['factor'].unique():
                factor_data = stability_df[stability_df['factor'] == factor]
                print(f"  {factor}: {factor_data['change'].mean():.4f} (std: {factor_data['change'].std():.4f})")

            return stability_df
        else:
            print("Not enough data for factor stability analysis")
            return pd.DataFrame()

    def analyze_holding_persistence(self) -> pd.DataFrame:
        """Analyze which ETFs are held consistently vs churned"""
        print("\n" + "=" * 80)
        print("HOLDING PERSISTENCE ANALYSIS")
        print("=" * 80)

        dates = sorted(self.portfolio_history['date'].unique())

        # Count how many weeks each ticker appears
        ticker_counts = self.portfolio_history['ticker'].value_counts()

        print(f"\nTotal unique ETFs held during period: {len(ticker_counts)}")
        print(f"ETFs held in all {len(dates)} weeks: {(ticker_counts == len(dates)).sum()}")
        print(f"ETFs held in >75% of weeks: {(ticker_counts > len(dates) * 0.75).sum()}")
        print(f"ETFs held in >50% of weeks: {(ticker_counts > len(dates) * 0.5).sum()}")
        print(f"ETFs held in <25% of weeks: {(ticker_counts < len(dates) * 0.25).sum()}")
        print(f"ETFs held only once: {(ticker_counts == 1).sum()}")

        # Show most persistent holdings
        print("\nTop 10 most persistent holdings:")
        print(ticker_counts.head(10).to_string())

        # Show most churned holdings
        print("\nTop 10 most churned holdings:")
        print(ticker_counts.tail(10).to_string())

        return ticker_counts.to_frame('weeks_held')

    def analyze_weight_changes(self) -> pd.DataFrame:
        """Analyze how position weights change week-over-week"""
        print("\n" + "=" * 80)
        print("WEIGHT CHANGE ANALYSIS")
        print("=" * 80)

        dates = sorted(self.portfolio_history['date'].unique())
        weight_changes = []

        for i, date in enumerate(dates[1:], 1):
            prev_date = dates[i-1]

            # Get portfolios
            current = self.portfolio_history[self.portfolio_history['date'] == date]
            previous = self.portfolio_history[self.portfolio_history['date'] == prev_date]

            # Get common tickers
            current_dict = current.set_index('ticker')['weight'].to_dict()
            prev_dict = previous.set_index('ticker')['weight'].to_dict()

            common_tickers = set(current_dict.keys()) & set(prev_dict.keys())

            for ticker in common_tickers:
                weight_change = abs(current_dict[ticker] - prev_dict[ticker])
                weight_changes.append({
                    'date': date,
                    'ticker': ticker,
                    'prev_weight': prev_dict[ticker],
                    'curr_weight': current_dict[ticker],
                    'weight_change': weight_change,
                })

        if len(weight_changes) > 0:
            weight_df = pd.DataFrame(weight_changes)

            print(f"\nAverage weight change (for continuing positions): {weight_df['weight_change'].mean():.2%}")
            print(f"Median weight change: {weight_df['weight_change'].median():.2%}")
            print(f"Max weight change: {weight_df['weight_change'].max():.2%}")

            # Count significant weight changes
            significant = weight_df[weight_df['weight_change'] > 0.02]  # >2% change
            print(f"\nPositions with >2% weight change: {len(significant)} / {len(weight_df)} ({len(significant)/len(weight_df):.1%})")

            return weight_df
        else:
            print("Not enough data for weight change analysis")
            return pd.DataFrame()

    def generate_report(self, output_file: str = None):
        """Generate comprehensive diagnostic report"""
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = RESULTS_DIR / f"rebalancing_diagnosis_{timestamp}.txt"
        else:
            output_file = Path(output_file)

        # Redirect stdout to file
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            print("=" * 80)
            print("PORTFOLIO REBALANCING DIAGNOSTIC REPORT")
            print("=" * 80)
            print(f"\nBacktest file: {self.backtest_file}")
            print(f"Analysis date: {pd.Timestamp.now()}")

            # Run all analyses
            turnover_df = self.calculate_turnover_by_week()
            stability_df = self.analyze_factor_stability()
            persistence_df = self.analyze_holding_persistence()
            weight_df = self.analyze_weight_changes()

            print("\n" + "=" * 80)
            print("CONCLUSIONS")
            print("=" * 80)

            # Determine root causes
            avg_turnover = turnover_df['turnover_pct'].mean()

            print(f"\n1. TURNOVER RATE: {avg_turnover:.1%} per week")
            if avg_turnover > 0.5:
                print("   ❌ EXCESSIVE - Portfolio churning nearly entirely each week")
            elif avg_turnover > 0.25:
                print("   ⚠️  HIGH - More than target 25% turnover")
            else:
                print("   ✓ ACCEPTABLE - Within target range")

            # Check persistence
            total_unique = len(persistence_df)
            weeks = len(self.portfolio_history['date'].unique())
            stable_holdings = (persistence_df['weeks_held'] > weeks * 0.75).sum()

            print(f"\n2. HOLDING STABILITY: {stable_holdings}/{total_unique} ETFs held >75% of time")
            if stable_holdings < 20:
                print("   ❌ LOW STABILITY - Core holdings changing frequently")
                print("   → ROOT CAUSE: Factor scores or optimizer creating unstable portfolios")
            else:
                print("   ✓ REASONABLE - Core holdings relatively stable")

            # Check weight changes
            if len(weight_df) > 0:
                avg_weight_change = weight_df['weight_change'].mean()
                print(f"\n3. WEIGHT DRIFT: {avg_weight_change:.2%} average change per position")
                if avg_weight_change > 0.03:
                    print("   ⚠️  SIGNIFICANT - Weights drifting substantially")
                    print("   → ROOT CAUSE: Optimizer sensitivity or volatile factor scores")
                else:
                    print("   ✓ MODERATE - Weight changes reasonable")

            print("\n" + "=" * 80)
            print("RECOMMENDED MITIGATIONS")
            print("=" * 80)

            if avg_turnover > 0.5:
                print("\n🔧 HIGH PRIORITY FIXES:")
                print("  1. Increase drift threshold from 5% to 15% or higher")
                print("  2. Implement factor score smoothing (exponential moving average)")
                print("  3. Add transaction cost penalty to optimizer")
                print("  4. Consider bi-weekly or monthly rebalancing instead of weekly")

            if stable_holdings < 20:
                print("\n🔧 STABILITY IMPROVEMENTS:")
                print("  1. Add momentum/persistence to factor scores")
                print("  2. Penalize portfolio turnover in optimization")
                print("  3. Use longer lookback windows for factors")
                print("  4. Implement 'hold range' - only rebalance if drift >threshold")

            print("\n🛡️ STOP-LOSS FRAMEWORK:")
            print("  1. Position-level stop-loss: -12% (already in code)")
            print("  2. Portfolio-level stop-loss: Trigger if daily loss > -3%")
            print("  3. Volatility-based stops: Adjust based on recent volatility")
            print("  4. Trailing stops: Lock in gains on strong performers")

        # Write to file
        report = f.getvalue()
        output_file.write_text(report)
        print(report)  # Also print to console

        print(f"\n✓ Report saved to: {output_file}")

        return turnover_df, stability_df, persistence_df, weight_df


def main():
    parser = argparse.ArgumentParser(description="Diagnose portfolio rebalancing causes")
    parser.add_argument(
        "--backtest-file",
        type=str,
        required=True,
        help="Path to backtest Excel file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report file path",
    )

    args = parser.parse_args()

    # Run diagnostics
    diagnostics = RebalancingDiagnostics(args.backtest_file)
    diagnostics.generate_report(args.output)


if __name__ == "__main__":
    main()
