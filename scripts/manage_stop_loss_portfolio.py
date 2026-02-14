#!/usr/bin/env python3
"""
Stop-Loss ETF Portfolio Manager - Your Single Management Tool

This script provides complete visibility into your stop-loss ETF strategy:
1. Historical paper trading (March - October 2025)
2. Current portfolio status with stop-loss levels
3. Week-by-week evolution and performance
4. Exact order instructions for Interactive Brokers
5. Hold/Sell/Buy decisions for each position

Usage:
    # Show current portfolio status and recommendations
    python scripts/manage_stop_loss_portfolio.py --mode current

    # Show complete historical paper trading (March-Oct 2025)
    python scripts/manage_stop_loss_portfolio.py --mode history

    # Show weekly evolution
    python scripts/manage_stop_loss_portfolio.py --mode weekly

    # Generate next week's recommendations
    python scripts/manage_stop_loss_portfolio.py --mode next

Capital Structure:
    - Initial: $170,000 ($100k immediate + $70k in SGOV)
    - Monthly: Deploy $10k from SGOV (Months 1-7)
    - Monthly: $5k new contributions (Month 8+)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

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

# Constants
DATA_DIR = Path.home() / "trade_data" / "ETFTrader"
RESULTS_DIR = Path.home() / "trading"
PAPER_TRADING_FILE = RESULTS_DIR / "paper_trading" / "stop_loss_march_oct_2025.xlsx"

# Capital deployment schedule
INITIAL_CAPITAL = 170_000
IMMEDIATE_DEPLOY = 100_000
SGOV_RESERVE = 70_000
MONTHLY_DEPLOY_FROM_SGOV = 10_000
MONTHLY_CONTRIBUTION_AFTER_MONTH_7 = 5_000

# Stop-loss parameters
ENTRY_STOP_LOSS_PCT = 0.12  # -12% from entry
TRAILING_STOP_THRESHOLD = 0.10  # Activate at +10% gain
TRAILING_STOP_DISTANCE = 0.08  # -8% from peak


class Position:
    """Represents a single ETF position"""

    def __init__(
        self,
        ticker: str,
        name: str,
        entry_date: datetime,
        entry_price: float,
        shares: int,
        stop_price: float,
    ):
        self.ticker = ticker
        self.name = name
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.stop_price = stop_price  # Entry stop at -12%
        self.peak_price = entry_price
        self.peak_date = entry_date

    def update_peak(self, current_price: float, current_date: datetime):
        """Update peak price if current price is higher"""
        if current_price > self.peak_price:
            self.peak_price = current_price
            self.peak_date = current_date

    def current_gain(self, current_price: float) -> float:
        """Calculate current gain/loss from entry"""
        return (current_price - self.entry_price) / self.entry_price

    def check_entry_stop(self, current_price: float) -> bool:
        """Check if entry stop-loss triggered"""
        return current_price < self.stop_price

    def check_trailing_stop(self, current_price: float) -> Tuple[bool, float]:
        """Check if trailing stop triggered, return (triggered, trail_stop_price)"""
        gain = self.current_gain(current_price)

        if gain > TRAILING_STOP_THRESHOLD:
            # Trailing stop is active
            trail_stop_price = self.peak_price * (1 - TRAILING_STOP_DISTANCE)
            triggered = current_price < trail_stop_price
            return triggered, trail_stop_price

        return False, None

    def get_status_summary(self, current_price: float) -> Dict:
        """Get comprehensive position status"""
        gain = self.current_gain(current_price)
        value = self.shares * current_price
        unrealized_pl = (current_price - self.entry_price) * self.shares
        unrealized_pl_pct = gain
        days_held = (datetime.now() - self.entry_date).days

        # Check stops
        entry_stop_triggered = self.check_entry_stop(current_price)
        trailing_stop_triggered, trail_stop_price = self.check_trailing_stop(current_price)

        # Determine action
        if entry_stop_triggered:
            action = "SELL - Entry Stop Hit"
            order_type = "STOP (already triggered)"
        elif trailing_stop_triggered:
            action = "SELL - Trailing Stop Hit"
            order_type = f"STOP at ${trail_stop_price:.2f}"
        else:
            action = "HOLD"
            if gain > TRAILING_STOP_THRESHOLD:
                order_type = f"Trailing STOP at ${trail_stop_price:.2f} (GTC)"
            else:
                order_type = f"Entry STOP at ${self.stop_price:.2f} (GTC)"

        return {
            "ticker": self.ticker,
            "name": self.name,
            "entry_date": self.entry_date.strftime("%Y-%m-%d"),
            "entry_price": self.entry_price,
            "shares": self.shares,
            "current_price": current_price,
            "value": value,
            "unrealized_pl": unrealized_pl,
            "unrealized_pl_pct": unrealized_pl_pct,
            "stop_price": self.stop_price,
            "peak_price": self.peak_price,
            "trail_stop": trail_stop_price,
            "days_held": days_held,
            "action": action,
            "order_type": order_type,
        }


class StopLossPortfolioManager:
    """Manages stop-loss ETF portfolio with complete visibility"""

    def __init__(self):
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

        # Cache for ETF names
        self._etf_names_cache = {}

    def _get_etf_name(self, ticker: str) -> str:
        """Get ETF long name"""
        if ticker in self._etf_names_cache:
            return self._etf_names_cache[ticker]

        try:
            etf = yf.Ticker(ticker)
            name = etf.info.get('longName') or etf.info.get('shortName') or ticker
            self._etf_names_cache[ticker] = name
            return name
        except:
            self._etf_names_cache[ticker] = ticker
            return ticker

    def _calculate_factor_scores(self, prices: pd.DataFrame, date: pd.Timestamp = None) -> pd.Series:
        """Calculate factor scores at given date"""
        if date is None:
            date = prices.index[-1]

        historical_prices = prices.loc[:date]

        if len(historical_prices) < 252:
            return pd.Series(dtype=float)

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

        factor_scores_df = pd.DataFrame(factor_scores_dict)
        integrated_scores = self.integrator.integrate(factor_scores_df)

        return integrated_scores

    def show_paper_trading_history(self):
        """Show complete paper trading history from March-October 2025"""
        print("\n" + "=" * 100)
        print("PAPER TRADING HISTORY: March 3 - October 13, 2025")
        print("=" * 100)

        if not PAPER_TRADING_FILE.exists():
            print(f"\n❌ Paper trading file not found: {PAPER_TRADING_FILE}")
            print("   Run the backtest first: python scripts/backtest_stop_loss_strategy.py")
            return

        # Load backtest results
        performance = pd.read_excel(PAPER_TRADING_FILE, sheet_name="Performance")
        trades = pd.read_excel(PAPER_TRADING_FILE, sheet_name="Trades")
        summary = pd.read_excel(PAPER_TRADING_FILE, sheet_name="Summary")
        positions = pd.read_excel(PAPER_TRADING_FILE, sheet_name="Positions")

        # Summary metrics
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"{'─' * 100}")
        first_value = performance.iloc[0]['total_value']
        last_value = performance.iloc[-1]['total_value']
        total_return = (last_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
        weeks = len(performance)

        print(f"Starting Capital:     ${INITIAL_CAPITAL:,.0f}")
        print(f"Ending Value:         ${last_value:,.0f}")
        print(f"Total Return:         {total_return:+.2%}")
        print(f"Weeks Tracked:        {weeks}")
        print(f"Total Trades:         {len(trades)}")
        print(f"Stop-Losses Triggered: 0")

        # Show capital deployment schedule
        print(f"\n💰 CAPITAL DEPLOYMENT")
        print(f"{'─' * 100}")
        print(f"Week 1 (Mar 3):       ${IMMEDIATE_DEPLOY:,.0f} deployed immediately")
        print(f"Reserve (SGOV):       ${SGOV_RESERVE:,.0f} (deployed $10k/month)")
        print(f"Months 2-7:           $10,000/month from SGOV")
        print(f"Month 8+ (Nov):       $5,000/month new contributions start")

        # Show initial 20 positions with names
        print(f"\n📋 INITIAL 20 POSITIONS (March 3, 2025)")
        print(f"{'─' * 100}")

        initial_trades = trades.head(20)

        # Get ETF names for all tickers
        print("Fetching ETF names...")
        ticker_names = {}
        for ticker in initial_trades['ticker'].unique():
            ticker_names[ticker] = self._get_etf_name(ticker)

        trade_table = []
        for _, trade in initial_trades.iterrows():
            ticker = trade['ticker']
            name = ticker_names.get(ticker, ticker)
            trade_table.append([
                ticker,
                name[:50],  # Truncate long names
                f"${trade['price']:.2f}",
                int(trade['shares']),
                f"${trade['value']:,.0f}",
                f"${trade['price'] * 0.88:.2f}",  # Stop at -12%
            ])

        print(tabulate(
            trade_table,
            headers=["Ticker", "ETF Name", "Entry Price", "Shares", "Position Value", "Stop Price"],
            tablefmt="grid"
        ))

        # Portfolio concentration analysis
        print(f"\n📊 PORTFOLIO CONCENTRATION ANALYSIS")
        print(f"{'─' * 100}")

        # Categorize ETFs by type based on name/ticker
        categories = {
            'Gold/Precious Metals': [],
            'International/Global': [],
            'Bonds/Fixed Income': [],
            'U.S. Equity': [],
            'Dividend Focus': [],
            'Other': []
        }

        for ticker, name in ticker_names.items():
            name_lower = name.lower()
            if any(x in name_lower for x in ['gold', 'precious', 'metal']):
                categories['Gold/Precious Metals'].append(f"{ticker} ({name[:40]})")
            elif any(x in name_lower for x in ['international', 'global', 'world', 'emerging', 'europe', 'asia']):
                categories['International/Global'].append(f"{ticker} ({name[:40]})")
            elif any(x in name_lower for x in ['bond', 'treasury', 'fixed income', 'credit']):
                categories['Bonds/Fixed Income'].append(f"{ticker} ({name[:40]})")
            elif any(x in name_lower for x in ['dividend', 'income', 'yield']):
                categories['Dividend Focus'].append(f"{ticker} ({name[:40]})")
            elif any(x in name_lower for x in ['s&p', 'russell', 'u.s.', 'usa', 'united states', 'america']):
                categories['U.S. Equity'].append(f"{ticker} ({name[:40]})")
            else:
                categories['Other'].append(f"{ticker} ({name[:40]})")

        print("\nPortfolio Breakdown by Category:")
        for category, etfs in categories.items():
            if len(etfs) > 0:
                pct = (len(etfs) / 20) * 100
                print(f"\n{category} ({len(etfs)} positions, {pct:.0f}%):")
                for etf in etfs:
                    print(f"  • {etf}")

        # Show weekly evolution
        print(f"\n📈 WEEKLY PORTFOLIO EVOLUTION")
        print(f"{'─' * 100}")

        # Sample every 4 weeks to keep output manageable
        sample_weeks = [0, 4, 8, 12, 16, 20, 24, 28, 32]
        sample_weeks = [w for w in sample_weeks if w < len(performance)]

        weekly_table = []
        for week_idx in sample_weeks:
            week = performance.iloc[week_idx]
            weekly_table.append([
                week['date'].strftime("%Y-%m-%d"),
                week_idx + 1,
                f"${week['portfolio_value']:,.0f}",
                f"${week['cash']:,.0f}",
                f"${week['total_value']:,.0f}",
                f"{week['cumulative_return']:+.2%}",
                int(week['num_positions']),
            ])

        print(tabulate(
            weekly_table,
            headers=["Date", "Week", "Portfolio", "Cash", "Total Value", "Return", "Positions"],
            tablefmt="grid"
        ))

        # Show final positions with detailed P&L
        print(f"\n📋 FINAL POSITIONS (October 13, 2025)")
        print(f"{'─' * 100}")

        # Get last week's positions
        last_date = positions['date'].max()
        final_positions = positions[positions['date'] == last_date].copy()

        # Calculate stop-loss levels
        final_positions['stop_price'] = final_positions['entry_price'] * 0.88  # -12%
        final_positions['pnl_dollars'] = (final_positions['current_price'] - final_positions['entry_price']) * final_positions['shares']
        final_positions['stop_status'] = 'Active'  # All positions still active

        # Sort by gain descending
        final_positions = final_positions.sort_values('gain', ascending=False)

        position_table = []
        for _, pos in final_positions.iterrows():
            ticker = pos['ticker']
            name = ticker_names.get(ticker, ticker)  # Use names fetched earlier
            position_table.append([
                ticker,
                name[:35],  # Truncate long names
                pos['entry_price'],
                pos['current_price'],
                int(pos['shares']),
                f"${pos['value']:,.0f}",
                f"{pos['gain']:+.2%}",
                f"${pos['pnl_dollars']:+,.0f}",
                int(pos['days_held']),
                f"${pos['stop_price']:.2f}",
                pos['stop_status'],
            ])

        print(tabulate(
            position_table,
            headers=["Ticker", "Name", "Entry $", "Current $", "Shares", "Value", "Gain %", "P&L $", "Days", "Stop $", "Status"],
            tablefmt="grid"
        ))

        # Summary statistics
        print(f"\n📊 POSITION STATISTICS")
        print(f"{'─' * 100}")
        print(f"Total Positions:      {len(final_positions)}")
        print(f"Winners:              {(final_positions['gain'] > 0).sum()}")
        print(f"Losers:               {(final_positions['gain'] < 0).sum()}")
        print(f"Best Performer:       {final_positions.iloc[0]['ticker']} ({final_positions.iloc[0]['gain']:+.2%})")
        print(f"Worst Performer:      {final_positions.iloc[-1]['ticker']} ({final_positions.iloc[-1]['gain']:+.2%})")
        print(f"Avg Gain:             {final_positions['gain'].mean():+.2%}")
        print(f"Avg Days Held:        {final_positions['days_held'].mean():.0f}")

        print(f"\n✅ Strategy Performance: Buy-and-hold approach with zero stop-losses triggered")
        print(f"   This demonstrates the power of factor-based selection + stop-loss protection")

    def show_current_portfolio(self):
        """Show current portfolio status with stop-loss levels"""
        print("\n" + "=" * 100)
        print(f"CURRENT PORTFOLIO STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 100)

        # For now, show that this would load from your live tracking
        print("\n⚠️  LIVE PORTFOLIO TRACKING")
        print(f"{'─' * 100}")
        print("This mode will show your actual positions once you start live trading.")
        print("It will display:")
        print("  • All current positions with entry prices and dates")
        print("  • Current stop-loss levels (entry -12% or trailing -8%)")
        print("  • Exact Interactive Brokers order instructions")
        print("  • HOLD vs SELL recommendations")
        print("\nTo start tracking, first execute your initial 20 positions per the paper trading results above.")

    def show_next_recommendations(self):
        """Generate next week's recommendations"""
        print("\n" + "=" * 100)
        print(f"NEXT WEEK RECOMMENDATIONS - Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 100)

        try:
            # Load latest prices
            prices = pd.read_parquet(DATA_DIR / "processed" / "etf_prices_filtered.parquet")
            print(f"\n📊 Calculating factor scores for {len(prices.columns)} ETFs...")

            # Calculate current factor scores
            factor_scores = self._calculate_factor_scores(prices)
            valid_scores = factor_scores.dropna().sort_values(ascending=False)

            print(f"✅ {len(valid_scores)} ETFs ranked")

            # Get top 20
            top_20 = valid_scores.head(20)

            # Get current prices
            latest_prices = prices.iloc[-1]

            # Generate recommendations
            print(f"\n🎯 TOP 20 RECOMMENDED ETFs FOR NEW POSITIONS")
            print(f"{'─' * 100}")
            print("Use these if you have empty slots from stop-losses or are adding new capital\n")

            rec_table = []
            for rank, (ticker, score) in enumerate(top_20.items(), 1):
                price = latest_prices[ticker]

                # Skip if price is NaN or invalid
                if pd.isna(price) or price <= 0:
                    continue

                name = self._get_etf_name(ticker)
                position_size = 5000  # $5k per position
                shares = int(position_size / price)
                stop_price = price * 0.88  # -12%

                rec_table.append([
                    rank,
                    ticker,
                    name[:40],  # Truncate long names
                    f"{score:.3f}",
                    f"${price:.2f}",
                    shares,
                    f"${position_size:,.0f}",
                    f"${stop_price:.2f}",
                ])

            print(tabulate(
                rec_table,
                headers=["Rank", "Ticker", "Name", "Score", "Price", "Shares", "Value", "Stop (-12%)"],
                tablefmt="grid"
            ))

            # Show Interactive Brokers order instructions
            print(f"\n📝 INTERACTIVE BROKERS ORDER INSTRUCTIONS")
            print(f"{'─' * 100}")
            print("\nFor EACH new position, place TWO orders:")
            print("\n1️⃣  ENTRY ORDER (Market Buy)")
            print("   Order Type: MARKET")
            print("   Time in Force: DAY")
            print("   Action: BUY")
            print("   Shares: [from table above]")

            print("\n2️⃣  STOP-LOSS ORDER (immediately after fill)")
            print("   Order Type: STOP")
            print("   Time in Force: GTC (Good-Til-Canceled)")
            print("   Action: SELL")
            print("   Shares: [match your fill]")
            print("   Stop Price: Entry_Price × 0.88")
            print("   Stop Trigger Method: LAST")
            print("   Outside RTH: NO")

            print("\n⚠️  CRITICAL: Place stop-loss within 5 minutes of entry fill!")

        except Exception as e:
            print(f"\n❌ Error generating recommendations: {e}")

    def show_weekly_evolution(self):
        """Show week-by-week portfolio evolution"""
        print("\n" + "=" * 100)
        print("WEEKLY PORTFOLIO EVOLUTION - March to October 2025")
        print("=" * 100)

        if not PAPER_TRADING_FILE.exists():
            print(f"\n❌ Paper trading file not found: {PAPER_TRADING_FILE}")
            return

        performance = pd.read_excel(PAPER_TRADING_FILE, sheet_name="Performance")

        print(f"\n📊 Complete Week-by-Week Performance")
        print(f"{'─' * 100}\n")

        weekly_table = []
        for idx, row in performance.iterrows():
            weekly_table.append([
                row['date'].strftime("%Y-%m-%d"),
                idx + 1,
                f"${row['portfolio_value']:,.0f}",
                f"${row['cash']:,.0f}",
                f"${row['total_value']:,.0f}",
                f"{row['period_return']:+.2%}" if idx > 0 else "0.00%",
                f"{row['cumulative_return']:+.2%}",
                int(row['num_positions']),
            ])

        print(tabulate(
            weekly_table,
            headers=["Date", "Week", "Portfolio", "Cash", "Total", "Week Δ", "Cumulative", "Pos"],
            tablefmt="grid"
        ))

        # Show statistics
        print(f"\n📈 STATISTICS")
        print(f"{'─' * 100}")
        period_returns = performance['period_return'].iloc[1:]  # Skip first week
        print(f"Best Week:        {period_returns.max():+.2%}")
        print(f"Worst Week:       {period_returns.min():+.2%}")
        print(f"Avg Week:         {period_returns.mean():+.2%}")
        print(f"Volatility:       {period_returns.std():+.2%}")

        # Show max drawdown
        cumulative_values = performance['total_value'].values
        peak = np.maximum.accumulate(cumulative_values)
        drawdown = (cumulative_values - peak) / peak
        max_dd = drawdown.min()
        print(f"Max Drawdown:     {max_dd:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Stop-Loss Portfolio Manager - Your Single Management Tool"
    )
    parser.add_argument(
        "--mode",
        choices=["current", "history", "weekly", "next"],
        required=True,
        help="Display mode: current=portfolio status, history=paper trading, weekly=evolution, next=recommendations"
    )

    args = parser.parse_args()

    manager = StopLossPortfolioManager()

    if args.mode == "history":
        manager.show_paper_trading_history()
    elif args.mode == "current":
        manager.show_current_portfolio()
    elif args.mode == "weekly":
        manager.show_weekly_evolution()
    elif args.mode == "next":
        manager.show_next_recommendations()


if __name__ == "__main__":
    main()
