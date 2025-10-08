"""
Compare Momentum Backtest Results

Analyzes multiple backtest runs to identify which parameters work best.
Helps iterate toward "run with winners, sell losers" that works out-of-sample.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
from datetime import datetime


def load_all_tests():
    """Load all momentum backtest results."""
    results_dir = project_root / "results" / "momentum_backtests"

    if not results_dir.exists():
        print("No test results found in results/momentum_backtests/")
        return []

    tests = []
    for test_dir in sorted(results_dir.glob("test_*")):
        results_file = test_dir / "results.json"
        params_file = test_dir / "params.json"

        if results_file.exists() and params_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            with open(params_file) as f:
                params = json.load(f)

            tests.append({
                'test_dir': test_dir.name,
                'params': params,
                'results': results
            })

    return tests


def compare_tests(tests):
    """Generate comparison table."""
    if not tests:
        print("No tests to compare")
        return

    print("\n" + "="*120)
    print("MOMENTUM BACKTEST COMPARISON")
    print("="*120 + "\n")

    # Create comparison DataFrame
    rows = []
    for test in tests:
        params = test['params']
        perf = test['results']['performance']
        turn = test['results']['turnover']
        risk = test['results']['risk_management']

        rows.append({
            'Test': test['test_dir'],
            'Mom_Period': params['momentum_period'],
            'Turnover_Penalty': params['turnover_penalty'],
            'Max_Pos': params['max_positions'],
            'Rel_Strength': params['rel_strength_enabled'],
            'Trend_Filter': params['trend_filter_enabled'],
            'Stop_Loss': params['enable_stop_loss'],
            'CAGR': perf['cagr'] * 100,
            'Sharpe': perf['sharpe'],
            'MaxDD': perf['max_drawdown'] * 100,
            'Turnover': turn['avg_turnover_pct'],
            'ETF_Changes': turn['etf_changes_per_rebalance'],
            'Stops': risk['stop_loss_triggers'],
            'Final_Value': perf['final_value']
        })

    df = pd.DataFrame(rows)

    # Sort by Sharpe
    df = df.sort_values('Sharpe', ascending=False)

    print(df.to_string(index=False))

    print("\n" + "="*120)
    print("KEY INSIGHTS")
    print("="*120 + "\n")

    # Best Sharpe
    best_sharpe = df.iloc[0]
    print(f"🏆 Best Sharpe Ratio: {best_sharpe['Sharpe']:.2f}")
    print(f"   Test: {best_sharpe['Test']}")
    print(f"   Parameters: Mom={best_sharpe['Mom_Period']}, Turnover_Penalty={best_sharpe['Turnover_Penalty']}, Max_Pos={best_sharpe['Max_Pos']}")
    print(f"   CAGR: {best_sharpe['CAGR']:.1f}%, Turnover: {best_sharpe['Turnover']:.0f}%, ETF Changes: {best_sharpe['ETF_Changes']:.1f}\n")

    # Lowest turnover with positive Sharpe
    positive_sharpe = df[df['Sharpe'] > 0]
    if not positive_sharpe.empty:
        best_efficiency = positive_sharpe.sort_values('Turnover').iloc[0]
        print(f"⚡ Most Efficient (Low Turnover + Positive Sharpe):")
        print(f"   Sharpe: {best_efficiency['Sharpe']:.2f}, CAGR: {best_efficiency['CAGR']:.1f}%")
        print(f"   Turnover: {best_efficiency['Turnover']:.0f}%, ETF Changes: {best_efficiency['ETF_Changes']:.1f}")
        print(f"   Parameters: Mom={best_efficiency['Mom_Period']}, Turnover_Penalty={best_efficiency['Turnover_Penalty']}\n")

    # Parameter importance
    print("\n📊 PARAMETER IMPACT ANALYSIS\n")

    # Group by momentum period
    if len(df['Mom_Period'].unique()) > 1:
        print("Momentum Period Impact:")
        by_momentum = df.groupby('Mom_Period').agg({
            'Sharpe': 'mean',
            'CAGR': 'mean',
            'Turnover': 'mean'
        }).round(2)
        print(by_momentum)
        print()

    # Group by turnover penalty
    if len(df['Turnover_Penalty'].unique()) > 1:
        print("Turnover Penalty Impact:")
        by_turnover = df.groupby('Turnover_Penalty').agg({
            'Sharpe': 'mean',
            'CAGR': 'mean',
            'Turnover': 'mean',
            'ETF_Changes': 'mean'
        }).round(2)
        print(by_turnover)
        print()

    # Relative strength vs not
    if len(df['Rel_Strength'].unique()) > 1:
        print("Relative Strength Impact:")
        by_relstr = df.groupby('Rel_Strength').agg({
            'Sharpe': 'mean',
            'CAGR': 'mean'
        }).round(2)
        print(by_relstr)
        print()

    print("\n" + "="*120)
    print("NEXT STEPS")
    print("="*120 + "\n")

    if df['Sharpe'].max() < 0:
        print("⚠️  ALL TESTS HAVE NEGATIVE SHARPE")
        print("\nThe signals are not working. Recommendations:")
        print("  1. Try even simpler signals (pure 12-month momentum only)")
        print("  2. Increase momentum period (252 days = 1 year)")
        print("  3. Try sector rotation instead of individual ETFs")
        print("  4. Consider the market period may be unfavorable for momentum")
    elif df['Sharpe'].max() < 0.5:
        print("⚠️  WEAK PERFORMANCE (Sharpe < 0.5)")
        print("\nThe strategy is marginally profitable. Try:")
        print("  1. Test different momentum periods")
        print("  2. Adjust turnover penalty")
        print("  3. Enable/disable relative strength and trend filter")
    else:
        print("✅ POSITIVE RESULTS FOUND")
        print(f"\nBest Sharpe: {df['Sharpe'].max():.2f}")
        print("\nOptimize further:")
        print("  1. Fine-tune turnover penalty around winning value")
        print("  2. Test slightly different momentum periods")
        print("  3. Run longer backtest period to validate")

    print("\nTo run more tests:")
    print("  python scripts/backtest_weekly_momentum.py --momentum-period 252")
    print("  python scripts/backtest_weekly_momentum.py --turnover-penalty 100.0")
    print("  python scripts/backtest_weekly_momentum.py --no-rel-strength")
    print("\n" + "="*120 + "\n")


def main():
    """Load and compare all tests."""
    tests = load_all_tests()

    if not tests:
        print("\nNo tests found. Run some backtests first:")
        print("  python scripts/backtest_weekly_momentum.py")
        print("  python scripts/backtest_weekly_momentum.py --momentum-period 252")
        print("  python scripts/backtest_weekly_momentum.py --turnover-penalty 100.0")
        return

    print(f"\nFound {len(tests)} backtest runs")
    compare_tests(tests)


if __name__ == "__main__":
    main()
