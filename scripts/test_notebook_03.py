"""
Test script to validate notebook 03 works correctly.

This runs the key notebook cells to ensure they execute without errors.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

print("="*80)
print("TESTING NOTEBOOK 03: Backtesting Results")
print("="*80)

# Cell 3: Load data
print("\n[Cell 3] Loading data and calculating factors...")
from src.backtesting import BacktestEngine, BacktestConfig, PerformanceMetrics
from src.portfolio import SimpleOptimizer, MinVarianceOptimizer, MeanVarianceOptimizer
from src.portfolio import ThresholdRebalancer, StopLossManager
from src.factors import (
    MomentumFactor, QualityFactor, SimplifiedValueFactor,
    VolatilityFactor, FactorIntegrator
)

# Generate synthetic data
np.random.seed(42)
tickers = [f'ETF{i:03d}' for i in range(100)]
dates = pd.date_range('2021-01-01', '2024-10-01', freq='D')

prices = pd.DataFrame(
    100 * np.exp(np.random.randn(len(dates), 100).cumsum(axis=0) * 0.01),
    index=dates,
    columns=tickers
)

print(f"✓ Data shape: {prices.shape}")
print(f"✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Cell 4: Calculate factors
print("\n[Cell 4] Calculating factors...")
momentum = MomentumFactor(lookback=252, skip_recent=21)
quality = QualityFactor(lookback=252)
value = SimplifiedValueFactor()
volatility = VolatilityFactor(lookback=60)

expense_ratios = pd.Series(
    np.random.uniform(0.0005, 0.01, len(tickers)),
    index=tickers
)

momentum_scores = momentum.calculate(prices)
quality_scores = quality.calculate(prices)
value_scores = value.calculate(prices, expense_ratios)
volatility_scores = volatility.calculate(prices)

factor_df = pd.DataFrame({
    'momentum': momentum_scores,
    'quality': quality_scores,
    'value': value_scores,
    'volatility': volatility_scores
})

integrator = FactorIntegrator(factor_weights={
    'momentum': 0.25,
    'quality': 0.25,
    'value': 0.25,
    'volatility': 0.25
})

combined_scores = integrator.integrate(factor_df)

# For backtesting, we need factor scores over time (dates x tickers)
# In production, these would be calculated with a rolling window
# For this demo, we'll replicate the current scores across all dates
factor_scores_ts = pd.DataFrame(
    np.tile(combined_scores.values, (len(prices), 1)),
    index=prices.index,
    columns=combined_scores.index
)

print(f"✓ Factor scores calculated for {len(combined_scores)} ETFs")
print(f"✓ Factor scores time series shape: {factor_scores_ts.shape}")

# Cell 6: Run backtests with corrected interface
print("\n[Cell 6] Running backtests...")

def run_backtest_with_optimizer(optimizer, prices, factor_scores_ts, name):
    """Run backtest with given optimizer."""
    print(f"  Running: {name}...")

    config = BacktestConfig(
        initial_capital=1_000_000,
        start_date=prices.index[0],
        end_date=prices.index[-1],
        rebalance_frequency='weekly',
        num_positions=20,
        stop_loss_pct=0.12,
        use_stop_loss=True
    )

    rebalancer = ThresholdRebalancer(drift_threshold=0.05)
    risk_manager = StopLossManager(position_stop_loss=config.stop_loss_pct)

    # FIXED: Create engine without optimizer/rebalancer/risk_manager
    engine = BacktestEngine(config=config)

    # FIXED: Pass them to run() method instead
    results = engine.run(prices, factor_scores_ts, optimizer, rebalancer, risk_manager)
    return results

results = {}

try:
    results['Simple'] = run_backtest_with_optimizer(
        SimpleOptimizer(num_positions=20),
        prices, factor_scores_ts, 'Simple Equal-Weight'
    )
    print("  ✓ Simple optimizer completed")

    results['MinVar'] = run_backtest_with_optimizer(
        MinVarianceOptimizer(num_positions=20, lookback=60, risk_penalty=0.01),
        prices, factor_scores_ts, 'Minimum Variance'
    )
    print("  ✓ MinVar optimizer completed")

    results['MVO'] = run_backtest_with_optimizer(
        MeanVarianceOptimizer(num_positions=20, lookback=60, risk_aversion=1.0, axioma_penalty=0.01),
        prices, factor_scores_ts, 'Mean-Variance (Axioma)'
    )
    print("  ✓ MVO optimizer completed")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All backtests complete")

# Cell 8: Extract results
print("\n[Cell 8] Extracting performance metrics...")
portfolio_values = {}
for name, result in results.items():
    portfolio_values[name] = result['portfolio_values']

perf_df = pd.DataFrame(portfolio_values)

metrics_calc = PerformanceMetrics()
summary = {}

for name in results.keys():
    returns = perf_df[name].pct_change().dropna()
    metrics = metrics_calc.calculate_all_metrics(perf_df[name], returns)
    summary[name] = metrics

summary_df = pd.DataFrame(summary).T

print("\nPerformance Summary:")
print("="*80)
print(summary_df[['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 'volatility']].to_string())
print("="*80)

# Verify results
print("\n[Verification]")
for name in results.keys():
    final_val = perf_df[name].iloc[-1]
    initial_val = perf_df[name].iloc[0]
    total_return = (final_val / initial_val - 1) * 100
    print(f"  {name}: ${final_val:,.0f} ({total_return:+.2f}%)")

print("\n" + "="*80)
print("✓ NOTEBOOK TEST PASSED - All cells executed successfully")
print("="*80)
print("\nThe notebook is ready to use. Key fixes:")
print("1. BacktestEngine() - no optimizer/rebalancer/risk_manager in __init__")
print("2. engine.run(prices, scores, optimizer, rebalancer, risk_manager)")
print("="*80)
