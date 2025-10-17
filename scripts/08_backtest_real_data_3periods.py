"""
Real Data Backtest - 3 Time Periods

Run backtests on real ETF data across 3 distinct market periods:
1. 2020-2021: COVID recovery + bull market
2. 2022-2023: Inflation/rates + volatility
3. 2024-2025: Current period

Tests all 4 optimizers: Simple, Rank, MinVar, MVO
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.factors import (
    MomentumFactor,
    QualityFactor,
    SimplifiedValueFactor,
    VolatilityFactor,
    FactorIntegrator
)
from src.portfolio import (
    SimpleOptimizer,
    RankBasedOptimizer,
    MinVarianceOptimizer,
    MeanVarianceOptimizer,
    ThresholdRebalancer,
    StopLossManager
)
from src.backtesting import BacktestEngine, BacktestConfig, PerformanceMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest_real_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load filtered price data."""
    logger.info("Loading filtered price data...")

    prices_file = project_root / 'data' / 'processed' / 'etf_prices_filtered.parquet'
    prices = pd.read_parquet(prices_file)

    logger.info(f"✓ Loaded {len(prices.columns)} ETFs")
    logger.info(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    logger.info(f"  Days: {len(prices)}")

    return prices


def calculate_factors(prices):
    """Calculate all factors on real data."""
    logger.info("\nCalculating factors...")

    # Initialize factors
    momentum = MomentumFactor(lookback=252, skip_recent=21)
    quality = QualityFactor(lookback=252)
    value = SimplifiedValueFactor()
    volatility = VolatilityFactor(lookback=60)

    # Generate expense ratios (use random for now, replace with real later)
    np.random.seed(42)
    expense_ratios = pd.Series(
        np.random.uniform(0.0005, 0.01, len(prices.columns)),
        index=prices.columns
    )

    # Calculate individual factors
    logger.info("  - Momentum...")
    momentum_scores = momentum.calculate(prices)

    logger.info("  - Quality...")
    quality_scores = quality.calculate(prices)

    logger.info("  - Value...")
    value_scores = value.calculate(prices, expense_ratios)

    logger.info("  - Volatility...")
    volatility_scores = volatility.calculate(prices)

    # Combine into DataFrame
    factor_df = pd.DataFrame({
        'momentum': momentum_scores,
        'quality': quality_scores,
        'value': value_scores,
        'volatility': volatility_scores
    })

    # Integrate using geometric mean
    integrator = FactorIntegrator(factor_weights={
        'momentum': 0.25,
        'quality': 0.25,
        'value': 0.25,
        'volatility': 0.25
    })

    combined_scores = integrator.integrate(factor_df)

    # Create time series (replicate for now - rolling calc would be better)
    factor_scores_ts = pd.DataFrame(
        np.tile(combined_scores.values, (len(prices), 1)),
        index=prices.index,
        columns=combined_scores.index
    )

    logger.info(f"✓ Factor scores calculated")

    return factor_scores_ts, combined_scores


def run_backtest(optimizer_name, optimizer, prices, factor_scores_ts, start_date, end_date, period_label):
    """Run backtest for one optimizer on one period."""
    logger.info(f"\n  {optimizer_name} optimizer...")

    # Filter data to period
    period_prices = prices.loc[start_date:end_date]
    period_scores = factor_scores_ts.loc[start_date:end_date]

    if len(period_prices) < 100:
        logger.warning(f"    Insufficient data for period: {len(period_prices)} days")
        return None

    # Configure backtest
    config = BacktestConfig(
        initial_capital=1_000_000,
        start_date=period_prices.index[0],
        end_date=period_prices.index[-1],
        rebalance_frequency='weekly',
        num_positions=20,
        stop_loss_pct=0.12,
        use_stop_loss=True
    )

    rebalancer = ThresholdRebalancer(drift_threshold=0.05)
    risk_manager = StopLossManager(position_stop_loss=config.stop_loss_pct)

    # Run backtest
    engine = BacktestEngine(config=config)

    try:
        results = engine.run(period_prices, period_scores, optimizer, rebalancer, risk_manager)

        # Extract key metrics
        metrics = results['metrics']

        logger.info(f"    Total Return: {metrics['total_return']:.2%}")
        logger.info(f"    CAGR: {metrics['cagr']:.2%}")
        logger.info(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"    Volatility: {metrics['volatility']:.2%}")
        logger.info(f"    Rebalances: {metrics['num_rebalances']}")

        return {
            'optimizer': optimizer_name,
            'period': period_label,
            'start_date': start_date,
            'end_date': end_date,
            'days': len(period_prices),
            **metrics
        }

    except Exception as e:
        logger.error(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_backtests(prices, factor_scores_ts):
    """Run backtests for all optimizers across all periods."""
    logger.info("\n" + "="*80)
    logger.info("RUNNING BACKTESTS")
    logger.info("="*80)

    # Define optimizers
    optimizers = {
        'Simple': SimpleOptimizer(num_positions=20),
        'RankBased': RankBasedOptimizer(num_positions=20),
        'MinVar': MinVarianceOptimizer(num_positions=20, lookback=60, risk_penalty=0.01),
        'MVO': MeanVarianceOptimizer(num_positions=20, lookback=60, risk_aversion=1.0, axioma_penalty=0.01)
    }

    # Define periods (adjusted for available data: Oct 2020 - Oct 2025)
    periods = [
        ('2020-10-05', '2021-12-31', 'Period 1: 2020-2021 (COVID Recovery)'),
        ('2022-01-01', '2023-12-31', 'Period 2: 2022-2023 (Inflation/Rates)'),
        ('2024-01-01', '2025-10-03', 'Period 3: 2024-2025 (Current)')
    ]

    # Run all combinations
    results_list = []

    for start, end, label in periods:
        logger.info(f"\n{'='*80}")
        logger.info(f"{label}")
        logger.info(f"  {start} to {end}")
        logger.info(f"{'='*80}")

        for opt_name, optimizer in optimizers.items():
            result = run_backtest(opt_name, optimizer, prices, factor_scores_ts, start, end, label)
            if result:
                results_list.append(result)

    # Combine results
    results_df = pd.DataFrame(results_list)

    return results_df


def analyze_results(results_df):
    """Analyze backtest results across periods."""
    logger.info("\n" + "="*80)
    logger.info("RESULTS ANALYSIS")
    logger.info("="*80)

    # Summary by optimizer
    logger.info("\n1. Performance by Optimizer (Average Across All Periods)")
    logger.info("-"*80)

    summary = results_df.groupby('optimizer').agg({
        'total_return': 'mean',
        'cagr': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'volatility': 'mean',
        'num_rebalances': 'sum'
    })

    print(summary.to_string())

    # Summary by period
    logger.info("\n\n2. Performance by Period (Average Across All Optimizers)")
    logger.info("-"*80)

    period_summary = results_df.groupby('period').agg({
        'total_return': 'mean',
        'cagr': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'volatility': 'mean'
    })

    print(period_summary.to_string())

    # Best performer by metric
    logger.info("\n\n3. Best Performer by Metric")
    logger.info("-"*80)

    metrics = ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown']

    for metric in metrics:
        if metric == 'max_drawdown':
            # For drawdown, "best" is least negative
            best_idx = results_df[metric].idxmax()
        else:
            best_idx = results_df[metric].idxmax()

        best = results_df.loc[best_idx]
        logger.info(f"\n{metric.upper().replace('_', ' ')}:")
        logger.info(f"  Winner: {best['optimizer']} in {best['period']}")
        logger.info(f"  Value: {best[metric]:.2%}" if 'ratio' not in metric else f"  Value: {best[metric]:.2f}")

    # Check if targets met
    logger.info("\n\n4. AQR Performance Targets")
    logger.info("-"*80)

    targets = {
        'cagr': (0.12, 'CAGR > 12%'),
        'sharpe_ratio': (0.8, 'Sharpe > 0.8'),
        'max_drawdown': (-0.25, 'Max DD < 25%')
    }

    for metric, (threshold, description) in targets.items():
        if metric == 'max_drawdown':
            passed = (results_df[metric] > threshold).sum()
        else:
            passed = (results_df[metric] > threshold).sum()

        total = len(results_df)
        pct = passed / total * 100

        status = "✓ PASS" if pct >= 66.7 else "✗ FAIL"
        logger.info(f"{description}: {passed}/{total} ({pct:.0f}%) {status}")


def save_results(results_df):
    """Save results to files."""
    output_dir = project_root / 'results' / 'real_data_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results
    results_file = output_dir / f'backtest_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"\n✓ Saved detailed results: {results_file}")

    # Save latest copy
    latest_file = output_dir / 'backtest_results_latest.csv'
    results_df.to_csv(latest_file, index=False)
    logger.info(f"✓ Saved latest results: {latest_file}")

    return results_file


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("REAL DATA BACKTEST - 3 PERIODS")
    logger.info("="*80)

    # Load data
    prices = load_data()

    # Calculate factors
    factor_scores_ts, combined_scores = calculate_factors(prices)

    # Run all backtests
    results_df = run_all_backtests(prices, factor_scores_ts)

    # Analyze results
    analyze_results(results_df)

    # Save results
    results_file = save_results(results_df)

    logger.info("\n" + "="*80)
    logger.info("✓ BACKTEST COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
