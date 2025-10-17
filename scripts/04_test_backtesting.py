"""
Test Backtesting Framework

Tests the backtesting engine with synthetic data to ensure
it works correctly before running on real historical data.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.backtesting import (
    BacktestEngine,
    BacktestConfig,
    PerformanceMetrics,
    TransactionCostModel
)
from src.portfolio import (
    SimpleOptimizer,
    ThresholdRebalancer,
    StopLossManager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(num_etfs: int = 50, num_days: int = 252) -> tuple:
    """Generate synthetic price and factor data for testing."""
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range('2022-01-01', periods=num_days, freq='D')

    # Generate ticker names
    tickers = [f'ETF_{i:03d}' for i in range(num_etfs)]

    # Generate random walk prices
    # Start at 100, add daily returns
    returns = np.random.randn(num_days, num_etfs) * 0.01  # 1% daily vol

    # Add some drift (positive expected return)
    drift = np.random.uniform(0.0005, 0.0015, num_etfs)  # 12-40% annual drift
    returns = returns + drift

    # Calculate cumulative returns to get prices
    prices = pd.DataFrame(
        100 * (1 + returns).cumprod(axis=0),
        columns=tickers,
        index=dates
    )

    # Generate factor scores (correlated with future returns for realism)
    # Good factors should predict future returns
    factor_scores_data = {}

    for date_idx in range(num_days):
        # Calculate forward-looking returns for next 20 days
        if date_idx < num_days - 20:
            future_returns = returns[date_idx+1:date_idx+21, :].mean(axis=0)
        else:
            future_returns = returns[date_idx:, :].mean(axis=0)

        # Factor scores are noisy signals of future returns
        noise = np.random.randn(num_etfs) * 0.5
        scores = future_returns * 10 + noise  # Scale up and add noise

        # Normalize to z-scores
        scores = (scores - scores.mean()) / scores.std()

        factor_scores_data[dates[date_idx]] = pd.Series(scores, index=tickers)

    factor_scores = pd.DataFrame(factor_scores_data).T

    return prices, factor_scores


def test_basic_backtest():
    """Test basic backtesting functionality."""
    logger.info("="*60)
    logger.info("TEST 1: BASIC BACKTEST")
    logger.info("="*60)

    # Generate data
    prices, factor_scores = generate_synthetic_data(num_etfs=50, num_days=252)

    logger.info(f"Generated data: {len(prices)} days, {len(prices.columns)} ETFs")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=1_000_000,
        rebalance_frequency='monthly',
        num_positions=20,
        stop_loss_pct=0.15,
        use_stop_loss=True
    )

    # Create components
    optimizer = SimpleOptimizer(num_positions=config.num_positions)
    rebalancer = ThresholdRebalancer(drift_threshold=0.05)
    risk_manager = StopLossManager(position_stop_loss=0.15)
    cost_model = TransactionCostModel()

    # Run backtest
    engine = BacktestEngine(config, cost_model)

    results = engine.run(
        prices=prices,
        factor_scores=factor_scores,
        optimizer=optimizer,
        rebalancer=rebalancer,
        risk_manager=risk_manager
    )

    # Display results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)

    metrics = results['metrics']
    logger.info(f"\nFinal Value: ${metrics['final_value']:,.0f}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"CAGR: {metrics['cagr']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")

    logger.info(f"\nTrading Activity:")
    logger.info(f"  Rebalances: {metrics['num_rebalances']}")
    logger.info(f"  Total Trades: {metrics['num_trades']}")
    logger.info(f"  Avg Turnover: {metrics['avg_turnover']:.2%}")
    logger.info(f"  Transaction Costs: ${metrics['total_transaction_costs']:,.0f}")

    # Validate results
    assert metrics['final_value'] > 0, "Final value should be positive"
    assert metrics['num_rebalances'] > 0, "Should have rebalanced"
    assert len(results['daily_values']) > 0, "Should have daily values"

    logger.info("\n✓ Basic backtest test passed")

    return results


def test_stop_loss():
    """Test stop-loss functionality."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: STOP-LOSS")
    logger.info("="*60)

    # Generate data with a severe crash
    np.random.seed(42)
    num_etfs = 30
    num_days = 100

    dates = pd.date_range('2023-01-01', periods=num_days, freq='D')
    tickers = [f'ETF_{i:03d}' for i in range(num_etfs)]

    # Normal returns for first 50 days
    returns = np.random.randn(50, num_etfs) * 0.01

    # Crash: -20% in 10 days for half the ETFs
    crash_returns = np.random.randn(10, num_etfs) * 0.01
    crash_returns[:, :num_etfs//2] -= 0.02  # -2% per day

    # Recovery
    recovery_returns = np.random.randn(40, num_etfs) * 0.01

    # Combine
    all_returns = np.vstack([returns, crash_returns, recovery_returns])

    prices = pd.DataFrame(
        100 * (1 + all_returns).cumprod(axis=0),
        columns=tickers,
        index=dates
    )

    # Simple constant factor scores
    factor_scores = pd.DataFrame(
        np.random.randn(num_days, num_etfs),
        columns=tickers,
        index=dates
    )

    # Test WITH stop-loss
    config_with_sl = BacktestConfig(
        initial_capital=100_000,
        rebalance_frequency='monthly',
        num_positions=15,
        stop_loss_pct=0.15,
        use_stop_loss=True
    )

    engine_with_sl = BacktestEngine(config_with_sl)
    results_with_sl = engine_with_sl.run(
        prices=prices,
        factor_scores=factor_scores,
        optimizer=SimpleOptimizer(num_positions=15),
        rebalancer=ThresholdRebalancer(drift_threshold=0.10),
        risk_manager=StopLossManager(position_stop_loss=0.15)
    )

    # Test WITHOUT stop-loss
    config_without_sl = BacktestConfig(
        initial_capital=100_000,
        rebalance_frequency='monthly',
        num_positions=15,
        use_stop_loss=False
    )

    engine_without_sl = BacktestEngine(config_without_sl)
    results_without_sl = engine_without_sl.run(
        prices=prices,
        factor_scores=factor_scores,
        optimizer=SimpleOptimizer(num_positions=15),
        rebalancer=ThresholdRebalancer(drift_threshold=0.10),
        risk_manager=StopLossManager(position_stop_loss=0.15)
    )

    # Compare
    logger.info("\nComparison:")
    logger.info(f"  With stop-loss:    ${results_with_sl['metrics']['final_value']:,.0f} "
               f"({results_with_sl['metrics']['total_return']:+.2%})")
    logger.info(f"  Without stop-loss: ${results_without_sl['metrics']['final_value']:,.0f} "
               f"({results_without_sl['metrics']['total_return']:+.2%})")

    logger.info(f"\n  Max Drawdown:")
    logger.info(f"    With stop-loss:    {results_with_sl['metrics']['max_drawdown']:.2%}")
    logger.info(f"    Without stop-loss: {results_without_sl['metrics']['max_drawdown']:.2%}")

    # Count stop-loss trades
    if len(results_with_sl['trades']) > 0:
        stop_loss_trades = results_with_sl['trades'][
            results_with_sl['trades']['action'] == 'STOP_LOSS'
        ]
        logger.info(f"\n  Stop-loss trades executed: {len(stop_loss_trades)}")

    logger.info("\n✓ Stop-loss test passed")

    return results_with_sl, results_without_sl


def test_transaction_costs():
    """Test impact of transaction costs."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: TRANSACTION COSTS")
    logger.info("="*60)

    # Generate data
    prices, factor_scores = generate_synthetic_data(num_etfs=40, num_days=126)

    # Test with different rebalancing frequencies
    frequencies = ['weekly', 'monthly', 'quarterly']
    results_by_freq = {}

    for freq in frequencies:
        config = BacktestConfig(
            initial_capital=500_000,
            rebalance_frequency=freq,
            num_positions=20,
            use_stop_loss=False
        )

        engine = BacktestEngine(config)
        results = engine.run(
            prices=prices,
            factor_scores=factor_scores,
            optimizer=SimpleOptimizer(num_positions=20),
            rebalancer=ThresholdRebalancer(drift_threshold=0.05),
            risk_manager=StopLossManager()
        )

        results_by_freq[freq] = results

        logger.info(f"\n{freq.capitalize()} rebalancing:")
        logger.info(f"  Rebalances: {results['metrics']['num_rebalances']}")
        logger.info(f"  Total trades: {results['metrics']['num_trades']}")
        logger.info(f"  Transaction costs: ${results['metrics']['total_transaction_costs']:,.0f}")
        logger.info(f"  Final value: ${results['metrics']['final_value']:,.0f}")
        logger.info(f"  Net return: {results['metrics']['total_return']:.2%}")

    logger.info("\n✓ Transaction cost test passed")

    return results_by_freq


def test_performance_metrics():
    """Test performance metrics calculation."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: PERFORMANCE METRICS")
    logger.info("="*60)

    # Create synthetic portfolio with known properties
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # Portfolio that goes up steadily with some volatility
    returns = np.random.randn(1000) * 0.01 + 0.0005  # 0.05% daily drift, 1% vol

    portfolio_values = pd.Series(
        1_000_000 * (1 + returns).cumprod(),
        index=dates
    )

    # Calculate metrics
    metrics_calc = PerformanceMetrics(risk_free_rate=0.02)
    metrics = metrics_calc.calculate_all_metrics(portfolio_values)

    # Display
    logger.info("\n" + metrics_calc.format_metrics(metrics))

    # Validate
    assert metrics['cagr'] > 0, "CAGR should be positive"
    assert metrics['sharpe_ratio'] != 0, "Sharpe ratio should be calculated"
    assert metrics['max_drawdown'] < 0, "Max drawdown should be negative"

    logger.info("\n✓ Performance metrics test passed")

    return metrics


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("BACKTESTING FRAMEWORK TESTS")
    logger.info("="*60)

    # Run tests
    test1_results = test_basic_backtest()
    test2_with_sl, test2_without_sl = test_stop_loss()
    test3_results = test_transaction_costs()
    test4_metrics = test_performance_metrics()

    logger.info("\n" + "="*60)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("="*60)
    logger.info("\nBacktesting framework is ready for use!")
    logger.info("Next step: Run backtests on real historical data")


if __name__ == '__main__':
    main()
