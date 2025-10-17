"""
Backtest: $1M Standard Portfolio

Standard portfolio backtest with $1,000,000 initial capital.
No additional contributions.

Metrics:
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown
- CAGR
- Win Rate
- Turnover
- Transaction Costs

Compares strategy against SPY benchmark.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
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
from src.factors import (
    MomentumFactor,
    QualityFactor,
    SimplifiedValueFactor,
    VolatilityFactor,
    FactorIntegrator
)

# Setup logging
log_file = project_root / "logs" / f"backtest_1m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load factor scores and price data."""
    data_dir = project_root / "data"

    # Load factor scores
    factor_file = data_dir / "factor_scores_latest.parquet"
    if not factor_file.exists():
        raise FileNotFoundError(
            f"Factor scores not found at {factor_file}. "
            "Run scripts/03_test_portfolio_construction.py first to generate data."
        )

    factor_scores = pd.read_parquet(factor_file)

    logger.info(f"Loaded factor scores for {len(factor_scores)} ETFs")

    # For demo purposes, generate synthetic price data
    # In production, this would load real historical prices
    logger.info("Generating synthetic price data for backtesting...")

    np.random.seed(42)
    num_days = 756  # ~3 years of daily data
    dates = pd.date_range('2022-01-01', periods=num_days, freq='D')

    tickers = factor_scores.index.tolist()

    # Generate random walk prices with drift
    # Better ETFs (higher factor scores) get better returns
    returns_data = []

    for ticker in tickers:
        score = factor_scores.loc[ticker, 'integrated']

        # Drift based on factor score (normalized)
        # Good factors should predict returns
        base_drift = 0.0005  # 12% annual
        factor_alpha = score * 0.0002  # Factor contribution
        total_drift = base_drift + factor_alpha

        # Generate returns
        returns = np.random.randn(num_days) * 0.01 + total_drift
        returns_data.append(returns)

    returns_array = np.array(returns_data).T

    prices = pd.DataFrame(
        100 * (1 + returns_array).cumprod(axis=0),
        columns=tickers,
        index=dates
    )

    logger.info(f"Generated price data: {len(dates)} days, {len(tickers)} ETFs")

    # Expand factor scores to match price dates
    # In reality, would recalculate factors each day
    # For demo, we'll replicate the scores
    expanded_scores = pd.DataFrame(
        [factor_scores['integrated'].values] * len(dates),
        columns=tickers,
        index=dates
    )

    return prices, expanded_scores


def run_backtest(prices: pd.DataFrame,
                factor_scores: pd.DataFrame,
                config: BacktestConfig) -> dict:
    """
    Run backtest with given configuration.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical prices
    factor_scores : pd.DataFrame
        Factor scores over time
    config : BacktestConfig
        Backtest configuration

    Returns
    -------
    dict
        Backtest results
    """
    logger.info("="*60)
    logger.info(f"BACKTEST CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Initial Capital: ${config.initial_capital:,.0f}")
    logger.info(f"Rebalance Frequency: {config.rebalance_frequency}")
    logger.info(f"Number of Positions: {config.num_positions}")
    logger.info(f"Stop Loss: {config.stop_loss_pct:.1%}" if config.use_stop_loss else "No stop-loss")

    # Create components
    optimizer = SimpleOptimizer(num_positions=config.num_positions)

    rebalancer = ThresholdRebalancer(
        drift_threshold=0.05,  # 5% drift threshold
        min_trade_size=0.01
    )

    risk_manager = StopLossManager(
        position_stop_loss=config.stop_loss_pct,
        portfolio_stop_loss=0.20
    )

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

    return results


def calculate_benchmark(prices: pd.DataFrame, initial_capital: float) -> pd.Series:
    """
    Calculate buy-and-hold SPY benchmark.

    For demo, use equal-weight portfolio of all ETFs.
    In production, would use actual SPY data.
    """
    # Equal weight all ETFs
    avg_returns = prices.pct_change().mean(axis=1)

    benchmark_values = initial_capital * (1 + avg_returns).cumprod()
    benchmark_values.iloc[0] = initial_capital

    return benchmark_values


def display_results(results: dict, benchmark: pd.Series):
    """Display comprehensive backtest results."""
    metrics = results['metrics']

    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS - $1M STANDARD PORTFOLIO")
    logger.info("="*60)

    # Portfolio Performance
    logger.info("\n" + "="*60)
    logger.info("PORTFOLIO PERFORMANCE")
    logger.info("="*60)
    logger.info(f"Initial Value:       ${results['config'].initial_capital:>15,.0f}")
    logger.info(f"Final Value:         ${metrics['final_value']:>15,.0f}")
    logger.info(f"Total Return:        {metrics['total_return']:>15.2%}")
    logger.info(f"CAGR:                {metrics['cagr']:>15.2%}")

    # Risk-Adjusted Metrics
    logger.info("\n" + "="*60)
    logger.info("RISK-ADJUSTED METRICS")
    logger.info("="*60)
    logger.info(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>15.2f}")
    logger.info(f"Sortino Ratio:       {metrics['sortino_ratio']:>15.2f}")
    logger.info(f"Calmar Ratio:        {metrics['calmar_ratio']:>15.2f}")
    logger.info(f"Volatility:          {metrics['volatility']:>15.2%}")

    # Drawdown Analysis
    logger.info("\n" + "="*60)
    logger.info("DRAWDOWN ANALYSIS")
    logger.info("="*60)
    logger.info(f"Max Drawdown:        {metrics['max_drawdown']:>15.2%}")
    logger.info(f"Max DD Duration:     {metrics['max_drawdown_duration']:>15.0f} days")
    logger.info(f"Current Drawdown:    {metrics['current_drawdown']:>15.2%}")

    # Win/Loss Statistics
    logger.info("\n" + "="*60)
    logger.info("WIN/LOSS STATISTICS")
    logger.info("="*60)
    logger.info(f"Win Rate:            {metrics['win_rate']:>15.2%}")
    logger.info(f"Average Win:         {metrics['avg_win']:>15.2%}")
    logger.info(f"Average Loss:        {metrics['avg_loss']:>15.2%}")
    logger.info(f"Best Day:            {metrics['best_day']:>15.2%}")
    logger.info(f"Worst Day:           {metrics['worst_day']:>15.2%}")
    logger.info(f"Best Month:          {metrics['best_month']:>15.2%}")
    logger.info(f"Worst Month:         {metrics['worst_month']:>15.2%}")

    # Trading Activity
    logger.info("\n" + "="*60)
    logger.info("TRADING ACTIVITY")
    logger.info("="*60)
    logger.info(f"Rebalances:          {metrics['num_rebalances']:>15.0f}")
    logger.info(f"Total Trades:        {metrics['num_trades']:>15.0f}")
    logger.info(f"Avg Turnover:        {metrics['avg_turnover']:>15.2%}")
    logger.info(f"Transaction Costs:   ${metrics['total_transaction_costs']:>14,.0f}")

    # Benchmark Comparison
    if benchmark is not None:
        benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1

        # Calculate benchmark metrics
        benchmark_returns = benchmark.pct_change().dropna()
        metrics_calc = PerformanceMetrics()
        benchmark_metrics = metrics_calc.calculate_all_metrics(benchmark)

        logger.info("\n" + "="*60)
        logger.info("BENCHMARK COMPARISON (Equal-Weight All ETFs)")
        logger.info("="*60)
        logger.info(f"Strategy Return:     {metrics['total_return']:>15.2%}")
        logger.info(f"Benchmark Return:    {benchmark_return:>15.2%}")
        logger.info(f"Outperformance:      {metrics['total_return'] - benchmark_return:>15.2%}")
        logger.info(f"")
        logger.info(f"Strategy Sharpe:     {metrics['sharpe_ratio']:>15.2f}")
        logger.info(f"Benchmark Sharpe:    {benchmark_metrics['sharpe_ratio']:>15.2f}")
        logger.info(f"")
        logger.info(f"Strategy Max DD:     {metrics['max_drawdown']:>15.2%}")
        logger.info(f"Benchmark Max DD:    {benchmark_metrics['max_drawdown']:>15.2%}")

    # Final Holdings
    logger.info("\n" + "="*60)
    logger.info("FINAL PORTFOLIO")
    logger.info("="*60)
    logger.info(f"Positions:           {metrics['final_positions']:>15.0f}")
    logger.info(f"Cash:                ${metrics['final_cash']:>14,.0f}")
    logger.info(f"Invested:            ${metrics['final_value'] - metrics['final_cash']:>14,.0f}")

    logger.info("\n" + "="*60)


def save_results(results: dict, output_dir: Path):
    """Save backtest results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save daily values
    daily_values_file = output_dir / f"daily_values_{timestamp}.csv"
    results['daily_values'].to_csv(daily_values_file)
    logger.info(f"Saved daily values to: {daily_values_file}")

    # Save daily returns
    daily_returns_file = output_dir / f"daily_returns_{timestamp}.csv"
    results['daily_returns'].to_csv(daily_returns_file)
    logger.info(f"Saved daily returns to: {daily_returns_file}")

    # Save trades
    if len(results['trades']) > 0:
        trades_file = output_dir / f"trades_{timestamp}.csv"
        results['trades'].to_csv(trades_file, index=False)
        logger.info(f"Saved trades to: {trades_file}")

    # Save metrics
    metrics_file = output_dir / f"metrics_{timestamp}.csv"
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to: {metrics_file}")

    # Save turnovers
    if len(results['turnovers']) > 0:
        turnovers_file = output_dir / f"turnovers_{timestamp}.csv"
        results['turnovers'].to_csv(turnovers_file)
        logger.info(f"Saved turnovers to: {turnovers_file}")


def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("$1M STANDARD PORTFOLIO BACKTEST")
    logger.info("="*60)
    logger.info(f"Started: {datetime.now()}")

    # Load data
    logger.info("\nLoading data...")
    prices, factor_scores = load_data()

    # Configure backtest (aligned with AQR plan)
    config = BacktestConfig(
        initial_capital=1_000_000,
        start_date=prices.index[0],
        end_date=prices.index[-1],
        rebalance_frequency='weekly',
        num_positions=20,
        stop_loss_pct=0.12,  # 12% stop-loss (AQR plan: 10-15%)
        use_stop_loss=True,
        risk_free_rate=0.02
    )

    # Run backtest
    logger.info("\nRunning backtest...")
    results = run_backtest(prices, factor_scores, config)

    # Calculate benchmark
    logger.info("\nCalculating benchmark...")
    benchmark = calculate_benchmark(prices, config.initial_capital)

    # Display results
    display_results(results, benchmark)

    # Save results
    output_dir = project_root / "results" / "backtest_1m"
    save_results(results, output_dir)

    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Completed: {datetime.now()}")
    logger.info(f"Log file: {log_file}")

    return results


if __name__ == '__main__':
    results = main()
