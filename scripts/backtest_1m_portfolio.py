"""
Backtest Scenario 1: $1M Initial Portfolio

Standard backtest with $1,000,000 initial capital.
Tests rolling window optimization with transaction costs and stop-loss.

Performance metrics:
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- CAGR
- Volatility
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.visualizations import BacktestVisualizer
from src.data_collection.asset_class_mapper import create_asset_class_map
from src.data_collection.etf_filters import apply_etf_filters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_etf_universe(n_etfs: int = 300) -> pd.DataFrame:
    """Load ETF universe for backtesting."""
    prices_dir = project_root / "data" / "raw" / "prices"
    etf_files = list(prices_dir.glob("*.csv"))

    logger.info(f"Loading {n_etfs}-ETF universe for backtesting...")

    # Load and score by data quality
    etf_scores = []
    for file in etf_files:
        try:
            df = pd.read_csv(file)
            date_col = next((col for col in df.columns if col.lower() == 'date'), None)
            if date_col is None:
                continue

            df[date_col] = pd.to_datetime(df[date_col])
            df.columns = [col.capitalize() for col in df.columns]

            # Score by length and completeness
            score = len(df) * (1 - df["Close"].isna().sum() / len(df))
            etf_scores.append({
                "ticker": file.stem,
                "score": score,
                "length": len(df),
                "file": file
            })
        except:
            continue

    # Sort and take top N
    etf_scores_df = pd.DataFrame(etf_scores)
    etf_scores_df = etf_scores_df.sort_values("score", ascending=False)
    top_etfs = etf_scores_df.head(n_etfs)

    # Load price data
    prices = {}
    for _, row in top_etfs.iterrows():
        df = pd.read_csv(row["file"])
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.columns = [col.capitalize() for col in df.columns]
            df = df.set_index("Date")
            prices[row["ticker"]] = df["Close"]

    prices_df = pd.DataFrame(prices).sort_index()

    # Apply filters
    prices_df = apply_etf_filters(
        prices_df,
        filter_leveraged=True,
        filter_high_volatility=True,
        max_volatility=0.35
    )

    logger.info(f"Loaded {prices_df.shape[1]} ETFs after filtering")
    logger.info(f"Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    logger.info(f"Total days: {len(prices_df)}")

    return prices_df


def load_benchmark() -> pd.Series:
    """Load SPY as benchmark."""
    spy_path = project_root / "data" / "raw" / "prices" / "SPY.csv"
    if not spy_path.exists():
        return None

    df = pd.read_csv(spy_path)
    date_col = next((col for col in df.columns if col.lower() == 'date'), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.columns = [col.capitalize() for col in df.columns]
        df = df.set_index("Date")
        return df["Close"]

    return None


def main():
    """Run $1M portfolio backtest."""
    logger.info("="*80)
    logger.info("BACKTEST SCENARIO 1: $1M INITIAL PORTFOLIO")
    logger.info("="*80)

    # Configuration
    INITIAL_CAPITAL = 1_000_000
    REBALANCE_FREQUENCY = 'monthly'  # monthly, quarterly
    LOOKBACK_PERIOD = 252  # 1 year
    VARIANT = 'balanced'
    ENABLE_STOP_LOSS = False  # DISABLED: Counterproductive
    STOP_LOSS_PCT = 0.10  # 10% (90% of purchase price)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Initial capital:     ${INITIAL_CAPITAL:,.0f}")
    logger.info(f"  Rebalance frequency: {REBALANCE_FREQUENCY}")
    logger.info(f"  Lookback period:     {LOOKBACK_PERIOD} days")
    logger.info(f"  Variant:             {VARIANT}")
    logger.info(f"  Stop-loss:           {STOP_LOSS_PCT*100:.0f}%")

    # Load data
    prices = load_etf_universe(n_etfs=300)
    benchmark = load_benchmark()

    # Load asset class map
    fundamentals_path = project_root / "data" / "raw" / "fundamentals.csv"
    asset_class_map = {}
    if fundamentals_path.exists():
        asset_class_map = create_asset_class_map(str(fundamentals_path))
        logger.info(f"  Asset class map:     {len(asset_class_map)} ETFs mapped")

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        rebalance_frequency=REBALANCE_FREQUENCY,
        lookback_period=LOOKBACK_PERIOD,
        variant=VARIANT,
        enable_stop_loss=ENABLE_STOP_LOSS,
        stop_loss_pct=STOP_LOSS_PCT,
        enable_transaction_costs=True,
        risk_free_rate=0.04,
        asset_class_map=asset_class_map
    )

    # Determine backtest period (use last 2 years with sufficient warmup)
    end_date = prices.index[-1]
    start_date = end_date - pd.Timedelta(days=730)  # 2 years

    # Ensure we have enough warmup data
    if start_date < prices.index[LOOKBACK_PERIOD]:
        start_date = prices.index[LOOKBACK_PERIOD]

    logger.info(f"\nBacktest period:")
    logger.info(f"  Start: {start_date.date()}")
    logger.info(f"  End:   {end_date.date()}")
    logger.info(f"  Days:  {(end_date - start_date).days}")

    # Run backtest
    results = engine.run(
        prices=prices,
        start_date=start_date,
        end_date=end_date,
        monthly_contribution=0,
        annual_contribution=0
    )

    # Print performance metrics
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE RESULTS")
    logger.info("="*80)

    metrics = results['metrics']

    print("\n📊 RETURN METRICS")
    print(f"  Initial Value:       ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Value:         ${results['portfolio_values']['value'].iloc[-1]:,.2f}")
    print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
    print(f"  CAGR:                {metrics['cagr']*100:>8.2f}%")
    print(f"  Volatility:          {metrics['volatility']*100:>8.2f}%")

    print("\n⚖️  RISK-ADJUSTED METRICS")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>8.2f}")

    print("\n📉 DRAWDOWN METRICS")
    print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
    print(f"  Avg Drawdown:        {metrics['avg_drawdown']*100:>8.2f}%")
    print(f"  Max DD Duration:     {metrics['max_drawdown_duration']:>8.0f} days")

    print("\n💰 TRANSACTION COSTS")
    print(f"  Total Costs:         ${metrics['total_transaction_costs']:>,.2f}")
    print(f"  Num Rebalances:      {metrics['num_rebalances']:>8.0f}")
    print(f"  Avg Turnover:        {metrics['avg_turnover']*100:>8.1f}%")

    if ENABLE_STOP_LOSS:
        print("\n🛑 STOP-LOSS ACTIVITY")
        print(f"  Stops Triggered:     {metrics.get('stop_loss_num_stops', 0):>8.0f}")
        if metrics.get('stop_loss_num_stops', 0) > 0:
            print(f"  Total Loss:          ${metrics.get('stop_loss_total_loss', 0):>,.2f}")
            print(f"  Avg Loss:            {metrics.get('stop_loss_avg_loss_pct', 0)*100:>8.2f}%")

    print("\n📊 DISTRIBUTION METRICS")
    print(f"  Win Rate:            {metrics['win_rate']*100:>8.2f}%")
    print(f"  Best Day:            {metrics['best_day']*100:>8.2f}%")
    print(f"  Worst Day:           {metrics['worst_day']*100:>8.2f}%")

    # Create visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    visualizer = BacktestVisualizer(output_dir="results/backtests/scenario1_1m")
    plots = visualizer.create_all_plots(
        results=results,
        initial_capital=INITIAL_CAPITAL,
        benchmark=benchmark
    )

    logger.info("\nPlots saved:")
    for name, path in plots.items():
        logger.info(f"  {name}: {path}")

    # Save detailed results
    output_dir = project_root / "results" / "backtests" / "scenario1_1m"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save portfolio values
    results['portfolio_values'].to_csv(output_dir / f"portfolio_values_{timestamp}.csv")

    # Save weights history
    results['weights'].to_csv(output_dir / f"weights_history_{timestamp}.csv")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / f"metrics_{timestamp}.csv", index=False)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\n✅ BACKTEST COMPLETE")


if __name__ == "__main__":
    main()
