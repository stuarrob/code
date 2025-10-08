"""
Backtest Scenario 2: $100k + $10k/month + $100k/year Contributions

Portfolio with regular contributions:
- Initial: $100,000
- Monthly: $10,000
- Annual: $100,000

Calculates time to reach $3M portfolio value.
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


def calculate_time_to_3m(portfolio_values: pd.Series, target: float = 3_000_000) -> dict:
    """
    Calculate when portfolio reaches target value.

    Parameters
    ----------
    portfolio_values : pd.Series
        Portfolio values over time
    target : float
        Target value (default $3M)

    Returns
    -------
    dict
        Time to target information
    """
    reached = portfolio_values >= target

    if not reached.any():
        return {
            'reached': False,
            'date': None,
            'days': None,
            'years': None,
            'final_value': portfolio_values.iloc[-1],
            'percentage_of_target': portfolio_values.iloc[-1] / target
        }

    first_date = reached.idxmax()
    start_date = portfolio_values.index[0]
    days = (first_date - start_date).days
    years = days / 365.25

    return {
        'reached': True,
        'date': first_date,
        'days': days,
        'years': years,
        'value_at_target': portfolio_values[first_date],
        'final_value': portfolio_values.iloc[-1]
    }


def main():
    """Run contribution portfolio backtest."""
    logger.info("="*80)
    logger.info("BACKTEST SCENARIO 2: CONTRIBUTION-BASED PORTFOLIO")
    logger.info("="*80)

    # Configuration
    INITIAL_CAPITAL = 100_000
    MONTHLY_CONTRIBUTION = 10_000
    ANNUAL_CONTRIBUTION = 100_000
    TARGET_VALUE = 3_000_000
    REBALANCE_FREQUENCY = 'monthly'
    LOOKBACK_PERIOD = 252  # 1 year
    VARIANT = 'balanced'
    ENABLE_STOP_LOSS = True
    STOP_LOSS_PCT = 0.10  # 10% (90% of purchase price)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Initial capital:     ${INITIAL_CAPITAL:,.0f}")
    logger.info(f"  Monthly contrib:     ${MONTHLY_CONTRIBUTION:,.0f}")
    logger.info(f"  Annual contrib:      ${ANNUAL_CONTRIBUTION:,.0f}")
    logger.info(f"  Target value:        ${TARGET_VALUE:,.0f}")
    logger.info(f"  Rebalance frequency: {REBALANCE_FREQUENCY}")
    logger.info(f"  Lookback period:     {LOOKBACK_PERIOD} days")
    logger.info(f"  Variant:             {VARIANT}")
    logger.info(f"  Stop-loss:           {STOP_LOSS_PCT*100:.0f}%")

    # Annual contribution schedule
    annual_total = MONTHLY_CONTRIBUTION * 12 + ANNUAL_CONTRIBUTION
    logger.info(f"\n  Total annual contributions: ${annual_total:,.0f}")

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

    # Determine backtest period (use full available history to see time to $3M)
    end_date = prices.index[-1]
    start_date = prices.index[LOOKBACK_PERIOD]

    logger.info(f"\nBacktest period:")
    logger.info(f"  Start: {start_date.date()}")
    logger.info(f"  End:   {end_date.date()}")
    logger.info(f"  Days:  {(end_date - start_date).days}")
    logger.info(f"  Years: {(end_date - start_date).days / 365.25:.1f}")

    # Run backtest
    results = engine.run(
        prices=prices,
        start_date=start_date,
        end_date=end_date,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        annual_contribution=ANNUAL_CONTRIBUTION
    )

    # Calculate time to $3M
    time_to_target = calculate_time_to_3m(results['portfolio_values']['value'], TARGET_VALUE)

    # Print performance metrics
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE RESULTS")
    logger.info("="*80)

    metrics = results['metrics']

    print("\n📊 PORTFOLIO GROWTH")
    print(f"  Initial Value:       ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Value:         ${results['portfolio_values']['value'].iloc[-1]:,.2f}")
    print(f"  Investment Gain:     ${results['portfolio_values']['value'].iloc[-1] - INITIAL_CAPITAL:,.2f}")

    # Calculate total contributions
    test_days = len(results['portfolio_values'])
    test_years = test_days / 252
    total_contributions = INITIAL_CAPITAL
    total_contributions += MONTHLY_CONTRIBUTION * 12 * test_years
    total_contributions += ANNUAL_CONTRIBUTION * test_years

    gains = results['portfolio_values']['value'].iloc[-1] - total_contributions

    print(f"\n💵 CONTRIBUTION ANALYSIS")
    print(f"  Total Contributed:   ${total_contributions:,.2f}")
    print(f"  Investment Gains:    ${gains:,.2f}")
    print(f"  Gain on Capital:     {(gains/total_contributions)*100:>8.2f}%")

    print("\n🎯 TIME TO $3M TARGET")
    if time_to_target['reached']:
        print(f"  ✅ TARGET REACHED!")
        print(f"  Date:                {time_to_target['date'].date()}")
        print(f"  Days:                {time_to_target['days']:,.0f}")
        print(f"  Years:               {time_to_target['years']:.2f}")
        print(f"  Value at target:     ${time_to_target['value_at_target']:,.2f}")
        print(f"  Final value:         ${time_to_target['final_value']:,.2f}")
    else:
        print(f"  ❌ TARGET NOT REACHED")
        print(f"  Current value:       ${time_to_target['final_value']:,.2f}")
        print(f"  Percentage of target: {time_to_target['percentage_of_target']*100:.1f}%")
        years_needed = test_years / time_to_target['percentage_of_target']
        print(f"  Est. years needed:   {years_needed:.1f} years")

    print("\n📈 RETURN METRICS")
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

    # Create visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    visualizer = BacktestVisualizer(output_dir="results/backtests/scenario2_contribution")
    plots = visualizer.create_all_plots(
        results=results,
        initial_capital=INITIAL_CAPITAL,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        annual_contribution=ANNUAL_CONTRIBUTION,
        benchmark=benchmark
    )

    logger.info("\nPlots saved:")
    for name, path in plots.items():
        logger.info(f"  {name}: {path}")

    # Save detailed results
    output_dir = project_root / "results" / "backtests" / "scenario2_contribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save portfolio values
    results['portfolio_values'].to_csv(output_dir / f"portfolio_values_{timestamp}.csv")

    # Save weights history
    results['weights'].to_csv(output_dir / f"weights_history_{timestamp}.csv")

    # Save metrics including time to target
    metrics_with_target = {**metrics, **{f'time_to_3m_{k}': v for k, v in time_to_target.items()}}
    metrics_df = pd.DataFrame([metrics_with_target])
    metrics_df.to_csv(output_dir / f"metrics_{timestamp}.csv", index=False)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\n✅ BACKTEST COMPLETE")


if __name__ == "__main__":
    main()
