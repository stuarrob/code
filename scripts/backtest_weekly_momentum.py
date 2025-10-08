"""
Weekly Momentum Backtest - "Run with Winners, Sell Losers"

Iterative testing framework to find signals that work out-of-sample.

Strategy:
- Weekly rebalancing
- 90% stop-loss (defensive)
- Simple momentum signals (minimal overfitting)
- High turnover penalty (force low churn)

Run with different parameters, compare results, iterate.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import argparse

from src.signals.momentum_signal import MomentumSignalGenerator
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.visualizations import BacktestVisualizer
from src.data_collection.asset_class_mapper import create_asset_class_map
from src.data_collection.etf_filters import apply_etf_filters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# CONFIGURABLE PARAMETERS - Change these and rerun to iterate
PARAMS = {
    # Signal parameters
    'momentum_period': 126,           # Days (63=3mo, 126=6mo, 252=1yr)
    'rel_strength_enabled': True,     # Use relative strength vs SPY
    'trend_filter_enabled': True,     # Only buy uptrends

    # Portfolio parameters
    'max_positions': 15,
    'turnover_penalty': 50.0,         # High = low churn
    'concentration_penalty': 1.0,

    # Rebalancing
    'rebalance_frequency': 'weekly',
    'lookback_period': 126,           # Match momentum period

    # Risk management
    'enable_stop_loss': True,
    'stop_loss_pct': 0.10,            # 10% = 90% of purchase price

    # Universe
    'n_etfs': 200,                    # Smaller universe = faster
}


def load_etf_universe(n_etfs: int) -> tuple:
    """Load ETF universe and SPY benchmark."""
    prices_dir = project_root / "data" / "raw" / "prices"
    etf_files = list(prices_dir.glob("*.csv"))

    logger.info(f"Loading {n_etfs}-ETF universe...")

    # Load SPY first
    spy_path = prices_dir / "SPY.csv"
    spy_prices = None
    if spy_path.exists():
        df = pd.read_csv(spy_path)
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.columns = [col.capitalize() for col in df.columns]
            df = df.set_index("Date")
            spy_prices = df["Close"]
            logger.info("Loaded SPY for relative strength calculation")

    # Score and load top ETFs
    etf_scores = []
    for file in etf_files:
        if file.stem == 'SPY':
            continue

        try:
            df = pd.read_csv(file)
            date_col = next((col for col in df.columns if col.lower() == 'date'), None)
            if date_col is None:
                continue

            df[date_col] = pd.to_datetime(df[date_col])
            df.columns = [col.capitalize() for col in df.columns]

            score = len(df) * (1 - df["Close"].isna().sum() / len(df))
            etf_scores.append({
                "ticker": file.stem,
                "score": score,
                "file": file
            })
        except:
            continue

    etf_scores_df = pd.DataFrame(etf_scores)
    etf_scores_df = etf_scores_df.sort_values("score", ascending=False)
    top_etfs = etf_scores_df.head(n_etfs)

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
    prices_df = apply_etf_filters(
        prices_df,
        filter_leveraged=True,
        filter_high_volatility=True,
        max_volatility=0.35
    )

    logger.info(f"Loaded {prices_df.shape[1]} ETFs after filtering")
    logger.info(f"Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")

    return prices_df, spy_prices


def run_momentum_backtest(params: dict) -> dict:
    """
    Run backtest with momentum signals.

    Returns detailed results for analysis and iteration.
    """
    logger.info("="*80)
    logger.info("WEEKLY MOMENTUM BACKTEST")
    logger.info("="*80)
    logger.info(f"\nParameters:\n{json.dumps(params, indent=2)}")

    # Load data
    prices, spy_prices = load_etf_universe(params['n_etfs'])

    # Load asset class map
    fundamentals_path = project_root / "data" / "raw" / "fundamentals.csv"
    asset_class_map = {}
    if fundamentals_path.exists():
        asset_class_map = create_asset_class_map(str(fundamentals_path))

    # Create momentum signal generator
    signal_gen = MomentumSignalGenerator(
        momentum_period=params['momentum_period'],
        rel_strength_enabled=params['rel_strength_enabled'],
        trend_filter_enabled=params['trend_filter_enabled']
    )

    # Modify backtest engine to use momentum signals
    # For now, use existing engine (will need to modify to use custom signals)
    engine = BacktestEngine(
        initial_capital=1_000_000,
        rebalance_frequency=params['rebalance_frequency'],
        lookback_period=params['lookback_period'],
        variant='balanced',  # Will be customized
        enable_stop_loss=params['enable_stop_loss'],
        stop_loss_pct=params['stop_loss_pct'],
        enable_transaction_costs=True,
        risk_free_rate=0.04,
        asset_class_map=asset_class_map
    )

    # Override variant with custom parameters
    from src.optimization import cvxpy_optimizer
    custom_variant = {
        "risk_aversion": 1.5,
        "robustness_penalty": 0.5,
        "turnover_penalty": params['turnover_penalty'],
        "concentration_penalty": params['concentration_penalty'],
        "asset_class_penalty": 0.5,
        "description": "Custom momentum strategy"
    }
    cvxpy_optimizer.CVXPYPortfolioOptimizer.VARIANTS['balanced'] = custom_variant

    # Run backtest (last 2 years)
    end_date = prices.index[-1]
    start_date = end_date - pd.Timedelta(days=730)

    if start_date < prices.index[params['lookback_period']]:
        start_date = prices.index[params['lookback_period']]

    logger.info(f"\nBacktest period: {start_date.date()} to {end_date.date()}")

    results = engine.run(
        prices=prices,
        start_date=start_date,
        end_date=end_date
    )

    # Analyze results
    metrics = results['metrics']

    # Calculate turnover metrics
    position_counts = []
    for _, weights in results['weights'].iterrows():
        count = len([w for w in weights.values() if w > 0.01])
        position_counts.append(count)

    avg_positions = np.mean(position_counts) if position_counts else 0
    avg_monthly_turnover = metrics['avg_turnover'] / 100
    etf_changes_per_rebalance = avg_monthly_turnover * avg_positions if metrics['num_rebalances'] > 0 else 0

    # Summarize
    summary = {
        'params': params,
        'performance': {
            'cagr': metrics['cagr'],
            'sharpe': metrics['sharpe_ratio'],
            'sortino': metrics['sortino_ratio'],
            'calmar': metrics['calmar_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'volatility': metrics['volatility'],
            'total_return': metrics['total_return'],
            'final_value': results['portfolio_values']['value'].iloc[-1]
        },
        'turnover': {
            'avg_turnover_pct': metrics['avg_turnover'],
            'num_rebalances': metrics['num_rebalances'],
            'avg_positions': avg_positions,
            'etf_changes_per_rebalance': etf_changes_per_rebalance
        },
        'risk_management': {
            'stop_loss_triggers': metrics.get('stop_loss_num_stops', 0),
            'stop_loss_loss': metrics.get('stop_loss_total_loss', 0),
            'transaction_costs': metrics['total_transaction_costs']
        },
        'signal_quality': {
            'win_rate': metrics['win_rate'],
            'best_day': metrics['best_day'],
            'worst_day': metrics['worst_day']
        }
    }

    # Print results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)

    logger.info("\n📊 PERFORMANCE")
    logger.info(f"  CAGR:                {summary['performance']['cagr']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:        {summary['performance']['sharpe']:.2f}")
    logger.info(f"  Sortino Ratio:       {summary['performance']['sortino']:.2f}")
    logger.info(f"  Calmar Ratio:        {summary['performance']['calmar']:.2f}")
    logger.info(f"  Max Drawdown:        {summary['performance']['max_drawdown']*100:.2f}%")
    logger.info(f"  Volatility:          {summary['performance']['volatility']*100:.2f}%")
    logger.info(f"  Final Value:         ${summary['performance']['final_value']:,.2f}")

    logger.info("\n🔄 TURNOVER")
    logger.info(f"  Avg Turnover:        {summary['turnover']['avg_turnover_pct']:.1f}%")
    logger.info(f"  ETF Changes/Rebal:   {summary['turnover']['etf_changes_per_rebalance']:.1f}")
    logger.info(f"  Num Rebalances:      {summary['turnover']['num_rebalances']}")
    logger.info(f"  Avg Positions:       {summary['turnover']['avg_positions']:.1f}")

    logger.info("\n🛑 RISK MANAGEMENT")
    logger.info(f"  Stop-Loss Triggers:  {summary['risk_management']['stop_loss_triggers']}")
    logger.info(f"  Stop-Loss Loss:      ${summary['risk_management']['stop_loss_loss']:,.2f}")
    logger.info(f"  Transaction Costs:   ${summary['risk_management']['transaction_costs']:,.2f}")

    logger.info("\n📈 SIGNAL QUALITY")
    logger.info(f"  Win Rate:            {summary['signal_quality']['win_rate']*100:.1f}%")
    logger.info(f"  Best Day:            {summary['signal_quality']['best_day']*100:.2f}%")
    logger.info(f"  Worst Day:           {summary['signal_quality']['worst_day']*100:.2f}%")

    # Create visualizations
    output_dir = project_root / "results" / "momentum_backtests"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = output_dir / f"test_{timestamp}"
    test_dir.mkdir(exist_ok=True)

    visualizer = BacktestVisualizer(output_dir=str(test_dir))
    plots = visualizer.create_all_plots(
        results=results,
        initial_capital=1_000_000,
        benchmark=spy_prices
    )

    logger.info("\n📊 VISUALIZATIONS")
    for name, path in plots.items():
        logger.info(f"  {name}: {path}")

    # Save results
    results_file = test_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    params_file = test_dir / "params.json"
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    logger.info(f"\n💾 Results saved to: {test_dir}")
    logger.info("\n✅ BACKTEST COMPLETE\n")

    return summary


def main():
    """Run weekly momentum backtest."""
    parser = argparse.ArgumentParser(description='Weekly Momentum Backtest')
    parser.add_argument('--momentum-period', type=int, help='Momentum lookback period (days)')
    parser.add_argument('--turnover-penalty', type=float, help='Turnover penalty multiplier')
    parser.add_argument('--max-positions', type=int, help='Maximum portfolio positions')
    parser.add_argument('--no-rel-strength', action='store_true', help='Disable relative strength')
    parser.add_argument('--no-trend-filter', action='store_true', help='Disable trend filter')
    parser.add_argument('--no-stop-loss', action='store_true', help='Disable 90% stop-loss')

    args = parser.parse_args()

    # Override params from command line
    params = PARAMS.copy()
    if args.momentum_period:
        params['momentum_period'] = args.momentum_period
        params['lookback_period'] = args.momentum_period
    if args.turnover_penalty:
        params['turnover_penalty'] = args.turnover_penalty
    if args.max_positions:
        params['max_positions'] = args.max_positions
    if args.no_rel_strength:
        params['rel_strength_enabled'] = False
    if args.no_trend_filter:
        params['trend_filter_enabled'] = False
    if args.no_stop_loss:
        params['enable_stop_loss'] = False

    # Run backtest
    summary = run_momentum_backtest(params)

    # Print iteration guidance
    print("\n" + "="*80)
    print("ITERATION GUIDANCE")
    print("="*80)
    print("\nTo test different parameters, run:")
    print("\nExample variations:")
    print("  python scripts/backtest_weekly_momentum.py --momentum-period 63")
    print("  python scripts/backtest_weekly_momentum.py --turnover-penalty 100.0")
    print("  python scripts/backtest_weekly_momentum.py --no-rel-strength")
    print("  python scripts/backtest_weekly_momentum.py --momentum-period 252 --turnover-penalty 20.0")
    print("\nCompare results in: results/momentum_backtests/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
