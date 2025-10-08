"""
Grid Search Backtest Experiment

Systematically test combinations of:
1. Signal parameters (momentum periods, RSI thresholds, etc.)
2. Optimization parameters (turnover penalty, concentration, rebalance frequency)
3. Stop-loss levels
4. Cash allocation strategies

Goal: Find robust parameters with:
- Low turnover (1-2 ETF changes per month)
- High risk-adjusted returns (Sharpe > 1.0)
- Reasonable drawdowns (<20%)
- Real out-of-sample performance

This is a LONG-RUNNING experiment that generates results for post-hoc analysis.
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
from itertools import product
import time

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.data_collection.asset_class_mapper import create_asset_class_map
from src.data_collection.etf_filters import apply_etf_filters

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise for grid search
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Grid search parameter space
GRID_PARAMS = {
    # Optimization variant
    'variant': ['max_sharpe', 'balanced', 'min_drawdown'],

    # Rebalancing frequency
    'rebalance_frequency': ['monthly', 'quarterly'],

    # Turnover penalty multipliers (higher = less churn)
    'turnover_multiplier': [1.0, 5.0, 10.0, 20.0, 50.0],

    # Concentration penalty multipliers
    'concentration_multiplier': [0.5, 1.0, 2.0],

    # Stop-loss (None = disabled)
    'stop_loss_pct': [None, 0.15, 0.20],

    # Maximum positions
    'max_positions': [10, 15, 20],

    # Cash allocation (force min cash %)
    'min_cash_pct': [0.0, 0.10, 0.20],

    # Signal lookback period
    'signal_lookback': [20, 63, 126],  # 1 month, 3 months, 6 months
}


def load_etf_universe(n_etfs: int = 200) -> pd.DataFrame:
    """Load smaller universe for faster grid search."""
    prices_dir = project_root / "data" / "raw" / "prices"
    etf_files = list(prices_dir.glob("*.csv"))

    logger.info(f"Loading {n_etfs}-ETF universe...")

    etf_scores = []
    for file in etf_files:
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
                "length": len(df),
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

    return prices_df


def run_single_backtest(params: dict, prices: pd.DataFrame, asset_class_map: dict) -> dict:
    """Run single backtest with given parameters."""
    try:
        # Create modified variant with turnover/concentration multipliers
        # (This is a placeholder - we'll need to modify the optimizer to accept these)

        engine = BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency=params['rebalance_frequency'],
            lookback_period=params['signal_lookback'],
            variant=params['variant'],
            enable_stop_loss=params['stop_loss_pct'] is not None,
            stop_loss_pct=params['stop_loss_pct'] if params['stop_loss_pct'] else 0.10,
            enable_transaction_costs=True,
            risk_free_rate=0.04,
            asset_class_map=asset_class_map
        )

        # Use last 3 years for backtest
        end_date = prices.index[-1]
        start_date = end_date - pd.Timedelta(days=1095)  # 3 years

        if start_date < prices.index[params['signal_lookback']]:
            start_date = prices.index[params['signal_lookback']]

        results = engine.run(
            prices=prices,
            start_date=start_date,
            end_date=end_date
        )

        metrics = results['metrics']

        # Calculate turnover per month
        num_months = (end_date - start_date).days / 30
        avg_monthly_turnover = metrics['avg_turnover'] / 100  # Convert to decimal

        # Estimate ETF changes per month
        avg_positions = np.mean([len([w for w in weights.values() if w > 0.01])
                                for _, weights in results['weights'].iterrows()])
        etf_changes_per_month = avg_monthly_turnover * avg_positions if metrics['num_rebalances'] > 0 else 0

        return {
            'params': params,
            'success': True,
            'cagr': metrics['cagr'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'volatility': metrics['volatility'],
            'avg_turnover': metrics['avg_turnover'],
            'num_rebalances': metrics['num_rebalances'],
            'transaction_costs': metrics['total_transaction_costs'],
            'stop_loss_triggers': metrics.get('stop_loss_num_stops', 0),
            'win_rate': metrics['win_rate'],
            'avg_positions': avg_positions,
            'etf_changes_per_month': etf_changes_per_month,
            'final_value': results['portfolio_values']['value'].iloc[-1]
        }

    except Exception as e:
        logger.error(f"Backtest failed with params {params}: {e}")
        return {
            'params': params,
            'success': False,
            'error': str(e)
        }


def generate_param_combinations():
    """Generate all parameter combinations for grid search."""
    keys = list(GRID_PARAMS.keys())
    values = [GRID_PARAMS[k] for k in keys]

    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


def main():
    """Run grid search experiment."""
    print("="*80)
    print("GRID SEARCH BACKTEST EXPERIMENT")
    print("="*80)

    # Generate all parameter combinations
    param_combinations = generate_param_combinations()
    total_experiments = len(param_combinations)

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Estimated time: {total_experiments * 30 / 3600:.1f} hours (30s per experiment)")
    print(f"\nParameter space:")
    for key, values in GRID_PARAMS.items():
        print(f"  {key}: {values}")

    # Load data
    print("\nLoading ETF universe...")
    prices = load_etf_universe(n_etfs=200)  # Smaller for speed
    print(f"Loaded: {prices.shape[1]} ETFs")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    fundamentals_path = project_root / "data" / "raw" / "fundamentals.csv"
    asset_class_map = {}
    if fundamentals_path.exists():
        asset_class_map = create_asset_class_map(str(fundamentals_path))

    # Run grid search
    results = []
    start_time = time.time()

    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80 + "\n")

    for i, params in enumerate(param_combinations, 1):
        exp_start = time.time()

        print(f"[{i}/{total_experiments}] Testing: "
              f"variant={params['variant']}, "
              f"rebal={params['rebalance_frequency']}, "
              f"turnover_mult={params['turnover_multiplier']}, "
              f"stop_loss={params['stop_loss_pct']}")

        result = run_single_backtest(params, prices, asset_class_map)
        results.append(result)

        exp_time = time.time() - exp_start

        if result['success']:
            print(f"  ✅ CAGR={result['cagr']*100:.1f}%, "
                  f"Sharpe={result['sharpe_ratio']:.2f}, "
                  f"Turnover={result['avg_turnover']:.0f}%, "
                  f"Changes/mo={result['etf_changes_per_month']:.1f} "
                  f"({exp_time:.1f}s)")
        else:
            print(f"  ❌ FAILED: {result['error']} ({exp_time:.1f}s)")

        # Save intermediate results every 10 experiments
        if i % 10 == 0:
            output_dir = project_root / "results" / "grid_search"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            interim_file = output_dir / f"grid_search_interim_{timestamp}.json"

            with open(interim_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"  💾 Saved interim results ({i} experiments)")

    # Save final results
    output_dir = project_root / "results" / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"grid_search_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary DataFrame
    successful_results = [r for r in results if r['success']]

    if successful_results:
        df = pd.DataFrame(successful_results)

        # Save detailed results
        csv_file = output_dir / f"grid_search_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        # Print top results
        print("\n" + "="*80)
        print("TOP 10 RESULTS BY SHARPE RATIO")
        print("="*80 + "\n")

        top_sharpe = df.nlargest(10, 'sharpe_ratio')
        for idx, row in top_sharpe.iterrows():
            print(f"Sharpe={row['sharpe_ratio']:.2f}, "
                  f"CAGR={row['cagr']*100:.1f}%, "
                  f"Turnover={row['avg_turnover']:.0f}%, "
                  f"Changes/mo={row['etf_changes_per_month']:.1f}")
            print(f"  Params: {row['params']}\n")

        print("\n" + "="*80)
        print("TOP 10 RESULTS BY CAGR")
        print("="*80 + "\n")

        top_cagr = df.nlargest(10, 'cagr')
        for idx, row in top_cagr.iterrows():
            print(f"CAGR={row['cagr']*100:.1f}%, "
                  f"Sharpe={row['sharpe_ratio']:.2f}, "
                  f"Turnover={row['avg_turnover']:.0f}%, "
                  f"Changes/mo={row['etf_changes_per_month']:.1f}")
            print(f"  Params: {row['params']}\n")

        print("\n" + "="*80)
        print("TOP 10 RESULTS BY LOW TURNOVER + HIGH SHARPE")
        print("="*80 + "\n")

        # Score: Sharpe / (1 + turnover%)
        df['efficiency_score'] = df['sharpe_ratio'] / (1 + df['avg_turnover'] / 100)
        top_efficient = df.nlargest(10, 'efficiency_score')

        for idx, row in top_efficient.iterrows():
            print(f"Efficiency={row['efficiency_score']:.2f}, "
                  f"Sharpe={row['sharpe_ratio']:.2f}, "
                  f"CAGR={row['cagr']*100:.1f}%, "
                  f"Turnover={row['avg_turnover']:.0f}%, "
                  f"Changes/mo={row['etf_changes_per_month']:.1f}")
            print(f"  Params: {row['params']}\n")

    total_time = time.time() - start_time
    print(f"\n✅ Grid search complete")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {output_dir}")
    print(f"  JSON: {results_file}")
    if successful_results:
        print(f"  CSV: {csv_file}")


if __name__ == "__main__":
    main()
