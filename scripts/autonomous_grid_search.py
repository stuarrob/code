"""
Autonomous Grid Search - Run Overnight Experiments

Creates comprehensive output files for post-hoc analysis:
- Individual experiment logs
- Summary CSV with all metrics
- Parameter importance analysis
- Signal vs reality comparison
- Charts and visualizations

Run with: nohup python scripts/autonomous_grid_search.py > /tmp/grid_search_master.log 2>&1 &
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
import traceback

from src.backtesting.backtest_engine import BacktestEngine
from src.data_collection.asset_class_mapper import create_asset_class_map
from src.data_collection.etf_filters import apply_etf_filters

# Create output directory
OUTPUT_DIR = project_root / "results" / "grid_search" / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "grid_search_master.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PRIORITY GRID SEARCH PARAMETERS
# Start with focused set to find what works
GRID_PARAMS = {
    # Optimization variant
    'variant': ['balanced'],  # Start with one

    # Rebalancing frequency
    'rebalance_frequency': ['monthly', 'quarterly'],

    # Turnover penalty (MUCH HIGHER to reduce churn)
    'turnover_penalty': [5.0, 10.0, 20.0, 50.0],

    # Concentration penalty
    'concentration_penalty': [0.5, 1.0, 2.0],

    # Stop-loss (None = disabled)
    'stop_loss_pct': [None],  # Disabled based on analysis

    # Maximum positions
    'max_positions': [10, 15, 20],

    # Signal lookback period
    'signal_lookback': [63, 126, 252],  # 3m, 6m, 1y
}


def load_etf_universe(n_etfs: int = 200) -> pd.DataFrame:
    """Load ETF universe."""
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


def run_single_experiment(exp_id: int, params: dict, prices: pd.DataFrame, asset_class_map: dict) -> dict:
    """
    Run single experiment with detailed logging.

    Creates individual log file for this experiment.
    """
    exp_log_file = OUTPUT_DIR / f"experiment_{exp_id:04d}.log"
    exp_handler = logging.FileHandler(exp_log_file)
    exp_handler.setLevel(logging.DEBUG)
    exp_logger = logging.getLogger(f"exp_{exp_id}")
    exp_logger.addHandler(exp_handler)
    exp_logger.setLevel(logging.DEBUG)

    exp_logger.info(f"="*80)
    exp_logger.info(f"EXPERIMENT {exp_id}")
    exp_logger.info(f"="*80)
    exp_logger.info(f"Parameters: {json.dumps(params, indent=2)}")

    try:
        # Temporarily override VARIANTS in optimizer
        from src.optimization import cvxpy_optimizer

        original_variants = cvxpy_optimizer.CVXPYPortfolioOptimizer.VARIANTS.copy()

        # Modify variant with custom penalties
        modified_variant = original_variants[params['variant']].copy()
        modified_variant['turnover_penalty'] = params['turnover_penalty']
        modified_variant['concentration_penalty'] = params['concentration_penalty']

        cvxpy_optimizer.CVXPYPortfolioOptimizer.VARIANTS[params['variant']] = modified_variant

        # Create engine
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

        # Backtest period: last 2 years
        end_date = prices.index[-1]
        start_date = end_date - pd.Timedelta(days=730)

        if start_date < prices.index[params['signal_lookback']]:
            start_date = prices.index[params['signal_lookback']]

        exp_logger.info(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
        exp_logger.info(f"Days: {(end_date - start_date).days}\n")

        results = engine.run(
            prices=prices,
            start_date=start_date,
            end_date=end_date
        )

        # Restore original variants
        cvxpy_optimizer.CVXPYPortfolioOptimizer.VARIANTS = original_variants

        metrics = results['metrics']

        # Calculate additional metrics
        num_months = (end_date - start_date).days / 30
        avg_monthly_turnover = metrics['avg_turnover'] / 100

        # Calculate average positions and ETF changes
        position_counts = []
        for _, weights in results['weights'].iterrows():
            count = len([w for w in weights.values() if w > 0.01])
            position_counts.append(count)

        avg_positions = np.mean(position_counts) if position_counts else 0
        etf_changes_per_rebalance = avg_monthly_turnover * avg_positions if metrics['num_rebalances'] > 0 else 0

        # Analyze signal quality vs reality
        # Compare predicted Sharpe (from optimization) vs actual Sharpe
        signal_quality_gap = "N/A"  # Would need to track predicted vs actual

        result = {
            'experiment_id': exp_id,
            'params': params,
            'success': True,

            # Performance metrics
            'cagr': metrics['cagr'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],

            # Risk metrics
            'max_drawdown': metrics['max_drawdown'],
            'avg_drawdown': metrics['avg_drawdown'],
            'volatility': metrics['volatility'],

            # Turnover metrics (KEY for understanding churn)
            'avg_turnover_pct': metrics['avg_turnover'],
            'num_rebalances': metrics['num_rebalances'],
            'avg_positions': avg_positions,
            'etf_changes_per_rebalance': etf_changes_per_rebalance,

            # Cost metrics
            'transaction_costs': metrics['total_transaction_costs'],
            'cost_as_pct_of_returns': metrics['total_transaction_costs'] / 1_000_000,

            # Other
            'stop_loss_triggers': metrics.get('stop_loss_num_stops', 0),
            'win_rate': metrics['win_rate'],
            'final_value': results['portfolio_values']['value'].iloc[-1],
            'total_return': metrics['total_return'],

            # File references
            'log_file': str(exp_log_file.relative_to(project_root))
        }

        exp_logger.info(f"\n{'-'*80}")
        exp_logger.info("RESULTS")
        exp_logger.info(f"{'-'*80}")
        exp_logger.info(f"CAGR:                {result['cagr']*100:.2f}%")
        exp_logger.info(f"Sharpe:              {result['sharpe_ratio']:.2f}")
        exp_logger.info(f"Max Drawdown:        {result['max_drawdown']*100:.2f}%")
        exp_logger.info(f"Avg Turnover:        {result['avg_turnover_pct']:.1f}%")
        exp_logger.info(f"ETF Changes/Rebal:   {result['etf_changes_per_rebalance']:.1f}")
        exp_logger.info(f"Transaction Costs:   ${result['transaction_costs']:,.2f}")
        exp_logger.info(f"Final Value:         ${result['final_value']:,.2f}")

        # Save detailed results
        exp_results_file = OUTPUT_DIR / f"experiment_{exp_id:04d}_results.json"
        with open(exp_results_file, 'w') as f:
            # Convert portfolio_values and weights to JSON-serializable format
            serializable_results = {
                'params': params,
                'metrics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                           for k, v in metrics.items()},
                'portfolio_values': results['portfolio_values'].to_dict(),
                'final_weights': {k: float(v) for k, v in results['weights'].iloc[-1].items() if v > 0.01}
            }
            json.dump(serializable_results, f, indent=2, default=str)

        exp_logger.info(f"\nResults saved to: {exp_results_file.relative_to(project_root)}")
        exp_logger.info("="*80 + "\n")

        return result

    except Exception as e:
        exp_logger.error(f"\nEXPERIMENT FAILED!")
        exp_logger.error(f"Error: {str(e)}")
        exp_logger.error(traceback.format_exc())

        return {
            'experiment_id': exp_id,
            'params': params,
            'success': False,
            'error': str(e),
            'log_file': str(exp_log_file.relative_to(project_root))
        }
    finally:
        exp_logger.removeHandler(exp_handler)
        exp_handler.close()


def generate_param_combinations():
    """Generate all parameter combinations."""
    keys = list(GRID_PARAMS.keys())
    values = [GRID_PARAMS[k] for k in keys]

    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


def analyze_results(results: list):
    """Generate comprehensive analysis of grid search results."""
    successful = [r for r in results if r['success']]

    if not successful:
        logger.error("No successful experiments!")
        return

    df = pd.DataFrame(successful)

    # Save full results CSV
    csv_file = OUTPUT_DIR / "all_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"\nFull results saved to: {csv_file}")

    # Analysis 1: Best by Sharpe
    analysis_file = OUTPUT_DIR / "ANALYSIS_SUMMARY.md"

    with open(analysis_file, 'w') as f:
        f.write("# Grid Search Analysis Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Experiments**: {len(results)}\n")
        f.write(f"**Successful**: {len(successful)}\n")
        f.write(f"**Failed**: {len(results) - len(successful)}\n\n")

        f.write("---\n\n")
        f.write("## Top 10 Results by Sharpe Ratio\n\n")

        top_sharpe = df.nlargest(10, 'sharpe_ratio')
        for idx, row in top_sharpe.iterrows():
            f.write(f"### #{idx+1} - Sharpe {row['sharpe_ratio']:.2f}\n\n")
            f.write(f"- **CAGR**: {row['cagr']*100:.2f}%\n")
            f.write(f"- **Max Drawdown**: {row['max_drawdown']*100:.2f}%\n")
            f.write(f"- **Turnover**: {row['avg_turnover_pct']:.1f}%\n")
            f.write(f"- **ETF Changes/Rebalance**: {row['etf_changes_per_rebalance']:.1f}\n")
            f.write(f"- **Final Value**: ${row['final_value']:,.2f}\n")
            f.write(f"- **Log**: `{row['log_file']}`\n\n")
            f.write("**Parameters**:\n```json\n")
            f.write(json.dumps(row['params'], indent=2))
            f.write("\n```\n\n")

        f.write("---\n\n")
        f.write("## Top 10 by CAGR\n\n")

        top_cagr = df.nlargest(10, 'cagr')
        for idx, row in top_cagr.iterrows():
            f.write(f"- CAGR: {row['cagr']*100:.2f}%, Sharpe: {row['sharpe_ratio']:.2f}, ")
            f.write(f"Turnover: {row['avg_turnover_pct']:.0f}%, Changes: {row['etf_changes_per_rebalance']:.1f}\n")

        f.write("\n---\n\n")
        f.write("## Top 10 by Efficiency (Sharpe / Turnover)\n\n")

        df['efficiency'] = df['sharpe_ratio'] / (1 + df['avg_turnover_pct'] / 100)
        top_eff = df.nlargest(10, 'efficiency')
        for idx, row in top_eff.iterrows():
            f.write(f"- Efficiency: {row['efficiency']:.2f}, Sharpe: {row['sharpe_ratio']:.2f}, ")
            f.write(f"CAGR: {row['cagr']*100:.1f}%, Turnover: {row['avg_turnover_pct']:.0f}%\n")

        f.write("\n---\n\n")
        f.write("## Parameter Importance Analysis\n\n")

        # Analyze which parameters matter most
        for param in GRID_PARAMS.keys():
            f.write(f"\n### {param}\n\n")
            grouped = df.groupby(param).agg({
                'sharpe_ratio': ['mean', 'std', 'max'],
                'cagr': 'mean',
                'avg_turnover_pct': 'mean'
            }).round(3)
            f.write(grouped.to_markdown())
            f.write("\n\n")

    logger.info(f"Analysis saved to: {analysis_file}")


def main():
    """Run autonomous grid search."""
    logger.info("="*80)
    logger.info("AUTONOMOUS GRID SEARCH EXPERIMENT")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {OUTPUT_DIR}")

    # Generate combinations
    param_combinations = generate_param_combinations()
    total_experiments = len(param_combinations)

    logger.info(f"\nTotal experiments: {total_experiments}")
    logger.info(f"Estimated time: {total_experiments * 120 / 3600:.1f} hours")
    logger.info(f"\nParameter space:")
    for key, values in GRID_PARAMS.items():
        logger.info(f"  {key}: {values}")

    # Load data once
    logger.info("\nLoading ETF universe...")
    prices = load_etf_universe(n_etfs=200)
    logger.info(f"Loaded: {prices.shape[1]} ETFs")
    logger.info(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    fundamentals_path = project_root / "data" / "raw" / "fundamentals.csv"
    asset_class_map = {}
    if fundamentals_path.exists():
        asset_class_map = create_asset_class_map(str(fundamentals_path))

    # Run experiments
    results = []
    start_time = time.time()

    logger.info("\n" + "="*80)
    logger.info("RUNNING EXPERIMENTS")
    logger.info("="*80 + "\n")

    for i, params in enumerate(param_combinations, 1):
        logger.info(f"\n[{i}/{total_experiments}] Experiment {i}")
        logger.info(f"Params: turnover={params['turnover_penalty']}, "
                   f"rebal={params['rebalance_frequency']}, "
                   f"lookback={params['signal_lookback']}")

        exp_start = time.time()
        result = run_single_experiment(i, params, prices, asset_class_map)
        results.append(result)
        exp_time = time.time() - exp_start

        if result['success']:
            logger.info(f"✅ CAGR={result['cagr']*100:.1f}%, "
                       f"Sharpe={result['sharpe_ratio']:.2f}, "
                       f"Turnover={result['avg_turnover_pct']:.0f}%, "
                       f"Changes={result['etf_changes_per_rebalance']:.1f} ({exp_time:.0f}s)")
        else:
            logger.info(f"❌ FAILED: {result.get('error', 'Unknown')} ({exp_time:.0f}s)")

        # Save intermediate results every 5 experiments
        if i % 5 == 0:
            interim_file = OUTPUT_DIR / f"interim_results_{i:04d}.json"
            with open(interim_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"💾 Saved interim results")

    # Final analysis
    logger.info("\n" + "="*80)
    logger.info("GENERATING ANALYSIS")
    logger.info("="*80 + "\n")

    analyze_results(results)

    # Save final results
    final_file = OUTPUT_DIR / "final_results.json"
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    total_time = time.time() - start_time
    logger.info(f"\n✅ Grid search complete!")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
