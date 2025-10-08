"""
COMPLETE CVXPY Optimization Test Suite

Runs ALL three universe sizes with CVXPY optimizer:
- 100-ETF Pilot
- 300-ETF Medium
- 753-ETF Full

Provides comprehensive results for PHASE3 documentation.
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
import time

from src.optimization.cvxpy_optimizer import create_optimizer
from src.optimization.constraints import PortfolioConstraints, RiskConstraints
from src.signals.composite_signal import CompositeSignalGenerator
from src.data_collection.etf_filters import apply_etf_filters
from src.data_collection.asset_class_mapper import create_asset_class_map

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_asset_class_map() -> dict:
    """Load asset class mapping."""
    fundamentals_path = project_root / "data" / "raw" / "fundamentals.csv"
    if not fundamentals_path.exists():
        return {}

    asset_class_map = create_asset_class_map(str(fundamentals_path))
    logger.info(f"Loaded asset class mapping for {len(asset_class_map)} ETFs")
    return asset_class_map


def load_etf_universe(n_etfs: int) -> pd.DataFrame:
    """Load ETF universe of specified size."""
    prices_dir = project_root / "data" / "raw" / "prices"
    etf_files = list(prices_dir.glob("*.csv"))

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

    return prices_df


def generate_signals(prices: pd.DataFrame) -> pd.Series:
    """Generate composite signals."""
    signal_gen = CompositeSignalGenerator()
    signals = {}

    for ticker in prices.columns:
        try:
            df = pd.DataFrame({
                "Open": prices[ticker],
                "High": prices[ticker],
                "Low": prices[ticker],
                "Close": prices[ticker],
                "Volume": 1000000
            })
            scores_df = signal_gen.generate_signals_for_etf(df)
            if scores_df is not None and not scores_df.empty and "composite_score" in scores_df.columns:
                signals[ticker] = scores_df["composite_score"].iloc[-1]
        except:
            continue

    return pd.Series(signals)


def run_optimization(
    returns: pd.DataFrame,
    signals: pd.Series,
    variant: str,
    asset_class_map: dict,
    prefilter: int = None
) -> dict:
    """Run CVXPY optimization for a single variant."""
    start_time = time.time()

    optimizer = create_optimizer(
        variant=variant,
        max_positions=20,
        max_weight=0.15,
        min_weight=0.02,
        asset_class_map=asset_class_map,
        max_asset_class_weight=0.20,
        prefilter_top_n=prefilter,
        use_ledoit_wolf=True,
        solver="ECOS",
        solver_tolerance=1e-4
    )

    result = optimizer.optimize(returns, signals)
    elapsed = time.time() - start_time

    # Validate constraints
    constraints = PortfolioConstraints(
        max_positions=20,
        max_weight=0.15,
        min_weight=0.02
    )
    is_valid, violations = constraints.validate_weights(result["weights"])

    # Risk constraints
    risk_constraints = RiskConstraints(max_cvar=0.20, max_drawdown=0.25)
    full_weights = result["full_weights"].reindex(returns.columns, fill_value=0).values
    risk_valid, risk_violations = risk_constraints.check_risk_constraints(full_weights, returns)

    return {
        "variant": variant,
        "result": result,
        "elapsed": elapsed,
        "constraints_valid": is_valid,
        "violations": violations,
        "risk_valid": risk_valid,
        "risk_violations": risk_violations
    }


def test_universe(
    universe_name: str,
    n_etfs: int,
    prefilter: int = None
) -> dict:
    """Test optimization on a specific universe size."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{universe_name.upper()} UNIVERSE TEST")
    logger.info(f"{'='*80}\n")

    # Load data
    logger.info(f"Loading {n_etfs}-ETF universe...")
    prices = load_etf_universe(n_etfs)
    logger.info(f"Loaded: {prices.shape[1]} ETFs after filtering")

    asset_class_map = load_asset_class_map()

    returns = prices.pct_change().dropna()
    signals = generate_signals(prices)

    # Align data
    common_tickers = returns.columns.intersection(signals.index)
    returns = returns[common_tickers]
    signals = signals[common_tickers]
    logger.info(f"Using {len(common_tickers)} ETFs with signals")

    # Run all variants
    variants = ["max_sharpe", "balanced", "min_drawdown"]
    results = []

    total_start = time.time()
    for variant in variants:
        logger.info(f"\nRunning {variant}...")
        result = run_optimization(returns, signals, variant, asset_class_map, prefilter)
        results.append(result)

        logger.info(
            f"  {variant}: {result['result']['metrics']['num_positions']} positions, "
            f"Sharpe={result['result']['metrics']['sharpe_ratio']:.2f}, "
            f"Time={result['elapsed']:.2f}s"
        )

    total_time = time.time() - total_start

    return {
        "universe_name": universe_name,
        "universe_size": n_etfs,
        "filtered_size": len(common_tickers),
        "prefilter": prefilter,
        "results": results,
        "total_time": total_time
    }


def print_summary(all_results: list):
    """Print summary of all test results."""
    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE TEST SUITE SUMMARY")
    logger.info(f"{'='*80}\n")

    for test in all_results:
        logger.info(f"\n{test['universe_name']} ({test['filtered_size']} ETFs):")
        logger.info(f"{'='*60}")

        print(f"{'Variant':<15} {'Pos':>5} {'Sharpe':>8} {'Return':>8} {'Vol':>8} {'MaxWt':>8} {'Time':>8} {'Valid':>6}")
        print("-" * 75)

        for res in test['results']:
            m = res['result']['metrics']
            valid = "✅" if res['constraints_valid'] and res['risk_valid'] else "⚠️"
            print(
                f"{res['variant']:<15} "
                f"{m['num_positions']:>5d} "
                f"{m['sharpe_ratio']:>8.2f} "
                f"{m['expected_return']*100:>7.2f}% "
                f"{m['volatility']*100:>7.2f}% "
                f"{m['max_weight']*100:>7.1f}% "
                f"{res['elapsed']:>7.2f}s "
                f"{valid:>6}"
            )

        logger.info(f"\nTotal time: {test['total_time']:.2f}s")


def save_results(all_results: list):
    """Save comprehensive results to JSON."""
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cvxpy_complete_suite_{timestamp}.json"

    json_results = {
        "test_date": timestamp,
        "optimizer": "CVXPY/ECOS",
        "solver_tolerance": 1e-4,
        "calibrated_penalties": True,
        "tests": []
    }

    for test in all_results:
        test_data = {
            "universe": test['universe_name'],
            "size": test['universe_size'],
            "filtered_size": test['filtered_size'],
            "prefilter": test['prefilter'],
            "total_time": test['total_time'],
            "variants": []
        }

        for res in test['results']:
            test_data["variants"].append({
                "variant": res['variant'],
                "metrics": res['result']['metrics'],
                "holdings": res['result']['weights'].to_dict(),
                "elapsed": res['elapsed'],
                "constraints_valid": res['constraints_valid'],
                "risk_valid": res['risk_valid']
            })

        json_results["tests"].append(test_data)

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    return output_file


def main():
    """Run complete test suite."""
    logger.info("="*80)
    logger.info("CVXPY COMPLETE OPTIMIZATION TEST SUITE")
    logger.info("Testing: 100-ETF, 300-ETF, 753-ETF")
    logger.info("="*80)

    all_results = []

    # Test 1: 100-ETF Pilot
    test1 = test_universe("100-ETF Pilot", 100, prefilter=None)
    all_results.append(test1)

    # Test 2: 300-ETF Medium
    test2 = test_universe("300-ETF Medium", 300, prefilter=None)
    all_results.append(test2)

    # Test 3: 753-ETF Full
    test3 = test_universe("753-ETF Full", 753, prefilter=400)
    all_results.append(test3)

    # Print summary
    print_summary(all_results)

    # Save results
    output_file = save_results(all_results)

    logger.info("\n✅ COMPLETE TEST SUITE FINISHED")
    logger.info(f"Results: {output_file}")


if __name__ == "__main__":
    main()
