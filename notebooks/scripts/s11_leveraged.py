"""
Step 11: Leveraged ETF Momentum Strategy — Backtest & Analysis

Runs the leveraged ETF strategy across multiple pair/config combinations.
Supports both quick single-pair tests and full grid backtests.

Usage (from notebook):
    from scripts.s11_leveraged import run_leveraged_backtest
    results = run_leveraged_backtest(DATA_DIR, OUTPUT_DIR)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure src/ is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


def run_leveraged_backtest(
    data_dir: Path,
    output_dir: Path,
    pairs: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    initial_capital: float = 1_000_000,
    save_results: bool = True,
) -> Dict:
    """Run leveraged ETF backtests and return results.

    Args:
        data_dir: Directory containing {TICKER}.parquet files (IB historical).
        output_dir: Directory for backtest output files.
        pairs: List of pair keys to test (e.g. ["QLD_TLT", "SSO_TLT"]).
            Default: all pairs with available data.
        configs: List of config keys (e.g. ["A", "D"]). Default: all.
        initial_capital: Starting capital for each backtest.
        save_results: Whether to save summary and histories to parquet.

    Returns:
        Dict with keys: summary (DataFrame), histories (dict of DataFrames),
        trades (dict of DataFrames).
    """
    from leveraged.backtest import CONFIGS, run_single_backtest, compute_metrics
    from leveraged.universe import PAIRS, check_data_availability, load_pair_data

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data availability
    print("=== Data Availability ===")
    avail = check_data_availability(data_dir)
    print(avail.to_string(index=False))
    print()

    # Filter to available pairs
    available_pairs = {}
    for key, pair in PAIRS.items():
        if pairs is not None and key not in pairs:
            continue
        prices = load_pair_data(pair, data_dir)
        if prices is not None:
            available_pairs[key] = pair
            print(f"  {pair.name}: {len(prices)} days ready")
        else:
            print(f"  {pair.name}: SKIPPED (missing data)")

    if not available_pairs:
        print("\nNo pairs have complete data. Collect missing tickers first.")
        return {"summary": pd.DataFrame(), "histories": {}, "trades": {}}

    if configs is None:
        configs = list(CONFIGS.keys())

    # Run backtests
    print(f"\n=== Running {len(available_pairs)} pairs x {len(configs)} configs ===\n")

    results = []
    histories = {}
    all_trades = {}

    for pair_key, pair in available_pairs.items():
        for config_key in configs:
            result = run_single_backtest(pair, data_dir, config_key, initial_capital)
            if result is None:
                continue

            m = result["metrics"]
            label = f"{pair_key}_{config_key}"

            results.append({
                "pair": pair.name,
                "pair_key": pair_key,
                "config": config_key,
                "config_name": CONFIGS[config_key]["name"],
                "cagr": m["cagr"],
                "total_return": m["total_return"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"],
                "calmar": m["calmar"],
                "volatility": m["volatility"],
                "time_in_market": m["time_in_market"],
                "final_value": m["final_value"],
                "cash_reserve": m["cash_reserve"],
                "years": m["years"],
                "n_trades": len(result["trades"]),
            })

            histories[label] = result["history"]
            all_trades[label] = result["trades"]

            print(
                f"  {pair.name} / {config_key} ({CONFIGS[config_key]['name']}): "
                f"CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, "
                f"MaxDD={m['max_drawdown']:.1%}, Final=${m['final_value']:,.0f}"
            )

    summary = pd.DataFrame(results)

    # Save outputs
    if save_results and not summary.empty:
        summary_path = output_dir / "backtest_results.parquet"
        summary.to_parquet(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")

        for label, hist in histories.items():
            hist_path = output_dir / f"history_{label}.parquet"
            hist.to_parquet(hist_path)

        print(f"Histories saved: {len(histories)} files")

    # Print summary table
    if not summary.empty:
        print("\n=== Summary ===")
        display_cols = [
            "pair", "config", "cagr", "sharpe", "sortino",
            "max_drawdown", "calmar", "time_in_market", "final_value", "cash_reserve",
        ]
        fmt = {
            "cagr": "{:.1%}".format, "sharpe": "{:.2f}".format,
            "sortino": "{:.2f}".format, "max_drawdown": "{:.1%}".format,
            "calmar": "{:.2f}".format, "time_in_market": "{:.0%}".format,
            "final_value": "${:,.0f}".format, "cash_reserve": "${:,.0f}".format,
        }
        print(summary[display_cols].to_string(index=False, formatters=fmt))

    return {"summary": summary, "histories": histories, "trades": all_trades}
