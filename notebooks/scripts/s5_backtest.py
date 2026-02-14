"""
Step 5: Backtesting & Performance Analysis

Runs backtest with configurable rebalance frequency.
Default: bimonthly (6x/year) to avoid excessive trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def run_backtest(
    prices: pd.DataFrame,
    combined_scores: pd.Series,
    num_positions: int = 20,
    rebalance_frequency: str = "bimonthly",
    stop_loss_pct: float = 0.12,
    drift_threshold: float = 0.05,
    initial_capital: float = 1_000_000,
    risk_free_rate: float = 0.04,
) -> dict:
    """Run backtest on historical prices.

    Args:
        prices: Price DataFrame (basic model, no leveraged).
        combined_scores: Integrated factor scores.
        rebalance_frequency: "bimonthly" (6/yr), "monthly" (12/yr),
                             "quarterly" (4/yr), "weekly" (52/yr).
        stop_loss_pct: Stop-loss threshold (default 12%).
        drift_threshold: Drift threshold for rebalancing.

    Returns:
        dict with keys: metrics, daily_values, results (full engine output)
    """
    from backtesting import BacktestConfig, BacktestEngine, TransactionCostModel
    from portfolio import RankBasedOptimizer, StopLossManager, ThresholdRebalancer

    config = BacktestConfig(
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
        num_positions=num_positions,
        stop_loss_pct=stop_loss_pct,
        use_stop_loss=True,
        risk_free_rate=risk_free_rate,
    )

    cost_model = TransactionCostModel(
        commission_per_trade=0.0,
        spread_bps=2.0,
        slippage_bps=2.0,
    )

    # Build factor scores across all dates (static scores)
    factor_scores_bt = pd.DataFrame(
        {col: combined_scores for col in prices.index}
    ).T
    factor_scores_bt.index = prices.index

    optimizer = RankBasedOptimizer(
        num_positions=num_positions, weighting_scheme="exponential",
    )
    rebalancer = ThresholdRebalancer(drift_threshold=drift_threshold)
    risk_manager = StopLossManager(position_stop_loss=stop_loss_pct)

    print(f"Running backtest...")
    print(f"  Rebalance frequency: {rebalance_frequency}")
    print(f"  Stop-loss: {stop_loss_pct:.0%}")
    print(f"  Drift threshold: {drift_threshold:.0%}")
    print(f"  Capital: ${initial_capital:,.0f}")

    results = BacktestEngine(config, cost_model).run(
        prices=prices,
        factor_scores=factor_scores_bt,
        optimizer=optimizer,
        rebalancer=rebalancer,
        risk_manager=risk_manager,
    )

    metrics = results["metrics"]
    daily_values = results["daily_values"]

    # Calculate rebalances per year
    trading_days = len(prices)
    years = trading_days / 252
    num_rebalances = metrics.get("num_rebalances", 0)
    rebalances_per_year = num_rebalances / years if years > 0 else 0

    print(f"\nResults:")
    print(f"  CAGR:              {metrics.get('cagr', 0):.1%}")
    print(f"  Sharpe:            {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino:           {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):.1%}")
    print(f"  Volatility:        {metrics.get('volatility', 0):.1%}")
    print(f"  Total Return:      {metrics.get('total_return', 0):.1%}")
    print(f"  Rebalances:        {num_rebalances} ({rebalances_per_year:.1f}/year)")
    print(f"  Win Rate:          {metrics.get('win_rate', 0):.0%}")

    if rebalances_per_year > 6:
        print(f"\n  WARNING: {rebalances_per_year:.1f} rebalances/year exceeds "
              f"target of 6. Consider using 'quarterly' frequency.")

    return {
        "metrics": metrics,
        "daily_values": daily_values,
        "results": results,
        "rebalances_per_year": rebalances_per_year,
    }
