"""
Step 5: Backtesting & Performance Analysis

Runs backtest with configurable rebalance frequency.
Default: bimonthly (6x/year) to avoid excessive trading.

NOTE: The backtest is a HISTORICAL SIMULATION only.  Stop-loss messages
show what would have happened in the past (e.g. during the 2022 drawdown).
They do NOT affect the current target portfolio — that is built from
today's factor scores in Step 4.
"""

import pandas as pd


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
    from backtesting import (
        BacktestConfig, BacktestEngine, TransactionCostModel,
    )
    from portfolio import (
        RankBasedOptimizer, StopLossManager, ThresholdRebalancer,
    )

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

    print("Running backtest (HISTORICAL SIMULATION)...")
    d0 = prices.index[0].date()
    d1 = prices.index[-1].date()
    print(f"  Period:     {d0} to {d1}")
    print(f"  Rebalance:  {rebalance_frequency}")
    print(f"  Stop-loss:  {stop_loss_pct:.0%}")
    print(f"  Drift:      {drift_threshold:.0%}")
    print(f"  Capital:    ${initial_capital:,.0f}")
    print()

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

    # Count stop-loss events
    trades_df = results.get("trades")
    num_stops = 0
    if trades_df is not None and len(trades_df) > 0:
        if isinstance(trades_df, pd.DataFrame):
            if "action" in trades_df.columns:
                sl = trades_df["action"] == "STOP_LOSS"
                num_stops = int(sl.sum())
        elif isinstance(trades_df, list):
            num_stops = sum(
                1 for t in trades_df
                if t.get("action") == "STOP_LOSS"
            )

    print(f"\n{'─' * 50}")
    print("BACKTEST RESULTS (historical simulation)")
    print(f"{'─' * 50}")
    print(f"  CAGR:              {metrics.get('cagr', 0):.1%}")
    print(f"  Sharpe:            {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino:           {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):.1%}")
    print(f"  Volatility:        {metrics.get('volatility', 0):.1%}")
    print(f"  Total Return:      {metrics.get('total_return', 0):.1%}")
    reb_yr = rebalances_per_year
    print(f"  Rebalances:        {num_rebalances} ({reb_yr:.1f}/year)")
    print(f"  Win Rate:          {metrics.get('win_rate', 0):.0%}")
    if num_stops > 0:
        msg = f"  Stop-loss events:  {num_stops} (historical)"
        print(f"\n{msg}")
    print("─" * 50)

    if num_stops > 0:
        n = num_stops
        print(f"\n  NOTE: {n} stop-loss events occurred DURING")
        print("  the backtest (mostly the 2022 drawdown).")
        print("  They show how the strategy protects capital.")
        print("  They do NOT affect today's target portfolio.")

    if rebalances_per_year > 6:
        print(
            f"\n  WARNING: {reb_yr:.1f} rebalances/year "
            f"exceeds target of 6. Consider 'quarterly'."
        )

    return {
        "metrics": metrics,
        "daily_values": daily_values,
        "results": results,
        "rebalances_per_year": rebalances_per_year,
    }
