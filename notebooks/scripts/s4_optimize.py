"""
Step 4: Portfolio Construction & Optimization

Builds target portfolio from factor scores using configurable optimizer.
Shows human-readable ETF names and factor score breakdown for each holding.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def build_portfolio(
    combined_scores: pd.Series,
    prices: pd.DataFrame,
    num_positions: int = 20,
    optimizer_type: str = "rankbased",
    factor_detail: pd.DataFrame = None,
    categories: dict = None,
    output_file: Path = None,
) -> pd.Series:
    """Construct target portfolio from factor scores.

    Args:
        combined_scores: Integrated factor scores (sorted descending).
        prices: Price DataFrame (basic model, no leveraged).
        num_positions: Number of positions.
        optimizer_type: "rankbased" | "mvo" | "minvar" | "simple"
        factor_detail: Optional DataFrame of individual factor scores per ticker.
        categories: Optional ticker->category mapping for display.
        output_file: Path to save target portfolio CSV.

    Returns:
        target_weights: Series of ticker -> weight.
    """
    from portfolio import (
        MeanVarianceOptimizer,
        MinVarianceOptimizer,
        RankBasedOptimizer,
        SimpleOptimizer,
    )

    print(f"Optimizer: {optimizer_type.upper()}, Positions: {num_positions}")

    if optimizer_type == "mvo":
        optimizer = MeanVarianceOptimizer(
            num_positions=num_positions, lookback=60,
            risk_aversion=1.0, use_factor_scores_as_alpha=True,
            min_weight=0.03, max_weight=0.08,
        )
        target_weights = optimizer.optimize(combined_scores, prices)
    elif optimizer_type == "rankbased":
        optimizer = RankBasedOptimizer(
            num_positions=num_positions, weighting_scheme="exponential",
        )
        target_weights = optimizer.optimize(combined_scores)
    elif optimizer_type == "minvar":
        optimizer = MinVarianceOptimizer(num_positions=num_positions, lookback=60)
        target_weights = optimizer.optimize(combined_scores, prices)
    else:
        optimizer = SimpleOptimizer(num_positions=num_positions)
        target_weights = optimizer.optimize(combined_scores)

    # Save
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        target_weights.to_csv(output_file, header=True)
        print(f"Saved: {output_file}")

    # Summary
    print(f"\nPortfolio: {len(target_weights)} positions")
    print(f"Max weight: {target_weights.max():.1%}")
    print(f"Min weight: {target_weights.min():.1%}")
    print(f"HHI: {(target_weights**2).sum():.4f}")

    # Expected volatility
    if prices is not None and len(prices) > 60:
        available = [t for t in target_weights.index if t in prices.columns]
        if len(available) == len(target_weights):
            ret = prices[available].pct_change().dropna()
            cov = ret.cov() * 252
            w = target_weights[available].values
            port_vol = np.sqrt(w @ cov.values @ w)
            print(f"Expected vol: {port_vol:.1%}")

    # Display holdings with names and factor breakdown
    _print_holdings_detail(target_weights, factor_detail)

    return target_weights


def _print_holdings_detail(
    weights: pd.Series,
    factor_detail: pd.DataFrame = None,
) -> None:
    """Print a human-readable portfolio table with ETF names and factor scores."""
    try:
        from utils.etf_names import lookup_names
    except ImportError:
        return

    tickers = weights.index.tolist()
    names = lookup_names(tickers, use_yfinance=True)

    # Build percentile ranks for available factors
    ranks = None
    cols = []
    if factor_detail is not None:
        cols = list(factor_detail.columns)
        ranks = pd.DataFrame()
        for c in cols:
            ranks[c] = factor_detail[c].rank(pct=True)

    print(f"\n{'─' * 90}")
    header = f"{'#':>2}  {'Ticker':<6}  {'Weight':>6}  {'Name':<42}"
    if cols:
        header += "  " + "  ".join(f"{c[:5].title():>5}" for c in cols)
    print(header)
    print(f"{'─' * 90}")

    for i, t in enumerate(tickers, 1):
        name = names.get(t, t)
        if len(name) > 42:
            name = name[:39] + "..."
        line = f"{i:>2}  {t:<6}  {weights[t]:>5.1%}  {name:<42}"

        if ranks is not None and t in ranks.index:
            for c in cols:
                val = ranks.loc[t, c]
                if pd.notna(val):
                    line += f"  {val:>4.0%} "
                else:
                    line += f"  {'—':>5}"
        print(line)

    print(f"{'─' * 90}")
    if cols:
        print("Factor columns show percentile rank vs full universe (higher = better)")
