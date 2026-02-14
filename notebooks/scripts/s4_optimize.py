"""
Step 4: Portfolio Construction & Optimization

Builds target portfolio from factor scores using configurable optimizer.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def build_portfolio(
    combined_scores: pd.Series,
    prices: pd.DataFrame,
    num_positions: int = 20,
    optimizer_type: str = "rankbased",
    categories: dict = None,
    output_file: Path = None,
) -> pd.Series:
    """Construct target portfolio from factor scores.

    Args:
        combined_scores: Integrated factor scores (sorted descending).
        prices: Price DataFrame (basic model, no leveraged).
        num_positions: Number of positions.
        optimizer_type: "rankbased" | "mvo" | "minvar" | "simple"
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

    return target_weights
