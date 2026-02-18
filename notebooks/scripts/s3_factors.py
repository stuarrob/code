"""
Step 3: Factor Scoring

Multi-factor model: Momentum, Quality, Value (if available), Volatility.
Leveraged/inverse ETFs are filtered out before scoring (basic model only).

If expense ratio data is unavailable, the value factor is skipped and its
weight is redistributed proportionally to the remaining factors.
"""

import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_WEIGHTS = {
    "momentum": 0.35,
    "quality": 0.30,
    "value": 0.15,
    "volatility": 0.20,
}


def score_factors(
    prices: pd.DataFrame,
    factor_weights: dict = None,
    categories: dict = None,
    raw_dir: Path = None,
) -> tuple:
    """Calculate integrated factor scores.

    Filters out leveraged/inverse ETFs before scoring.

    Returns:
        (combined_scores, prices_basic, factor_detail)
        - combined_scores: Series of integrated scores, sorted descending
        - prices_basic: prices DataFrame with leveraged ETFs removed
        - factor_detail: DataFrame of individual factor scores per ticker
    """
    from factors import (
        FactorIntegrator,
        MomentumFactor,
        QualityFactor,
        SimplifiedValueFactor,
        VolatilityFactor,
    )
    from data_collection.etf_filters import filter_leveraged_etfs

    if factor_weights is None:
        factor_weights = DEFAULT_WEIGHTS.copy()
    else:
        factor_weights = factor_weights.copy()

    # Filter leveraged ETFs for the basic model
    all_price_tickers = prices.columns.tolist()
    basic_tickers = filter_leveraged_etfs(all_price_tickers)
    prices_basic = prices[basic_tickers]
    excluded = len(all_price_tickers) - len(basic_tickers)
    print(f"Basic model: {len(basic_tickers)} tickers "
          f"({excluded} leveraged/inverse excluded)")

    # Calculate factors
    print("Calculating factors...")
    momentum_scores = MomentumFactor(lookback=252, skip_recent=21).calculate(prices_basic)
    print(f"  Momentum:   {momentum_scores.dropna().shape[0]} tickers")

    quality_scores = QualityFactor(lookback=252).calculate(prices_basic)
    print(f"  Quality:    {quality_scores.dropna().shape[0]} tickers")

    # Value factor — use real expense ratios if available
    value_scores = None
    expense_ratios = None
    if raw_dir and (raw_dir / "fundamentals.csv").exists():
        fund_df = pd.read_csv(raw_dir / "fundamentals.csv")
        valid_er = fund_df.set_index("ticker")["expense_ratio"].dropna()
        if len(valid_er) > 0:
            expense_ratios = valid_er.reindex(prices_basic.columns).fillna(
                valid_er.median()
            )

    if expense_ratios is not None and expense_ratios.notna().sum() > 0:
        value_scores = SimplifiedValueFactor().calculate(prices_basic, expense_ratios)
        print(f"  Value:      {value_scores.dropna().shape[0]} tickers")
    else:
        print("  Value:      SKIPPED (no expense ratio data available)")
        print("              Redistributing weight to remaining factors")
        # Remove value from weights and renormalise
        factor_weights.pop("value", None)
        total = sum(factor_weights.values())
        factor_weights = {k: v / total for k, v in factor_weights.items()}
        print(f"              Adjusted weights: "
              f"{', '.join(f'{k}={v:.0%}' for k, v in factor_weights.items())}")

    volatility_scores = VolatilityFactor(lookback=60).calculate(prices_basic)
    print(f"  Volatility: {volatility_scores.dropna().shape[0]} tickers")

    # Integrate — only include factors that produced scores
    factor_dict = {
        "momentum": momentum_scores,
        "quality": quality_scores,
        "volatility": volatility_scores,
    }
    if value_scores is not None:
        factor_dict["value"] = value_scores

    factor_df = pd.DataFrame(factor_dict)

    integrator = FactorIntegrator(factor_weights=factor_weights)
    combined_scores = integrator.integrate(factor_df)
    print(f"\nIntegrated scores: {len(combined_scores)} tickers")
    print(f"Active factors: {', '.join(factor_weights.keys())} "
          f"(weights: {', '.join(f'{v:.0%}' for v in factor_weights.values())})")

    return combined_scores, prices_basic, factor_df
