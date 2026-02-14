"""
Step 3: Factor Scoring

4-factor model: Momentum (35%), Quality (30%), Value (15%), Volatility (20%).
Leveraged/inverse ETFs are filtered out before scoring (basic model only).
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
        (combined_scores, prices_basic)
        - combined_scores: Series of integrated scores, sorted descending
        - prices_basic: prices DataFrame with leveraged ETFs removed
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
        factor_weights = DEFAULT_WEIGHTS

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
    if raw_dir and (raw_dir / "fundamentals.csv").exists():
        fund_df = pd.read_csv(raw_dir / "fundamentals.csv")
        expense_ratios = fund_df.set_index("ticker")["expense_ratio"].dropna()
        expense_ratios = expense_ratios.reindex(prices_basic.columns).fillna(
            expense_ratios.median()
        )
    else:
        expense_ratios = pd.Series(
            np.random.uniform(0.0005, 0.01, len(prices_basic.columns)),
            index=prices_basic.columns,
        )

    value_scores = SimplifiedValueFactor().calculate(prices_basic, expense_ratios)
    print(f"  Value:      {value_scores.dropna().shape[0]} tickers")

    volatility_scores = VolatilityFactor(lookback=60).calculate(prices_basic)
    print(f"  Volatility: {volatility_scores.dropna().shape[0]} tickers")

    # Integrate
    factor_df = pd.DataFrame({
        "momentum": momentum_scores,
        "quality": quality_scores,
        "value": value_scores,
        "volatility": volatility_scores,
    })

    integrator = FactorIntegrator(factor_weights=factor_weights)
    combined_scores = integrator.integrate(factor_df)
    print(f"\nIntegrated scores: {len(combined_scores)} tickers")

    return combined_scores, prices_basic
