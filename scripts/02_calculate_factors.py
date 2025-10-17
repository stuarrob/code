"""
Calculate Factor Scores for ETF Universe

This script calculates all factor scores (momentum, quality, value, volatility)
for the entire ETF universe and saves results for analysis.

Filters:
- Remove leveraged ETFs (2x, 3x, inverse)
- Remove high volatility ETFs (>35% annual vol)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.factors import (
    MomentumFactor,
    QualityFactor,
    SimplifiedValueFactor,
    VolatilityFactor,
    FactorIntegrator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_etf_data():
    """Load ETF price data."""
    data_file = project_root / "data" / "processed" / "etf_prices_filtered.parquet"

    if not data_file.exists():
        raise FileNotFoundError(
            f"ETF data not found at {data_file}. "
            f"Run scripts/01_collect_universe.py first."
        )

    logger.info(f"Loading ETF data from {data_file}")
    df = pd.read_parquet(data_file)

    logger.info(f"Loaded {len(df.columns)} ETFs with {len(df)} days of data")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def create_expense_ratios(tickers):
    """
    Create mock expense ratios for ETFs.

    In production, would fetch from ETF provider APIs.
    For now, use reasonable defaults based on ETF type.
    """
    expense_ratios = {}

    for ticker in tickers:
        # Heuristic expense ratios
        if ticker.startswith('V'):  # Vanguard
            er = np.random.uniform(0.03, 0.10) / 100
        elif ticker.startswith('I'):  # iShares
            er = np.random.uniform(0.05, 0.20) / 100
        elif ticker.startswith('SP'):  # SPDR
            er = np.random.uniform(0.05, 0.25) / 100
        elif 'Q' in ticker:  # Tech/QQQ style
            er = np.random.uniform(0.15, 0.30) / 100
        else:
            er = np.random.uniform(0.10, 0.50) / 100

        expense_ratios[ticker] = er

    return pd.Series(expense_ratios)


def filter_etfs(prices: pd.DataFrame,
                max_volatility: float = 0.35,
                min_history: int = 252) -> pd.DataFrame:
    """
    Filter ETF universe.

    Removes:
    - Leveraged ETFs (2x, 3x, inverse)
    - High volatility ETFs (>35% annual)
    - ETFs with insufficient history
    """
    logger.info(f"Filtering ETFs: max_vol={max_volatility:.0%}, min_history={min_history}d")

    initial_count = len(prices.columns)

    # Filter 1: Sufficient history
    valid_history = prices.count() >= min_history
    prices = prices.loc[:, valid_history]
    logger.info(f"After history filter: {len(prices.columns)}/{initial_count} ETFs")

    # Filter 2: Remove leveraged ETFs
    leveraged_keywords = ['2X', '3X', 'ULTRA', 'INVERSE', 'SHORT', 'BEAR', 'BULL']
    leveraged_etfs = []

    for ticker in prices.columns:
        if any(keyword in ticker.upper() for keyword in leveraged_keywords):
            leveraged_etfs.append(ticker)

    prices = prices.drop(columns=leveraged_etfs, errors='ignore')
    logger.info(f"Removed {len(leveraged_etfs)} leveraged ETFs")
    logger.info(f"After leverage filter: {len(prices.columns)}/{initial_count} ETFs")

    # Filter 3: Remove high volatility ETFs
    returns = prices.pct_change().dropna()
    annual_vol = returns.std() * np.sqrt(252)

    high_vol_etfs = annual_vol[annual_vol > max_volatility].index.tolist()
    prices = prices.drop(columns=high_vol_etfs, errors='ignore')

    logger.info(f"Removed {len(high_vol_etfs)} high-volatility ETFs (>{max_volatility:.0%})")
    logger.info(f"After volatility filter: {len(prices.columns)}/{initial_count} ETFs")

    return prices


def calculate_all_factors(prices: pd.DataFrame,
                          expense_ratios: pd.Series) -> pd.DataFrame:
    """Calculate all factor scores."""
    logger.info("Calculating factors...")

    # Initialize factors
    momentum = MomentumFactor(lookback=252, skip_recent=21)
    quality = QualityFactor(lookback=252)
    value = SimplifiedValueFactor()
    volatility = VolatilityFactor(lookback=60)

    # Calculate scores
    logger.info("  Calculating momentum...")
    momentum_scores = momentum.calculate(prices)

    logger.info("  Calculating quality...")
    quality_scores = quality.calculate(prices)

    logger.info("  Calculating value...")
    value_scores = value.calculate(prices, expense_ratios)

    logger.info("  Calculating volatility...")
    volatility_scores = volatility.calculate(prices)

    # Combine
    factor_scores = pd.DataFrame({
        'momentum': momentum_scores,
        'quality': quality_scores,
        'value': value_scores,
        'low_volatility': volatility_scores
    })

    factor_scores = factor_scores.dropna()
    logger.info(f"Factor calculation complete: {len(factor_scores)} ETFs")

    return factor_scores


def integrate_factors(factor_scores: pd.DataFrame) -> pd.Series:
    """Integrate factors using geometric mean."""
    logger.info("Integrating factors...")

    integrator = FactorIntegrator({
        'momentum': 0.35,
        'quality': 0.30,
        'value': 0.15,
        'low_volatility': 0.20
    })

    integrated_scores = integrator.integrate(factor_scores)
    return integrated_scores


def main():
    """Main execution."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("="*60)
    logger.info("FACTOR CALCULATION")
    logger.info("="*60)

    prices = load_etf_data()
    filtered_prices = filter_etfs(prices, max_volatility=0.35, min_history=300)
    expense_ratios = create_expense_ratios(filtered_prices.columns)
    factor_scores = calculate_all_factors(filtered_prices, expense_ratios)
    integrated_scores = integrate_factors(factor_scores)

    factor_scores['integrated'] = integrated_scores
    factor_scores = factor_scores.sort_values('integrated', ascending=False)

    # Save
    output_file = project_root / "data" / f"factor_scores_{timestamp}.parquet"
    factor_scores.to_parquet(output_file)

    latest_file = project_root / "data" / "factor_scores_latest.parquet"
    factor_scores.to_parquet(latest_file)

    logger.info(f"\nSaved to: {latest_file}")
    logger.info(f"\nTop 20 ETFs:")
    logger.info(factor_scores.head(20))

    return factor_scores


if __name__ == '__main__':
    factor_scores = main()
