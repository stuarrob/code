"""
Validate Real ETF Data

Load real ETF data, examine data quality, and prepare for backtesting.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_etf_universe():
    """Load the ETF universe metadata."""
    universe_file = project_root / 'data' / 'raw' / 'etf_universe.csv'

    df = pd.read_csv(universe_file)
    logger.info(f"Loaded universe: {len(df)} ETFs")

    # Count successful downloads
    successful = df[df['success'] == True]
    logger.info(f"  Successful downloads: {len(successful)}")
    logger.info(f"  Failed downloads: {len(df) - len(successful)}")

    return df


def load_all_prices():
    """Load all ETF price data."""
    prices_dir = project_root / 'data' / 'raw' / 'prices'

    all_prices = {}
    failed = []

    logger.info("\nLoading price data...")

    price_files = list(prices_dir.glob('*.csv'))
    logger.info(f"Found {len(price_files)} price files")

    for i, price_file in enumerate(price_files):
        ticker = price_file.stem

        try:
            df = pd.read_csv(price_file, index_col=0, parse_dates=True)

            # Use Adj Close if available, else Close
            if 'Adj Close' in df.columns:
                all_prices[ticker] = df['Adj Close']
            elif 'Close' in df.columns:
                all_prices[ticker] = df['Close']
            else:
                logger.warning(f"  {ticker}: No Close/Adj Close column")
                failed.append(ticker)

        except Exception as e:
            logger.warning(f"  {ticker}: Failed to load - {e}")
            failed.append(ticker)

        if (i + 1) % 100 == 0:
            logger.info(f"  Loaded {i+1}/{len(price_files)}...")

    # Combine into DataFrame
    prices_df = pd.DataFrame(all_prices)

    logger.info(f"\n✓ Loaded {len(prices_df.columns)} ETFs")
    logger.info(f"  Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    logger.info(f"  Days: {len(prices_df)}")
    logger.info(f"  Failed: {len(failed)}")

    return prices_df, failed


def analyze_data_quality(prices):
    """Analyze data quality."""
    logger.info("\n" + "="*80)
    logger.info("DATA QUALITY ANALYSIS")
    logger.info("="*80)

    # Missing data
    missing_pct = (prices.isna().sum() / len(prices)) * 100

    logger.info(f"\nMissing Data:")
    logger.info(f"  ETFs with no missing data: {(missing_pct == 0).sum()}")
    logger.info(f"  ETFs with <5% missing: {(missing_pct < 5).sum()}")
    logger.info(f"  ETFs with >20% missing: {(missing_pct > 20).sum()}")

    if (missing_pct > 20).any():
        logger.info(f"\n  High missing data ETFs:")
        high_missing = missing_pct[missing_pct > 20].sort_values(ascending=False)
        for ticker, pct in high_missing.head(10).items():
            logger.info(f"    {ticker}: {pct:.1f}%")

    # Data coverage by period
    logger.info(f"\nData Coverage by Period:")
    periods = [
        ('2020-10-01', '2021-12-31', '2020-2021'),
        ('2022-01-01', '2022-12-31', '2022'),
        ('2023-01-01', '2023-12-31', '2023'),
        ('2024-01-01', '2024-12-31', '2024'),
        ('2025-01-01', '2025-10-03', '2025')
    ]

    for start, end, label in periods:
        try:
            period_data = prices.loc[start:end]
            complete = period_data.notna().all()
            logger.info(f"  {label}: {complete.sum()} ETFs with complete data")
        except:
            logger.info(f"  {label}: No data in this range")

    # Price ranges (sanity check)
    logger.info(f"\nPrice Ranges (sanity check):")
    current_prices = prices.iloc[-1].dropna()
    logger.info(f"  Min price: ${current_prices.min():.2f}")
    logger.info(f"  Median price: ${current_prices.median():.2f}")
    logger.info(f"  Max price: ${current_prices.max():.2f}")

    # Extremely low prices (potential data issues)
    very_low = current_prices[current_prices < 5]
    if len(very_low) > 0:
        logger.info(f"\n  ETFs with price < $5 (may have data issues):")
        for ticker, price in very_low.head(10).items():
            logger.info(f"    {ticker}: ${price:.2f}")

    # Volatility sanity check
    returns = prices.pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized

    logger.info(f"\nVolatility (annualized):")
    logger.info(f"  Median: {volatility.median():.1%}")
    logger.info(f"  95th percentile: {volatility.quantile(0.95):.1%}")

    # Extremely high volatility
    high_vol = volatility[volatility > 0.50]  # >50% annual vol
    if len(high_vol) > 0:
        logger.info(f"\n  High volatility ETFs (>50% annual):")
        for ticker, vol in high_vol.sort_values(ascending=False).head(10).items():
            logger.info(f"    {ticker}: {vol:.1%}")

    return {
        'missing_pct': missing_pct,
        'volatility': volatility,
        'current_prices': current_prices
    }


def filter_for_backtesting(prices, quality_stats, min_price=10, max_vol=0.35, max_missing=0.10):
    """
    Filter ETFs suitable for backtesting.

    Args:
        min_price: Minimum current price (default: $10)
        max_vol: Maximum annualized volatility (default: 35%)
        max_missing: Maximum missing data percentage (default: 10%)
    """
    logger.info("\n" + "="*80)
    logger.info("FILTERING FOR BACKTEST")
    logger.info("="*80)

    logger.info(f"\nCriteria:")
    logger.info(f"  Min price: ${min_price}")
    logger.info(f"  Max volatility: {max_vol:.0%}")
    logger.info(f"  Max missing data: {max_missing:.0%}")

    # Get quality metrics
    missing_pct = quality_stats['missing_pct'] / 100  # Convert to fraction
    volatility = quality_stats['volatility']
    current_prices = quality_stats['current_prices']

    # Start with all tickers
    eligible = set(prices.columns)
    initial_count = len(eligible)

    # Filter 1: Price
    low_price = set(current_prices[current_prices < min_price].index)
    eligible -= low_price
    logger.info(f"\n  Remove low price (<${min_price}): {len(low_price)} removed, {len(eligible)} remain")

    # Filter 2: Volatility
    high_vol = set(volatility[volatility > max_vol].index)
    eligible -= high_vol
    logger.info(f"  Remove high vol (>{max_vol:.0%}): {len(high_vol)} removed, {len(eligible)} remain")

    # Filter 3: Missing data
    high_missing = set(missing_pct[missing_pct > max_missing].index)
    eligible -= high_missing
    logger.info(f"  Remove high missing (>{max_missing:.0%}): {len(high_missing)} removed, {len(eligible)} remain")

    # Filter 4: Need at least 252 days of data for momentum
    insufficient_data = []
    for ticker in list(eligible):
        if prices[ticker].notna().sum() < 252:
            insufficient_data.append(ticker)
    eligible -= set(insufficient_data)
    logger.info(f"  Remove insufficient data (<252 days): {len(insufficient_data)} removed, {len(eligible)} remain")

    logger.info(f"\n✓ Final: {len(eligible)} ETFs eligible ({len(eligible)/initial_count:.1%} of original)")

    # Filter prices DataFrame
    filtered_prices = prices[list(eligible)].dropna(how='all')

    return filtered_prices, list(eligible)


def save_filtered_data(prices, eligible_tickers):
    """Save filtered data for backtesting."""
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)

    # Save prices
    prices_file = output_dir / 'etf_prices_filtered.parquet'
    prices.to_parquet(prices_file)
    logger.info(f"\n✓ Saved filtered prices: {prices_file}")
    logger.info(f"  Shape: {prices.shape} (days × ETFs)")

    # Save ticker list
    tickers_file = output_dir / 'eligible_tickers.txt'
    with open(tickers_file, 'w') as f:
        for ticker in sorted(eligible_tickers):
            f.write(f"{ticker}\n")
    logger.info(f"✓ Saved ticker list: {tickers_file}")

    return prices_file, tickers_file


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("REAL DATA VALIDATION")
    logger.info("="*80)

    # Load universe
    universe = load_etf_universe()

    # Load all prices
    prices, failed = load_all_prices()

    # Analyze quality
    quality_stats = analyze_data_quality(prices)

    # Filter for backtesting
    filtered_prices, eligible_tickers = filter_for_backtesting(prices, quality_stats)

    # Save filtered data
    prices_file, tickers_file = save_filtered_data(filtered_prices, eligible_tickers)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Original ETFs:     {len(universe)}")
    logger.info(f"With price data:   {len(prices.columns)}")
    logger.info(f"Failed to load:    {len(failed)}")
    logger.info(f"Eligible for BT:   {len(eligible_tickers)}")
    logger.info(f"Date range:        {filtered_prices.index[0].date()} to {filtered_prices.index[-1].date()}")
    logger.info(f"Trading days:      {len(filtered_prices)}")
    logger.info("="*80)

    logger.info(f"\n✓ Ready for backtesting!")
    logger.info(f"  Prices: {prices_file}")
    logger.info(f"  Tickers: {tickers_file}")


if __name__ == '__main__':
    main()
