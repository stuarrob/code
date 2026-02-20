"""
Step 2: Historical Data Collection

On-demand update via Databento (fast, no IB connection needed).
Falls back to cached data if Databento is unavailable.
"""

import pandas as pd
from pathlib import Path


def collect_data(
    tickers: list,
    cache_dir: Path,
    processed_dir: Path,
    run_collection: bool = True,
) -> pd.DataFrame:
    """Collect/update historical price data via Databento.

    Returns wide DataFrame (dates x tickers) of close prices.
    """
    import sys
    sys.path.insert(0, str(cache_dir.parent.parent / "src"))

    prices_db_path = processed_dir / "etf_prices_db.parquet"
    prices_ib_path = processed_dir / "etf_prices_ib.parquet"
    prices_yf_path = processed_dir / "etf_prices_filtered.parquet"

    prices = None

    # Check for existing consolidated data
    if prices_db_path.exists():
        prices = pd.read_parquet(prices_db_path)
        print(f"Loaded prices: {prices.shape[1]} tickers x {prices.shape[0]} days")
        print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    elif prices_ib_path.exists():
        prices = pd.read_parquet(prices_ib_path)
        print(f"Loaded IB prices: {prices.shape[1]} tickers x {prices.shape[0]} days")
        print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    elif prices_yf_path.exists():
        prices = pd.read_parquet(prices_yf_path)
        print(f"Loaded yfinance prices: {prices.shape[1]} tickers x {prices.shape[0]} days")
    else:
        # Build from individual parquets if available
        cached = [f for f in cache_dir.glob("*.parquet") if f.stem != "manifest"]
        if len(cached) > 50:
            print(f"Building from {len(cached)} cached parquets...")
            from data_collection.databento_collector import DatabentoCollector
            collector = DatabentoCollector(cache_dir=str(cache_dir))
            prices = collector.build_price_matrix()

    if not run_collection:
        if prices is None:
            print("No cached prices and collection disabled.")
        return prices

    # Run on-demand update via Databento
    try:
        from data_collection.databento_collector import DatabentoCollector

        collector = DatabentoCollector(cache_dir=str(cache_dir))
        prices, results = collector.update_tickers(tickers)

    except Exception as e:
        print(f"Databento update failed: {e}")
        if prices is not None:
            print(f"Using existing cached prices ({prices.shape[1]} tickers).")

    return prices


def apply_quality_filter(prices: pd.DataFrame, min_days: int = 252) -> pd.DataFrame:
    """Filter tickers by data quality: require min_days history, <10% missing."""
    if prices is None:
        return None

    missing_pct = prices.isnull().sum() / len(prices) * 100
    adequate = (prices.count() >= min_days) & (missing_pct < 10)
    before = len(prices.columns)
    prices = prices.loc[:, adequate].ffill().bfill()

    print(f"Quality filter: {before} -> {len(prices.columns)} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Trading days: {len(prices)}")

    return prices
