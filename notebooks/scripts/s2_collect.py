"""
Step 2: Historical Data Collection

Smart collection from IB Gateway with per-ticker parquet caching.
- CURRENT tickers: skipped (no IB request)
- STALE tickers: incremental update (fetch only the gap)
- MISSING tickers: full 5Y download
"""

import pandas as pd
from pathlib import Path


def collect_data(
    tickers: list,
    ib_cache_dir: Path,
    processed_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 5,
    run_collection: bool = True,
) -> pd.DataFrame:
    """Collect/update historical price data.

    Returns wide DataFrame (dates x tickers) of close prices.
    """
    import sys
    sys.path.insert(0, str(ib_cache_dir.parent.parent / "src"))

    prices_ib_path = processed_dir / "etf_prices_ib.parquet"
    prices_yf_path = processed_dir / "etf_prices_filtered.parquet"

    prices = None

    # Check for existing consolidated data
    if prices_ib_path.exists():
        prices = pd.read_parquet(prices_ib_path)
        print(f"Loaded IB prices: {prices.shape[1]} tickers x {prices.shape[0]} days")
        print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    elif prices_yf_path.exists():
        prices = pd.read_parquet(prices_yf_path)
        print(f"Loaded yfinance prices: {prices.shape[1]} tickers x {prices.shape[0]} days")
    else:
        # Build from individual parquets if available
        cached = [f for f in ib_cache_dir.glob("*.parquet") if f.stem != "manifest"]
        if len(cached) > 50:
            print(f"Building from {len(cached)} cached parquets...")
            from data_collection.ib_data_collector import IBDataCollector

            class _DummyIB:
                pass
            collector = IBDataCollector(ib=_DummyIB(), cache_dir=str(ib_cache_dir))
            prices = collector.build_price_matrix()

    if not run_collection:
        if prices is None:
            print("No cached prices and collection disabled.")
        return prices

    # Run smart collection via IB
    try:
        import importlib
        import data_collection.ib_data_collector as _mod
        importlib.reload(_mod)
        from data_collection.ib_data_collector import IBDataCollector
        from ib_insync import IB

        ib = IB()
        ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10)
        print(f"Connected to IB: {ib.managedAccounts()[0]}\n")

        collector = IBDataCollector(
            ib=ib,
            cache_dir=str(ib_cache_dir),
            rate_limit_interval=12.0,
            duration="5 Y",
        )

        prices, results = collector.collect_universe(tickers)
        ib.disconnect()

    except Exception as e:
        print(f"IB failed: {e}")
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
