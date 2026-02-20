#!/usr/bin/env python3
"""
Daily ETF Data Collection

Connects to IB Gateway and updates historical price data for all ETFs
in the universe. Uses smart caching: only fetches incremental updates
for tickers that are stale or missing.

Waits for IB Gateway to be available (polls every 60s for up to 2 hours).

Outputs:
    ~/trade_data/ETFTrader/ib_historical/{TICKER}.parquet — per-ticker cache
    ~/trade_data/ETFTrader/processed/etf_prices_ib.parquet — combined matrix

Idempotent: skips tickers already up-to-date today.

Usage:
    python scripts/daily_etf_data.py              # full universe update
    python scripts/daily_etf_data.py --dry-run    # show cache state only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATA_DIR = Path.home() / "trade_data" / "ETFTrader"
IB_CACHE_DIR = DATA_DIR / "ib_historical"
PROCESSED_DIR = DATA_DIR / "processed"

IB_HOST = "127.0.0.1"
IB_PORT = 4001
IB_CLIENT_ID = 22  # Dedicated client ID for daily ETF job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_etf")


def get_universe() -> list:
    """Load the ETF universe ticker list."""
    from data_collection.comprehensive_etf_list import load_full_universe
    tickers, _ = load_full_universe()
    return tickers


def collect_etf_data(tickers: list) -> pd.DataFrame:
    """Collect/update ETF price data via IB."""
    from ib_wait import wait_for_ib
    from data_collection.ib_data_collector import IBDataCollector

    ib = wait_for_ib(IB_HOST, IB_PORT, IB_CLIENT_ID)

    collector = IBDataCollector(
        ib=ib,
        cache_dir=str(IB_CACHE_DIR),
        rate_limit_interval=12.0,
        duration="5 Y",
    )

    prices, results = collector.collect_universe(tickers)
    ib.disconnect()

    # Save combined price matrix
    if prices is not None and not prices.empty:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "etf_prices_ib.parquet"
        prices.to_parquet(out_path)
        log.info("Saved price matrix: %s (%d tickers x %d days)",
                 out_path, prices.shape[1], prices.shape[0])

    return prices


def main():
    parser = argparse.ArgumentParser(description="Daily ETF data collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cache state without collecting")
    args = parser.parse_args()

    start = time.time()

    tickers = get_universe()
    log.info("Universe: %d tickers", len(tickers))

    if args.dry_run:
        from data_collection.ib_data_collector import IBDataCollector

        class _DummyIB:
            pass
        collector = IBDataCollector(ib=_DummyIB(), cache_dir=str(IB_CACHE_DIR))
        current = stale = missing = 0
        for t in tickers:
            state, _ = collector._check_cache(t)
            if state == "current":
                current += 1
            elif state == "stale":
                stale += 1
            else:
                missing += 1
        log.info("Cache state: %d current, %d stale, %d missing", current, stale, missing)
        return

    prices = collect_etf_data(tickers)

    elapsed = time.time() - start
    log.info("Done in %.1f minutes", elapsed / 60)
    if prices is not None:
        log.info("Final matrix: %d tickers x %d days", prices.shape[1], prices.shape[0])


if __name__ == "__main__":
    main()
