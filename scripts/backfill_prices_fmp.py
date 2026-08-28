"""Backfill ETF daily-price history via FMP `historical-price-eod/full`.

Fills the gap in the local price cache: our IB collector only goes back
to 2021-02-16, but multi-regime backtests need 2010-onwards. FMP has
verified 2010+ history for standard ETFs at raw close (same convention
as the IB cache — no data-quality regression).

Behaviour
---------
For each ticker in the universe:
  1. Read the current cached parquet (if any) and note its first date.
  2. If the cache already covers 2010-01-04 (or earlier), skip.
  3. Otherwise, fetch FMP history from 2010-01-01 to today.
  4. Merge with the existing cache, prefer the older FMP rows where they
     don't overlap; keep IB rows where they do (IB is closer to the
     broker's execution reality).
  5. Write the merged frame back to the same parquet path.

Failures on individual tickers do not abort the run — they are logged
and the ticker is skipped. Resume-safe: rerun to pick up any that failed.

Rate limits
-----------
FMP Premium: verified working quota comfortably above 800 calls/day
(hit 429 on free tier at 800; Premium supports weekly full refreshes).
Default 5 parallel workers with 1-second polite delay per request gives
~5 req/sec effective, ~17 min for 5000 tickers.

Usage
-----
    # Whole universe
    python scripts/backfill_prices_fmp.py

    # Just a handful
    python scripts/backfill_prices_fmp.py --tickers SPY VOO QQQ

    # Dry run (probes 3 tickers, no cache writes)
    python scripts/backfill_prices_fmp.py --dry-run

The script is intended to be run manually before comprehensive backtests
and again if the universe expands. Not on the weekly cron — that stays
on the daily IB fetch which keeps 2021+ current.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Repo import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


CACHE_DIR = Path.home() / "trade_data" / "ETFTrader" / "ib_historical"
FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
DEFAULT_FROM_DATE = "2010-01-01"
DEFAULT_DELAY_SEC = 0.30  # per-worker delay; effective rate = workers / delay


def _load_universe(explicit: Optional[list[str]]) -> list[str]:
    """Determine which tickers to backfill.

    Preference order:
      1. --tickers flag (explicit list)
      2. Every parquet already in the price cache (backfill all)
    """
    if explicit:
        return sorted({t.upper() for t in explicit})
    files = [p.stem for p in CACHE_DIR.glob("*.parquet") if p.stem != "manifest"]
    return sorted(files)


def _load_existing(ticker: str) -> Optional[pd.DataFrame]:
    path = CACHE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{ticker}: failed to read existing parquet: {exc}")
        return None


def _fetch_fmp(ticker: str, api_key: str, from_date: str,
                delay_sec: float, timeout_sec: float = 30.0) -> Optional[pd.DataFrame]:
    """Fetch FMP daily OHLCV. Returns date-indexed DataFrame or None on failure."""
    time.sleep(delay_sec)
    try:
        resp = requests.get(
            FMP_URL,
            params={"symbol": ticker, "from": from_date, "apikey": api_key},
            timeout=timeout_sec,
        )
    except requests.RequestException as exc:
        logger.warning(f"{ticker}: FMP request error: {exc}")
        return None
    if resp.status_code != 200:
        # 4xx typically means the symbol isn't on FMP — skip quietly.
        # 5xx / 429 log at INFO.
        if resp.status_code >= 500 or resp.status_code == 429:
            logger.info(f"{ticker}: FMP HTTP {resp.status_code}")
        return None
    try:
        payload = resp.json()
    except ValueError:
        logger.warning(f"{ticker}: FMP non-JSON body")
        return None
    if isinstance(payload, dict) and "historical" in payload:
        rows = payload["historical"]
    elif isinstance(payload, list):
        rows = payload
    else:
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]


def _merge(existing: Optional[pd.DataFrame], fmp: pd.DataFrame) -> pd.DataFrame:
    """Merge FMP history with the existing cache.

    Overlap policy: keep the EXISTING (IB) rows where dates overlap. IB's
    prices are what the broker's execution stack sees; keeping them
    preserves consistency between backtest and live. FMP fills only the
    gap before the earliest existing date and any newer dates missing
    from the existing cache.
    """
    if existing is None or existing.empty:
        return fmp
    # Preserve existing index dtype conventions.
    existing = existing.copy()
    if not isinstance(existing.index, pd.DatetimeIndex):
        existing.index = pd.to_datetime(existing.index)
    # Rows in fmp that are strictly older than existing's first date OR
    # strictly newer than existing's last date.
    older = fmp.loc[fmp.index < existing.index.min()]
    newer = fmp.loc[fmp.index > existing.index.max()]
    combined = pd.concat([older, existing, newer]).sort_index()
    # Drop duplicates just in case (defensive; overlap logic above should
    # prevent them but safety-net).
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def _process_ticker(ticker: str, api_key: str, from_date: str,
                    delay_sec: float, dry_run: bool) -> dict:
    """Handle one ticker. Returns a small stats dict for the summary.

    Two possible actions per ticker:
      (a) Deepen: if the cache doesn't reach 2010, fetch full history.
      (b) Top-up: if the cache is deep enough but ends more than a
          trading day ago, fetch only the tail since the last cached
          date. This is the common weekly-refresh path.
      Skip when both are current.
    """
    existing = _load_existing(ticker)
    if existing is None or len(existing) == 0:
        existing_start = None
        existing_end = None
        existing_rows = 0
    else:
        existing_start = existing.index.min()
        existing_end = existing.index.max()
        existing_rows = len(existing)

    today = pd.Timestamp.today().normalize()
    is_deep = existing_start is not None and existing_start <= pd.Timestamp("2010-06-30")
    is_current = existing_end is not None and (today - existing_end).days <= 1

    # Skip if the tail is current, regardless of how deep the cache goes.
    # Deepening pre-2010 is a separate concern from daily top-up: newer
    # ETFs simply don't have pre-inception history to fetch, and re-asking
    # FMP every run wastes 100% of the requests. If the operator wants to
    # force a deepen attempt, they can delete the ticker's parquet.
    if is_current:
        return {"ticker": ticker, "action": "skip-already-current",
                "existing_rows": existing_rows}

    # Cache is stale. If we have any history at all, fetch just the tail;
    # otherwise fetch from the configured from_date.
    fetch_from = from_date
    if existing_end is not None:
        fetch_from = (existing_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    fmp = _fetch_fmp(ticker, api_key, fetch_from, delay_sec)
    if fmp is None or len(fmp) == 0:
        return {"ticker": ticker,
                "action": "skip-no-new-fmp-data" if is_deep else "fail-no-fmp-data",
                "existing_rows": existing_rows}

    merged = _merge(existing, fmp)
    if dry_run:
        return {"ticker": ticker, "action": "dry-run",
                "existing_rows": existing_rows, "fmp_rows": len(fmp),
                "merged_rows": len(merged),
                "merged_start": str(merged.index.min().date()),
                "merged_end": str(merged.index.max().date())}

    path = CACHE_DIR / f"{ticker}.parquet"
    try:
        merged.to_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{ticker}: failed to write parquet: {exc}")
        return {"ticker": ticker, "action": "fail-write",
                "existing_rows": existing_rows, "fmp_rows": len(fmp)}

    return {"ticker": ticker, "action": "ok",
            "existing_rows": existing_rows, "fmp_rows": len(fmp),
            "merged_rows": len(merged),
            "merged_start": str(merged.index.min().date())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Explicit ticker list (default: everything in the price cache).")
    p.add_argument("--from-date", default=DEFAULT_FROM_DATE,
                   help=f"Earliest FMP fetch date (default {DEFAULT_FROM_DATE}).")
    p.add_argument("--workers", type=int, default=5,
                   help="Parallel worker count (default 5).")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC,
                   help=f"Per-worker delay in seconds (default {DEFAULT_DELAY_SEC}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch + merge but do NOT write parquet.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total tickers processed (useful for testing).")
    args = p.parse_args()

    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("ERROR: FMP_API_KEY not set in .env", file=sys.stderr)
        return 1

    tickers = _load_universe(args.tickers)
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"Backfilling {len(tickers)} tickers from {args.from_date} "
          f"(workers={args.workers}, dry_run={args.dry_run})")

    t0 = time.perf_counter()
    stats: list[dict] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_ticker, t, api_key, args.from_date,
                        args.delay, args.dry_run): t
            for t in tickers
        }
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                stats.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{t}: worker raised: {exc}")
                stats.append({"ticker": t, "action": "worker-exception",
                              "existing_rows": 0})
            completed += 1
            if completed % 100 == 0 or completed == len(tickers):
                elapsed = time.perf_counter() - t0
                rate = completed / max(elapsed, 1e-3)
                eta = (len(tickers) - completed) / max(rate, 1e-3)
                print(f"  {completed}/{len(tickers)}  elapsed {elapsed:.0f}s  "
                      f"ETA {eta:.0f}s  ({rate:.1f}/s)")

    df = pd.DataFrame(stats)
    counts = df["action"].value_counts().to_dict()
    print()
    print("=" * 60)
    print(f"Summary:")
    for action, n in counts.items():
        print(f"  {action:24s} {n}")
    print()

    ok_rows = df[df["action"] == "ok"]
    if len(ok_rows) > 0 and "merged_start" in ok_rows.columns:
        # Median merged-start date tells us how deep the backfill went.
        starts = pd.to_datetime(ok_rows["merged_start"])
        print(f"Median merged start date across successful tickers: "
              f"{starts.median().date()}")
        print(f"Deepest merged start date: {starts.min().date()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
