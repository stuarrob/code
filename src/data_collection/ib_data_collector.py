"""
IB Gateway Historical Data Collector

Batch downloads daily historical data for the full US ETF universe
from Interactive Brokers Gateway with rate limiting, per-ticker
parquet caching, and resume support.

Usage:
    from ib_insync import IB
    from data_collection.ib_data_collector import IBDataCollector

    ib = IB()
    ib.connect("127.0.0.1", 4001, clientId=5, readonly=True)

    collector = IBDataCollector(ib, cache_dir="data/ib_historical")

    # Smart collection (skips current, increments stale, full-downloads new):
    prices_df, results_df = collector.collect_universe(tickers)

Rate limits:
    IB allows ~60 historical data requests per 10 minutes.
    Default 12s spacing = 50 requests/10min (safe margin).
    Pacing violations trigger exponential backoff.
"""

import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Stock


class IBDataCollector:
    """Batch collector for IB historical data with rate limiting."""

    def __init__(
        self,
        ib: IB,
        cache_dir: str = "data/ib_historical",
        rate_limit_interval: float = 12.0,
        duration: str = "5 Y",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST",
        stale_days: int = 1,
    ):
        self.ib = ib
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_interval = rate_limit_interval
        self.duration = duration
        self.bar_size = bar_size
        self.what_to_show = what_to_show
        self.stale_days = stale_days
        self.manifest_path = self.cache_dir / "manifest.csv"

    def _log(self, msg: str):
        """Print a timestamped log line (newline, not carriage return)."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {msg}", flush=True)

    def _log_summary(self, msg: str):
        """Print a summary line without timestamp."""
        print(f"  {msg}", flush=True)

    # ------------------------------------------------------------------
    # Contract qualification
    # ------------------------------------------------------------------

    def qualify_contracts(
        self, tickers: List[str], batch_size: int = 50
    ) -> Tuple[List[Stock], List[str]]:
        """Qualify contracts on IB in batches.

        Returns (qualified_contracts, failed_tickers).
        """
        qualified = []
        failed = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            for t in batch:
                c = Stock(t, "SMART", "USD")
                try:
                    self.ib.qualifyContracts(c)
                    if c.conId:
                        qualified.append(c)
                    else:
                        failed.append(t)
                except Exception:
                    failed.append(t)
            self.ib.sleep(0.5)

            done = min(i + batch_size, len(tickers))
            if done % 200 == 0 or done == len(tickers):
                self._log(
                    f"Qualifying: {done}/{len(tickers)} "
                    f"({len(qualified)} ok, {len(failed)} failed)"
                )

        return qualified, failed

    # ------------------------------------------------------------------
    # Cache inspection
    # ------------------------------------------------------------------

    def _check_cache(self, ticker: str) -> Tuple[str, Optional[pd.Timestamp], int]:
        """Check the cache state for a single ticker.

        Returns (state, last_date, gap_days) where state is one of:
            "current"  — cached and up-to-date (gap <= stale_days)
            "stale"    — cached but needs incremental update
            "missing"  — no cached data, needs full download
        """
        cache_file = self.cache_dir / f"{ticker}.parquet"
        if not cache_file.exists():
            return "missing", None, 0

        try:
            df = pd.read_parquet(cache_file)
            if df.empty:
                return "missing", None, 0

            last_date = pd.Timestamp(df.index.max())
            today = pd.Timestamp.now().normalize()
            gap_days = (today - last_date).days

            if gap_days <= self.stale_days:
                return "current", last_date, gap_days
            else:
                return "stale", last_date, gap_days

        except Exception:
            return "missing", None, 0

    def get_cached_tickers(self) -> List[str]:
        """Return list of tickers already downloaded."""
        return [
            f.stem
            for f in self.cache_dir.glob("*.parquet")
            if f.stem != "manifest"
        ]

    def get_stale_tickers(self, max_age_days: int = 1) -> List[str]:
        """Return list of cached tickers that need updating."""
        stale = []
        today = pd.Timestamp.now().normalize()
        for f in self.cache_dir.glob("*.parquet"):
            if f.stem == "manifest":
                continue
            try:
                df = pd.read_parquet(f)
                last_date = pd.Timestamp(df.index.max())
                if (today - last_date).days > max_age_days:
                    stale.append(f.stem)
            except Exception:
                stale.append(f.stem)
        return sorted(stale)

    # ------------------------------------------------------------------
    # Single-ticker download / update
    # ------------------------------------------------------------------

    def download_single(self, contract: Stock) -> Tuple[Optional[pd.DataFrame], str]:
        """Full 5Y download for a single contract.

        Returns (DataFrame, log_message) or (None, log_message).
        Saves to cache_dir/{ticker}.parquet.
        """
        ticker = contract.symbol
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=self.duration,
                barSizeSetting=self.bar_size,
                whatToShow=self.what_to_show,
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                return None, f"NEW {ticker}: no data returned by IB"

            df = pd.DataFrame(
                [
                    {
                        "date": b.date,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                ]
            )
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            cache_file = self.cache_dir / f"{ticker}.parquet"
            df.to_parquet(cache_file)

            date_range = f"{df.index[0].date()} to {df.index[-1].date()}"
            msg = f"NEW {ticker}: downloaded {len(df)} bars ({date_range})"
            return df, msg

        except Exception as e:
            if "pacing" in str(e).lower():
                raise
            return None, f"NEW {ticker}: ERROR - {e}"

    def update_single(self, contract: Stock) -> Tuple[Optional[pd.DataFrame], str]:
        """Incrementally update cached data for a single contract.

        Checks the last date in the existing parquet, then only
        requests new bars since that date.

        Returns (DataFrame, log_message).
        """
        ticker = contract.symbol
        cache_file = self.cache_dir / f"{ticker}.parquet"

        if not cache_file.exists():
            return self.download_single(contract)

        try:
            existing = pd.read_parquet(cache_file)
            if existing.empty:
                return self.download_single(contract)

            last_date = pd.Timestamp(existing.index.max())
            today = pd.Timestamp.now().normalize()
            gap_days = (today - last_date).days

            if gap_days <= self.stale_days:
                msg = (
                    f"CURRENT {ticker}: last bar {last_date.date()}, "
                    f"gap {gap_days}d — already up to date"
                )
                return existing, msg

            # Calculate duration string for IB
            # Add a few extra days of overlap to handle weekends/holidays
            fetch_days = gap_days + 5
            if fetch_days <= 365:
                duration_str = f"{fetch_days} D"
            else:
                months = fetch_days // 30 + 1
                if months <= 12:
                    duration_str = f"{months} M"
                else:
                    duration_str = f"{months // 12 + 1} Y"

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting=self.bar_size,
                whatToShow=self.what_to_show,
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                msg = (
                    f"UPDATE {ticker}: last bar {last_date.date()}, "
                    f"requested {duration_str} — no new data returned"
                )
                return existing, msg

            new_df = pd.DataFrame(
                [
                    {
                        "date": b.date,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                ]
            )
            new_df["date"] = pd.to_datetime(new_df["date"])
            new_df = new_df.set_index("date").sort_index()

            # Merge: new data overwrites overlapping dates
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()

            # Save updated cache
            combined.to_parquet(cache_file)

            new_bars = len(combined) - len(existing)
            new_range = f"{new_df.index[0].date()} to {new_df.index[-1].date()}"
            msg = (
                f"UPDATE {ticker}: last bar was {last_date.date()}, "
                f"gap {gap_days}d, fetched {duration_str} → "
                f"+{new_bars} new bars ({new_range}), "
                f"total now {len(combined)} bars"
            )
            return combined, msg

        except Exception as e:
            if "pacing" in str(e).lower():
                raise
            return None, f"UPDATE {ticker}: ERROR - {e}"

    # ------------------------------------------------------------------
    # Universe-level collection (the main entry point)
    # ------------------------------------------------------------------

    def collect_universe(
        self,
        tickers: List[str],
        progress_callback: Optional[Callable] = None,
        max_consecutive_failures: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Smart collection: skip current, increment stale, full-download new.

        For each ticker:
          - CURRENT (cached, gap <= stale_days): skip entirely, no IB request
          - STALE (cached, gap > stale_days): incremental update (fetch only gap)
          - MISSING (no cache): full 5Y download

        This is the single entry point — there is no need for separate
        "collect" vs "update" modes.

        Returns:
            (prices_df, results_df)
        """
        self._log_summary("=" * 60)
        self._log_summary("DATA COLLECTION — scanning cache...")
        self._log_summary("=" * 60)

        # Phase 1: Scan cache to classify every ticker
        current_tickers = []
        stale_tickers = []   # (ticker, last_date, gap_days)
        missing_tickers = []

        for t in tickers:
            state, last_date, gap_days = self._check_cache(t)
            if state == "current":
                current_tickers.append(t)
            elif state == "stale":
                stale_tickers.append((t, last_date, gap_days))
            else:
                missing_tickers.append(t)

        self._log_summary("")
        self._log_summary(f"Universe: {len(tickers)} tickers")
        self._log_summary(
            f"  CURRENT (up-to-date, skip):    {len(current_tickers)}"
        )
        self._log_summary(
            f"  STALE (incremental update):     {len(stale_tickers)}"
        )
        self._log_summary(
            f"  MISSING (full 5Y download):     {len(missing_tickers)}"
        )
        self._log_summary("")

        # Show stale ticker details
        if stale_tickers:
            self._log_summary("Stale tickers to update:")
            for t, last_date, gap in stale_tickers[:20]:
                self._log_summary(
                    f"    {t:<6} last: {last_date.date()}  gap: {gap}d"
                )
            if len(stale_tickers) > 20:
                self._log_summary(
                    f"    ... and {len(stale_tickers) - 20} more"
                )
            self._log_summary("")

        # Nothing to do?
        if not stale_tickers and not missing_tickers:
            self._log(f"All {len(current_tickers)} tickers are current. "
                      "Building price matrix...")
            prices = self.build_price_matrix()
            results = pd.DataFrame(
                {"ticker": tickers, "status": "current", "bars": 0}
            )
            return prices, results

        # Phase 2: Qualify contracts that need IB requests
        need_work = [t for t, _, _ in stale_tickers] + missing_tickers
        self._log(f"Qualifying {len(need_work)} contracts on IB...")
        qualified, qual_failed = self.qualify_contracts(need_work)
        self._log(
            f"Qualified: {len(qualified)}, "
            f"Failed: {len(qual_failed)}"
        )
        if qual_failed:
            self._log_summary(
                f"  Failed tickers: "
                f"{', '.join(qual_failed[:15])}"
                f"{'...' if len(qual_failed) > 15 else ''}"
            )
        self._log_summary("")

        # Build lookup for stale tickers (to decide update vs download)
        stale_set = {t for t, _, _ in stale_tickers}

        # Phase 3: Process with rate limiting
        results_list = []
        consecutive_fails = 0
        delay = self.rate_limit_interval
        start_time = time.time()
        ib_requests = 0
        skipped_current = 0
        updated_count = 0
        new_count = 0
        failed_count = 0

        self._log_summary("=" * 60)
        self._log(f"Starting IB requests ({len(qualified)} tickers)...")
        self._log_summary("=" * 60)

        for i, contract in enumerate(qualified):
            ticker = contract.symbol
            is_stale = ticker in stale_set

            # ETA calculation
            elapsed = time.time() - start_time
            if ib_requests > 0:
                rate = ib_requests / elapsed
                remaining = len(qualified) - i - 1
                eta_min = (remaining / rate) / 60 if rate > 0 else 0
            else:
                eta_min = 0

            try:
                if is_stale:
                    df, msg = self.update_single(contract)
                else:
                    df, msg = self.download_single(contract)

                ib_requests += 1

                if df is not None and len(df) > 0:
                    status = "ok"
                    consecutive_fails = 0
                    delay = self.rate_limit_interval
                    if is_stale:
                        updated_count += 1
                    else:
                        new_count += 1
                else:
                    status = "empty"
                    consecutive_fails += 1
                    failed_count += 1

                # Log with progress counter and ETA
                progress = f"[{i + 1}/{len(qualified)}]"
                eta = f"ETA: {eta_min:.0f}m" if eta_min > 0 else ""
                self._log(f"{progress} {msg} {eta}")

                results_list.append(
                    {"ticker": ticker, "status": status,
                     "bars": len(df) if df is not None else 0}
                )

            except Exception as e:
                err_msg = str(e).lower()
                if "pacing" in err_msg:
                    delay = min(delay * 2, 120)
                    self._log(
                        f"[{i + 1}/{len(qualified)}] {ticker}: "
                        f"PACING VIOLATION — backing off to {delay:.0f}s"
                    )
                    time.sleep(60)
                    # Retry once
                    try:
                        if is_stale:
                            df, msg = self.update_single(contract)
                        else:
                            df, msg = self.download_single(contract)
                        ib_requests += 1

                        if df is not None and len(df) > 0:
                            results_list.append(
                                {"ticker": ticker, "status": "ok",
                                 "bars": len(df)}
                            )
                            consecutive_fails = 0
                            if is_stale:
                                updated_count += 1
                            else:
                                new_count += 1
                            self._log(f"  RETRY OK: {msg}")
                        else:
                            results_list.append(
                                {"ticker": ticker, "status": "retry_empty",
                                 "bars": 0}
                            )
                            failed_count += 1
                            self._log(f"  RETRY: {msg}")
                    except Exception as e2:
                        results_list.append(
                            {"ticker": ticker, "status": "retry_failed",
                             "bars": 0}
                        )
                        consecutive_fails += 1
                        failed_count += 1
                        self._log(f"  RETRY FAILED: {e2}")
                else:
                    results_list.append(
                        {"ticker": ticker, "status": "error", "bars": 0}
                    )
                    consecutive_fails += 1
                    failed_count += 1
                    self._log(
                        f"[{i + 1}/{len(qualified)}] {ticker}: ERROR — {e}"
                    )

            self._append_manifest(results_list[-1])

            if progress_callback:
                ok = results_list[-1]["status"] == "ok"
                progress_callback(i + 1, len(qualified), ticker, ok)

            if consecutive_fails >= max_consecutive_failures:
                self._log(
                    f"ABORTING: {consecutive_fails} consecutive failures. "
                    f"Check IB connection."
                )
                break

            # Rate limit (extra pause every 45 requests)
            if ib_requests > 0 and ib_requests % 45 == 0:
                self._log(f"Batch pause (60s) after {ib_requests} IB requests...")
                time.sleep(60)

            time.sleep(delay)

        # Add qualification failures
        for t in qual_failed:
            results_list.append(
                {"ticker": t, "status": "not_qualified", "bars": 0}
            )

        results_df = pd.DataFrame(results_list)
        total_time = (time.time() - start_time) / 60

        # Final summary
        self._log_summary("")
        self._log_summary("=" * 60)
        self._log_summary("COLLECTION COMPLETE")
        self._log_summary("=" * 60)
        self._log_summary(f"  Current (skipped):       {len(current_tickers)}")
        self._log_summary(f"  Updated (incremental):   {updated_count}")
        self._log_summary(f"  New (full download):     {new_count}")
        self._log_summary(f"  Failed/empty:            {failed_count}")
        self._log_summary(f"  Not qualified on IB:     {len(qual_failed)}")
        self._log_summary(f"  IB requests made:        {ib_requests}")
        self._log_summary(f"  Time: {total_time:.1f} minutes")
        self._log_summary("=" * 60)
        self._log_summary("")

        self._log("Building price matrix...")
        prices = self.build_price_matrix()

        return prices, results_df

    # Keep update_universe as an alias for backward compatibility
    def update_universe(
        self,
        tickers: List[str],
        progress_callback: Optional[Callable] = None,
        max_consecutive_failures: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Alias for collect_universe (which handles updates automatically)."""
        return self.collect_universe(
            tickers, progress_callback, max_consecutive_failures
        )

    # ------------------------------------------------------------------
    # IB scanner
    # ------------------------------------------------------------------

    def scan_ib_etf_universe(self, max_results: int = 5000) -> List[str]:
        """Discover all US ETFs available on IB using the scanner API."""
        from ib_insync import ScannerSubscription

        sub = ScannerSubscription(
            instrument="STK",
            locationCode="STK.US",
            scanCode="TOP_PERC_GAIN",
            numberOfRows=max_results,
        )
        tag_values = [("stockTypeFilter", "ETF")]

        try:
            results = self.ib.reqScannerData(
                sub, scannerSubscriptionFilterOptions=tag_values
            )
            tickers = [
                r.contractDetails.contract.symbol for r in results
            ]
            self._log(f"IB scanner found {len(tickers)} US ETFs")
            return sorted(set(tickers))
        except Exception as e:
            self._log(f"IB scanner failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Price matrix
    # ------------------------------------------------------------------

    def build_price_matrix(
        self, min_history_days: int = 252
    ) -> pd.DataFrame:
        """Build consolidated close-price matrix from cached parquets.

        Returns wide DataFrame (dates x tickers). Only includes tickers
        with >= min_history_days of data.
        """
        dfs = {}
        short_count = 0
        for f in sorted(self.cache_dir.glob("*.parquet")):
            ticker = f.stem
            if ticker == "manifest":
                continue
            try:
                df = pd.read_parquet(f)
                if len(df) >= min_history_days and "close" in df.columns:
                    dfs[ticker] = df["close"]
                else:
                    short_count += 1
            except Exception:
                pass

        if not dfs:
            return pd.DataFrame()

        prices = pd.DataFrame(dfs)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        # Save consolidated file
        output_path = (
            self.cache_dir.parent / "processed" / "etf_prices_ib.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(output_path)

        self._log_summary(
            f"Price matrix: {prices.shape[1]} tickers x "
            f"{prices.shape[0]} days "
            f"({short_count} excluded: <{min_history_days} days)"
        )
        self._log_summary(
            f"  Range: {prices.index[0].date()} to "
            f"{prices.index[-1].date()}"
        )
        self._log_summary(f"  Saved: {output_path}")

        return prices

    # ------------------------------------------------------------------
    # Manifest (download log)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict:
        """Load download manifest."""
        if not self.manifest_path.exists():
            return {"ticker": []}
        try:
            df = pd.read_csv(self.manifest_path)
            return df.to_dict(orient="list")
        except Exception:
            return {"ticker": []}

    def _append_manifest(self, row: Dict):
        """Append a single result row to the manifest."""
        file_exists = self.manifest_path.exists()
        with open(self.manifest_path, "a", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["ticker", "status", "bars", "timestamp"]
            )
            if not file_exists:
                w.writeheader()
            row_with_ts = {**row, "timestamp": datetime.now().isoformat()}
            w.writerow(row_with_ts)


def collect_with_fallback(
    tickers: List[str],
    ib: Optional[IB] = None,
    cache_dir: str = "data/ib_historical",
    yf_fallback: bool = True,
) -> pd.DataFrame:
    """Try IB first, fall back to yfinance.

    If ib is None or not connected, uses yfinance for all tickers.
    If ib is connected, uses IBDataCollector, then fills gaps
    with yfinance for any missing tickers.

    Returns wide DataFrame (dates x tickers) of close prices.
    """
    prices = pd.DataFrame()

    # Try IB
    if ib is not None and ib.isConnected():
        collector = IBDataCollector(ib=ib, cache_dir=cache_dir)
        prices, _ = collector.collect_universe(tickers)

    # Fill gaps with yfinance
    if yf_fallback:
        ib_tickers = set(prices.columns) if not prices.empty else set()
        missing = [t for t in tickers if t not in ib_tickers]

        if missing:
            print(f"  Fetching {len(missing)} tickers from yfinance...")
            try:
                import yfinance as yf

                batch_size = 500
                yf_frames = []
                for i in range(0, len(missing), batch_size):
                    batch = missing[i : i + batch_size]
                    data = yf.download(
                        " ".join(batch),
                        period="5y",
                        auto_adjust=True,
                        progress=False,
                    )
                    if "Close" in data.columns or len(batch) == 1:
                        close = (
                            data["Close"]
                            if len(batch) > 1
                            else data["Close"].to_frame(batch[0])
                        )
                        yf_frames.append(close)

                if yf_frames:
                    yf_prices = pd.concat(yf_frames, axis=1)
                    if prices.empty:
                        prices = yf_prices
                    else:
                        prices = prices.join(yf_prices, how="outer")
                    print(
                        f"  yfinance: got {len(yf_prices.columns)} tickers"
                    )
            except Exception as e:
                print(f"  yfinance fallback failed: {e}")

    return prices
