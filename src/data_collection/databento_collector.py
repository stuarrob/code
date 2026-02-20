"""
Databento Historical Data Collector

On-demand daily data collection for the US ETF universe via
Databento's XNAS.ITCH dataset. Fetches OHLCV bars in batches,
writes to the same per-ticker parquet cache used by IBDataCollector.

No cron, no rate limiting — just call update_tickers() when you
need fresh data. Typical daily update completes in seconds.

Usage:
    from data_collection.databento_collector import DatabentoCollector

    collector = DatabentoCollector()
    prices, results = collector.update_tickers(tickers)
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger("databento_collector")

DEFAULT_CACHE_DIR = str(Path.home() / "trade_data" / "ETFTrader" / "ib_historical")
DEFAULT_DATASET = "XNAS.ITCH"


class DatabentoCollector:
    """On-demand daily data collector using Databento API.

    Reads from and writes to the same per-ticker parquet cache
    used by IBDataCollector. Fetches only the gap between last
    cached date and today.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        dataset: str = DEFAULT_DATASET,
        stale_days: int = 1,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.stale_days = stale_days

        # Resolve API key: param > env var > .env file
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                self.api_key = os.getenv("DATABENTO_API_KEY")
            except ImportError:
                pass

        if not self.api_key:
            raise ValueError(
                "Databento API key not found. Set DATABENTO_API_KEY env var "
                "or pass api_key parameter."
            )

        import databento as db
        self.client = db.Historical(self.api_key)

    # ------------------------------------------------------------------
    # Cache inspection
    # ------------------------------------------------------------------

    def _check_cache(self, ticker: str) -> Tuple[str, Optional[pd.Timestamp], int]:
        """Check cache state for a single ticker.

        Returns (state, last_date, gap_days) where state is one of:
            "current"  — cached and up-to-date (gap <= stale_days)
            "stale"    — cached but needs incremental update
            "missing"  — no cached data
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

    # ------------------------------------------------------------------
    # Databento fetch
    # ------------------------------------------------------------------

    def _fetch_batch(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV-1d bars for a batch of tickers.

        Returns a DataFrame with columns:
            symbol, date, open, high, low, close, volume
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.client.timeseries.get_range(
                dataset=self.dataset,
                symbols=tickers,
                schema="ohlcv-1d",
                start=start,
                end=end,
            )

        df = data.to_df()
        if df.empty:
            return pd.DataFrame()

        # Normalize: ts_event (UTC datetime) → date-only
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize()
        df = df[["symbol", "date", "open", "high", "low", "close", "volume"]]
        df["volume"] = df["volume"].astype(float)

        return df

    # ------------------------------------------------------------------
    # Merge and save
    # ------------------------------------------------------------------

    def _merge_and_save(self, ticker: str, new_data: pd.DataFrame) -> str:
        """Merge new bars with existing cache, save, return status message."""
        cache_file = self.cache_dir / f"{ticker}.parquet"

        # Format new data to match IB cache format
        ticker_data = new_data[new_data["symbol"] == ticker].copy()
        if ticker_data.empty:
            return f"  {ticker}: no data returned by Databento"

        ticker_data = ticker_data.set_index("date")[["open", "high", "low", "close", "volume"]]
        ticker_data = ticker_data.sort_index()
        ticker_data = ticker_data[~ticker_data.index.duplicated(keep="last")]

        if cache_file.exists():
            try:
                existing = pd.read_parquet(cache_file)
                old_last = existing.index.max()

                # Boundary validation: check for suspicious price jumps (possible split)
                if not existing.empty and not ticker_data.empty:
                    cached_last_close = float(existing.iloc[-1]["close"])
                    # Find the first new bar AFTER the cached data
                    new_after = ticker_data[ticker_data.index > old_last]
                    if not new_after.empty and cached_last_close > 0:
                        new_first_close = float(new_after.iloc[0]["close"])
                        ratio = new_first_close / cached_last_close
                        if ratio > 1.5 or ratio < 0.67:
                            log.warning(
                                "%s: boundary price jump %.1f%% "
                                "(cached=%.2f, new=%.2f) — possible split",
                                ticker,
                                (ratio - 1) * 100,
                                cached_last_close,
                                new_first_close,
                            )

                # Merge: new data overwrites overlapping dates
                combined = pd.concat([existing, ticker_data])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()

                new_bars = len(combined) - len(existing)
                combined.to_parquet(cache_file)
                return (
                    f"  {ticker}: +{new_bars} bars "
                    f"(total {len(combined)}, "
                    f"last={combined.index[-1].date()})"
                )
            except Exception as e:
                log.warning("%s: cache read failed (%s), overwriting", ticker, e)

        # No existing cache — just save
        ticker_data.to_parquet(cache_file)
        return (
            f"  {ticker}: NEW {len(ticker_data)} bars "
            f"({ticker_data.index[0].date()} to {ticker_data.index[-1].date()})"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def update_tickers(
        self,
        tickers: List[str],
        batch_size: int = 500,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Update cached data for the given tickers.

        Scans cache, groups stale tickers by gap start date,
        fetches from Databento in batches, merges with cache,
        and builds the consolidated price matrix.

        Returns (price_matrix, results_df).
        """
        start_time = time.time()

        # Phase 1: Scan cache
        current = []
        stale = []    # (ticker, last_date, gap_days)
        missing = []

        for t in tickers:
            state, last_date, gap_days = self._check_cache(t)
            if state == "current":
                current.append(t)
            elif state == "stale":
                stale.append((t, last_date, gap_days))
            else:
                missing.append(t)

        log.info(
            "Universe: %d tickers — %d current, %d stale, %d missing",
            len(tickers), len(current), len(stale), len(missing),
        )

        if not stale and not missing:
            log.info("All tickers current. Building price matrix...")
            prices = self.build_price_matrix()
            results = pd.DataFrame(
                {"ticker": tickers, "status": "current", "bars": 0}
            )
            return prices, results

        # Phase 2: Group stale tickers by start date for efficient batching
        # Most tickers share the same last_date, so they go in one batch
        today_str = datetime.now().strftime("%Y-%m-%d")
        fetch_groups: Dict[str, List[str]] = {}

        for t, last_date, gap_days in stale:
            # Start from the day after last cached date
            start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            fetch_groups.setdefault(start, []).append(t)

        # Missing tickers: no cache, but Databento XNAS.ITCH only goes back to ~May 2018
        # For tickers with existing IB cache that's stale, we fill the gap
        # For truly missing tickers (no cache at all), skip — they need IB or don't exist
        if missing:
            log.info(
                "Skipping %d tickers with no cache (need IB initial download "
                "or are unavailable): %s%s",
                len(missing),
                ", ".join(missing[:10]),
                "..." if len(missing) > 10 else "",
            )

        # Phase 3: Fetch from Databento
        results_list = []
        updated_count = 0
        fetch_errors = 0

        for start_date, group_tickers in sorted(fetch_groups.items()):
            log.info(
                "Fetching %d tickers from %s to %s...",
                len(group_tickers), start_date, today_str,
            )

            # Process in batches
            for i in range(0, len(group_tickers), batch_size):
                batch = group_tickers[i : i + batch_size]

                try:
                    batch_df = self._fetch_batch(batch, start_date, today_str)

                    if batch_df.empty:
                        log.warning(
                            "No data returned for batch of %d tickers "
                            "(start=%s)",
                            len(batch), start_date,
                        )
                        for t in batch:
                            results_list.append(
                                {"ticker": t, "status": "no_data", "bars": 0}
                            )
                        continue

                    # Save each ticker
                    returned_symbols = set(batch_df["symbol"].unique())
                    for t in batch:
                        if t in returned_symbols:
                            msg = self._merge_and_save(t, batch_df)
                            log.info(msg)
                            results_list.append(
                                {"ticker": t, "status": "updated", "bars": 1}
                            )
                            updated_count += 1
                        else:
                            results_list.append(
                                {"ticker": t, "status": "not_in_databento", "bars": 0}
                            )

                except Exception as e:
                    log.error("Batch fetch failed: %s", e)
                    fetch_errors += 1
                    for t in batch:
                        results_list.append(
                            {"ticker": t, "status": "error", "bars": 0}
                        )

        # Add current/missing tickers to results
        for t in current:
            results_list.append({"ticker": t, "status": "current", "bars": 0})
        for t in missing:
            results_list.append({"ticker": t, "status": "missing_cache", "bars": 0})

        results_df = pd.DataFrame(results_list)
        elapsed = time.time() - start_time

        log.info(
            "Done in %.1fs: %d updated, %d current, %d missing, %d errors",
            elapsed, updated_count, len(current), len(missing), fetch_errors,
        )

        # Phase 4: Build price matrix
        log.info("Building price matrix...")
        prices = self.build_price_matrix()

        return prices, results_df

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
            self.cache_dir.parent / "processed" / "etf_prices_db.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(output_path)

        log.info(
            "Price matrix: %d tickers x %d days (%d excluded: <%d days)",
            prices.shape[1], prices.shape[0], short_count, min_history_days,
        )
        log.info(
            "  Range: %s to %s",
            prices.index[0].date(), prices.index[-1].date(),
        )
        log.info("  Saved: %s", output_path)

        return prices
