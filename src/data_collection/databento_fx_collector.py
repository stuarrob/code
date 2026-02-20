"""
Databento Historical FX Option Surface Collector

Fetches 3+ years of CME FX futures option data from Databento's GLBX.MDP3
dataset, computes implied volatilities via Black-76, and produces a DataFrame
compatible with the existing SABR calibration pipeline.

Data flow:
    1. Fetch option OHLCV settlement prices (ohlcv-1d)
    2. Fetch option definitions (strike, expiry, underlying)
    3. Fetch underlying futures OHLCV settlement prices
    4. Compute implied vols via Black-76
    5. Output in FXOptionCollector format for SABR calibration

Usage:
    from data_collection.databento_fx_collector import DatabentoFXCollector

    collector = DatabentoFXCollector()
    surface_df = collector.fetch_history(
        currencies=["EUR", "GBP"],
        start="2023-01-01",
        end="2026-02-20",
    )
"""

import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("databento_fx")

DEFAULT_CACHE_DIR = str(Path.home() / "trade_data" / "ETFTrader" / "fx_options_db")

# Currency -> (futures parent symbol, options parent symbol)
SYMBOL_MAP = {
    "EUR": ("6E.FUT", "EUU.OPT"),
    "GBP": ("6B.FUT", "GBU.OPT"),
    "AUD": ("6A.FUT", "ADU.OPT"),
    "CAD": ("6C.FUT", "CAU.OPT"),
    "CHF": ("6S.FUT", "OZS.OPT"),
    "JPY": ("6J.FUT", "JPU.OPT"),
}


class DatabentoFXCollector:
    """Historical FX option surface collector via Databento API.

    Fetches option + futures OHLCV data and definitions from GLBX.MDP3,
    computes implied vols via Black-76, and outputs a DataFrame matching
    the FXOptionCollector format for SABR calibration.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        dataset: str = "GLBX.MDP3",
        max_dte: int = 365,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.max_dte = max_dte

        # Resolve API key
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
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        stype_in: str = "parent",
    ) -> pd.DataFrame:
        """Fetch OHLCV-1d data for a parent symbol."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.client.timeseries.get_range(
                dataset=self.dataset,
                symbols=[symbol],
                stype_in=stype_in,
                schema="ohlcv-1d",
                start=start,
                end=end,
            )

        df = data.to_df()
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize()
        return df

    def _fetch_definitions(
        self,
        symbol: str,
        date: str,
        stype_in: str = "parent",
    ) -> pd.DataFrame:
        """Fetch instrument definitions for a single day."""
        # Add one day to end to ensure we capture the target date
        end_dt = pd.Timestamp(date) + pd.Timedelta(days=1)
        end_str = end_dt.strftime("%Y-%m-%d")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.client.timeseries.get_range(
                dataset=self.dataset,
                symbols=[symbol],
                stype_in=stype_in,
                schema="definition",
                start=date,
                end=end_str,
            )

        df = data.to_df()
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        return df

    # ------------------------------------------------------------------
    # Definition metadata builder
    # ------------------------------------------------------------------

    def _build_symbol_metadata(
        self,
        opt_symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Build a symbol-to-metadata map from sampled definitions.

        Fetches definitions at regular intervals across the date range
        to capture all option contracts that existed.
        """
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        # Sample definitions on the first business day of each month
        sample_dates = pd.date_range(start_dt, end_dt, freq="MS")
        # Shift to next business day
        sample_dates = [
            d + pd.tseries.offsets.BDay(0) for d in sample_dates
        ]

        all_defs = []
        for d in sample_dates:
            d_str = d.strftime("%Y-%m-%d")
            try:
                defs = self._fetch_definitions(opt_symbol, d_str)
                if not defs.empty:
                    all_defs.append(defs)
                    log.debug("  Definitions for %s: %d rows", d_str, len(defs))
            except Exception as e:
                log.warning("  Failed to fetch definitions for %s: %s", d_str, e)

        if not all_defs:
            return pd.DataFrame()

        combined = pd.concat(all_defs, ignore_index=True)

        # Filter to calls and puts only (exclude spreads T, multi-leg M)
        if "instrument_class" in combined.columns:
            combined = combined[combined["instrument_class"].isin(["C", "P"])].copy()

        # Deduplicate by raw_symbol (keep the first definition seen)
        if "raw_symbol" in combined.columns:
            combined = combined.drop_duplicates(subset=["raw_symbol"], keep="first")

        # Extract key fields
        meta = pd.DataFrame()
        meta["symbol"] = combined["raw_symbol"].values
        meta["right"] = combined["instrument_class"].values
        meta["strike"] = pd.to_numeric(combined["strike_price"], errors="coerce").values

        # Parse expiration
        if "expiration" in combined.columns:
            meta["expiration_dt"] = pd.to_datetime(
                combined["expiration"].values, utc=True
            ).tz_localize(None)
            meta["expiration"] = meta["expiration_dt"].dt.strftime("%Y%m%d")

        # Underlying futures symbol
        if "underlying" in combined.columns:
            meta["underlying"] = combined["underlying"].values

        meta = meta.dropna(subset=["strike", "expiration_dt"])

        log.info(
            "  Symbol metadata: %d unique options (%d calls, %d puts)",
            len(meta),
            (meta["right"] == "C").sum(),
            (meta["right"] == "P").sum(),
        )
        return meta

    # ------------------------------------------------------------------
    # Match options to underlying futures
    # ------------------------------------------------------------------

    def _match_and_compute_iv(
        self,
        opt_ohlcv: pd.DataFrame,
        fut_ohlcv: pd.DataFrame,
        meta: pd.DataFrame,
        currency: str,
    ) -> pd.DataFrame:
        """Match options to underlying futures and compute implied vols.

        Args:
            opt_ohlcv: Option OHLCV data (date, symbol, close, volume, ...)
            fut_ohlcv: Futures OHLCV data (date, symbol, close, ...)
            meta: Symbol metadata (symbol, right, strike, expiration_dt, underlying)
            currency: Currency code (e.g., "EUR")

        Returns:
            DataFrame matching FXOptionCollector format.
        """
        from fx_options.black76 import implied_vol

        # Build futures price lookup: (date, symbol) -> close
        fut_prices = {}
        for _, row in fut_ohlcv.iterrows():
            key = (row["date"], row["symbol"])
            fut_prices[key] = row["close"]

        # Build metadata lookup: symbol -> (right, strike, expiration_dt, underlying)
        meta_dict = {}
        for _, row in meta.iterrows():
            meta_dict[row["symbol"]] = {
                "right": row["right"],
                "strike": row["strike"],
                "expiration_dt": row["expiration_dt"],
                "expiration": row["expiration"],
                "underlying": row.get("underlying", ""),
            }

        rows = []
        no_meta = 0
        no_underlying = 0
        iv_fail = 0
        total = 0

        for _, orow in opt_ohlcv.iterrows():
            sym = orow["symbol"]
            date = orow["date"]
            opt_close = orow["close"]

            # Skip zero/negative prices
            if not (opt_close > 0):
                continue

            total += 1

            # Look up metadata
            m = meta_dict.get(sym)
            if m is None:
                no_meta += 1
                continue

            strike = m["strike"]
            exp_dt = m["expiration_dt"]
            right = m["right"]
            underlying_sym = m["underlying"]
            expiration = m["expiration"]

            # Compute DTE
            dte = (exp_dt - date).days
            if dte <= 0 or dte > self.max_dte:
                continue

            # Look up underlying futures price
            F = fut_prices.get((date, underlying_sym))
            if F is None or F <= 0:
                no_underlying += 1
                continue

            # Compute time to expiry in years
            T = dte / 365.0

            # Compute implied vol via Black-76
            is_call = (right == "C")
            iv = implied_vol(opt_close, F, strike, T, is_call)

            if np.isnan(iv) or iv <= 0 or iv > 5.0:
                iv_fail += 1
                iv = np.nan

            moneyness = strike / F if F > 0 else np.nan

            rows.append({
                "timestamp": date,
                "currency": currency,
                "expiration": expiration,
                "dte": dte,
                "strike": strike,
                "right": right,
                "mid": opt_close,
                "volume": int(orow.get("volume", 0)),
                "impliedVol": iv,
                "underlyingPrice": F,
                "moneyness": moneyness,
            })

        log.info(
            "  %s: %d rows from %d options "
            "(no_meta=%d, no_underlying=%d, iv_fail=%d)",
            currency, len(rows), total, no_meta, no_underlying, iv_fail,
        )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, currency: str, data_type: str) -> Path:
        """Return cache file path for a currency and data type."""
        return self.cache_dir / f"{currency}_{data_type}.parquet"

    def _load_cache(self, currency: str, data_type: str) -> Optional[pd.DataFrame]:
        """Load cached data if it exists."""
        path = self._cache_path(currency, data_type)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        return None

    def _save_cache(self, df: pd.DataFrame, currency: str, data_type: str):
        """Save data to cache."""
        path = self._cache_path(currency, data_type)
        df.to_parquet(path, index=False)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fetch_history(
        self,
        currencies: Optional[List[str]] = None,
        start: str = "2023-02-20",
        end: str = "2026-02-20",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical FX option surfaces and compute implied vols.

        Returns a DataFrame compatible with the SABR calibration pipeline,
        matching the FXOptionCollector output format.

        Args:
            currencies: List of currency codes. Default: all 6 majors.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            use_cache: If True, load from cache if available.
        """
        if currencies is None:
            currencies = list(SYMBOL_MAP.keys())

        all_surfaces = []
        start_time = time.time()

        for ccy in currencies:
            if ccy not in SYMBOL_MAP:
                log.warning("Unknown currency: %s, skipping", ccy)
                continue

            fut_parent, opt_parent = SYMBOL_MAP[ccy]
            log.info("Processing %s (futures=%s, options=%s)...", ccy, fut_parent, opt_parent)

            # Check cache
            if use_cache:
                cached = self._load_cache(ccy, "surface")
                if cached is not None and len(cached) > 0:
                    cached_dates = pd.to_datetime(cached["timestamp"])
                    if cached_dates.min() <= pd.Timestamp(start) + pd.Timedelta(days=7):
                        log.info(
                            "  %s: loaded from cache (%d rows, %s to %s)",
                            ccy, len(cached),
                            cached_dates.min().date(), cached_dates.max().date(),
                        )
                        all_surfaces.append(cached)
                        continue

            ccy_start = time.time()

            # Step 1: Fetch option OHLCV
            log.info("  %s: fetching option OHLCV (%s to %s)...", ccy, start, end)
            opt_ohlcv = self._fetch_ohlcv(opt_parent, start, end)
            if opt_ohlcv.empty:
                log.warning("  %s: no option OHLCV data", ccy)
                continue
            log.info("  %s: %d option bars, %d unique symbols", ccy, len(opt_ohlcv), opt_ohlcv["symbol"].nunique())

            # Step 2: Fetch futures OHLCV
            log.info("  %s: fetching futures OHLCV...", ccy)
            fut_ohlcv = self._fetch_ohlcv(fut_parent, start, end)
            if fut_ohlcv.empty:
                log.warning("  %s: no futures OHLCV data", ccy)
                continue

            # Filter futures to outright contracts only (no spreads like "6EH5-6EM5")
            fut_ohlcv = fut_ohlcv[~fut_ohlcv["symbol"].str.contains("-", na=False)]
            log.info("  %s: %d futures bars, %d contracts", ccy, len(fut_ohlcv), fut_ohlcv["symbol"].nunique())

            # Step 3: Build symbol metadata from definitions
            log.info("  %s: fetching definitions...", ccy)
            meta = self._build_symbol_metadata(opt_parent, start, end)
            if meta.empty:
                log.warning("  %s: no definition metadata", ccy)
                continue

            # Step 4: Match and compute implied vols
            log.info("  %s: computing implied vols...", ccy)
            surface = self._match_and_compute_iv(opt_ohlcv, fut_ohlcv, meta, ccy)

            if not surface.empty:
                # Save to cache
                self._save_cache(surface, ccy, "surface")
                all_surfaces.append(surface)

                ccy_elapsed = time.time() - ccy_start
                iv_count = surface["impliedVol"].notna().sum()
                log.info(
                    "  %s: done in %.1fs — %d rows, %d with IV, "
                    "%d unique dates",
                    ccy, ccy_elapsed, len(surface), iv_count,
                    surface["timestamp"].nunique(),
                )

        elapsed = time.time() - start_time

        if not all_surfaces:
            log.warning("No data collected for any currency")
            return pd.DataFrame()

        combined = pd.concat(all_surfaces, ignore_index=True)
        log.info(
            "Total: %d rows, %d currencies, %.1f minutes",
            len(combined),
            combined["currency"].nunique(),
            elapsed / 60,
        )

        return combined

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        currencies: Optional[List[str]] = None,
        start: str = "2023-02-20",
        end: str = "2026-02-20",
    ) -> float:
        """Estimate the Databento API cost for a historical fetch."""
        if currencies is None:
            currencies = list(SYMBOL_MAP.keys())

        total = 0.0
        for ccy in currencies:
            if ccy not in SYMBOL_MAP:
                continue
            fut_parent, opt_parent = SYMBOL_MAP[ccy]

            for sym in [opt_parent, fut_parent]:
                try:
                    cost = self.client.metadata.get_cost(
                        dataset=self.dataset,
                        symbols=[sym],
                        stype_in="parent",
                        schema="ohlcv-1d",
                        start=start,
                        end=end,
                    )
                    print(f"  {ccy} {sym}: ${cost:.2f}")
                    total += cost
                except Exception as e:
                    print(f"  {ccy} {sym}: error — {e}")

        print(f"\n  Total estimated cost: ${total:.2f}")
        return total
