"""
FX Option Surface Collector

Captures point-in-time snapshots of PHLX FX option surfaces from
Interactive Brokers Gateway. Collects bid/ask/mid prices and Greeks
(delta, gamma, vega, theta, implied volatility) for all major currency
pairs with maturities up to 1 year.

Each run produces a snapshot that is appended to per-currency parquet
files, building a historical volatility surface over time.

Usage:
    from ib_insync import IB
    from data_collection.fx_option_collector import FXOptionCollector

    ib = IB()
    ib.connect("127.0.0.1", 4001, clientId=6, readonly=True)

    collector = FXOptionCollector(ib)
    surface_df, results_df = collector.collect_all()

Rate limits:
    Uses reqTickers (market data snapshots), not reqHistoricalData.
    IB allows ~100 concurrent market data lines.
    Requests are batched at 90, with 2s pause between batches.
    Full collection (~2,880 contracts) takes ~10-15 minutes.
"""

import csv
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, Forex, Option

# ------------------------------------------------------------------
# Currency pair configuration
# ------------------------------------------------------------------

FX_PAIRS = {
    "EUR": {"pair": "EURUSD", "contract_size": 10_000},
    "GBP": {"pair": "GBPUSD", "contract_size": 10_000},
    "AUD": {"pair": "AUDUSD", "contract_size": 10_000},
    "CAD": {"pair": "USDCAD", "contract_size": 10_000},
    "CHF": {"pair": "USDCHF", "contract_size": 10_000},
    "JPY": {"pair": "USDJPY", "contract_size": 1_000_000},
}

ALL_CURRENCIES = list(FX_PAIRS.keys())


class FXOptionCollector:
    """Snapshot collector for PHLX FX option surfaces."""

    def __init__(
        self,
        ib: IB,
        cache_dir: str = str(
            Path.home() / "trade_data" / "ETFTrader" / "fx_options"
        ),
        batch_size: int = 90,
        batch_pause: float = 2.0,
        snapshot_timeout: float = 10,
        max_maturity_days: int = 365,
        strike_filter_pct: float = 0.15,
        currencies: Optional[List[str]] = None,
    ):
        self.ib = ib
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.batch_pause = batch_pause
        self.snapshot_timeout = snapshot_timeout
        self.max_maturity_days = max_maturity_days
        self.strike_filter_pct = strike_filter_pct
        self.currencies = currencies or ALL_CURRENCIES
        self.manifest_path = self.cache_dir / "manifest.csv"

    # ------------------------------------------------------------------
    # Logging (same pattern as IBDataCollector)
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        """Print a timestamped log line."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {msg}", flush=True)

    def _log_summary(self, msg: str):
        """Print a summary line without timestamp."""
        print(f"  {msg}", flush=True)

    # ------------------------------------------------------------------
    # Manifest logging
    # ------------------------------------------------------------------

    def _append_manifest(self, row: dict):
        """Append a result row to the manifest CSV."""
        write_header = not self.manifest_path.exists()
        with open(self.manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _check_cache(self, currency: str) -> Tuple[str, Optional[pd.Timestamp]]:
        """Check cache state for a currency.

        Returns (state, last_snapshot_time) where state is:
            "current"  - snapshot taken today
            "stale"    - has data but not from today
            "missing"  - no cached data
        """
        path = self.cache_dir / f"{currency}.parquet"
        if not path.exists():
            return "missing", None

        try:
            df = pd.read_parquet(path)
            if df.empty or "timestamp" not in df.columns:
                return "missing", None
            last_ts = pd.Timestamp(df["timestamp"].max())
            if last_ts.date() == datetime.now().date():
                return "current", last_ts
            return "stale", last_ts
        except Exception:
            return "missing", None

    # ------------------------------------------------------------------
    # Spot price retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_price(val) -> bool:
        """Check if a price value is valid (not None, not NaN, not -1)."""
        if val is None:
            return False
        try:
            return not math.isnan(val) and val > 0
        except (TypeError, ValueError):
            return False

    def _get_spot_prices(self) -> Dict[str, float]:
        """Get current spot prices for configured currency pairs.

        Returns dict mapping currency code to spot rate.
        """
        spots = {}
        contracts = []
        currency_order = []

        for ccy in self.currencies:
            pair_info = FX_PAIRS[ccy]
            pair = pair_info["pair"]
            fx = Forex(pair)
            contracts.append(fx)
            currency_order.append(ccy)

        qualified = self.ib.qualifyContracts(*contracts)
        if not qualified:
            self._log("WARNING: Could not qualify any FX contracts")
            return spots

        # Request frozen+delayed data (works outside market hours)
        self.ib.reqMarketDataType(4)

        # Use reqMktData + ib.sleep for proper event processing
        tickers = []
        for con in qualified:
            ticker = self.ib.reqMktData(con, "", False, False)
            tickers.append(ticker)

        self.ib.sleep(3)  # Process events for up to 3 seconds

        for ccy, con, ticker in zip(currency_order, qualified, tickers):
            mid = None
            if self._valid_price(ticker.bid) and self._valid_price(ticker.ask):
                mid = (ticker.bid + ticker.ask) / 2
            elif self._valid_price(ticker.last):
                mid = ticker.last
            elif self._valid_price(ticker.close):
                mid = ticker.close
            elif self._valid_price(ticker.marketPrice()):
                mid = ticker.marketPrice()

            if mid and mid > 0:
                spots[ccy] = mid
                self._log(f"{ccy}: spot = {mid:.5f}")
            else:
                self._log(
                    f"{ccy}: WARNING - no spot price "
                    f"(bid={ticker.bid} ask={ticker.ask} last={ticker.last} close={ticker.close})"
                )

            self.ib.cancelMktData(con)

        return spots

    # ------------------------------------------------------------------
    # Option chain discovery
    # ------------------------------------------------------------------

    def _get_option_chains(
        self, spots: Dict[str, float]
    ) -> Dict[str, dict]:
        """Get option chain parameters for each currency pair.

        Calls reqSecDefOptParams for each currency and filters for PHLX.
        Returns dict with expirations, strikes, tradingClass per currency.
        """
        chains = {}
        today = datetime.now().date()
        max_date = today + timedelta(days=self.max_maturity_days)

        for ccy in self.currencies:
            if ccy not in spots:
                self._log(f"{ccy}: skipping chain (no spot price)")
                continue

            pair_info = FX_PAIRS[ccy]
            pair = pair_info["pair"]

            # Qualify the underlying to get conId
            fx = Forex(pair)
            qualified = self.ib.qualifyContracts(fx)
            if not qualified:
                self._log(f"{ccy}: could not qualify underlying {pair}")
                continue

            con_id = fx.conId

            # Request option chain parameters
            all_chains = self.ib.reqSecDefOptParams(
                ccy, "", "CASH", con_id
            )

            if not all_chains:
                # Try with the full pair symbol
                all_chains = self.ib.reqSecDefOptParams(
                    pair[:3], "", "CASH", con_id
                )

            if not all_chains:
                self._log(f"{ccy}: no option chains returned")
                continue

            # Filter for PHLX exchange
            phlx_chains = [c for c in all_chains if c.exchange == "PHLX"]

            if not phlx_chains:
                # Try any available exchange
                exchanges = set(c.exchange for c in all_chains)
                self._log(
                    f"{ccy}: no PHLX chain found. Available: {exchanges}"
                )
                # Use the first available exchange
                if all_chains:
                    phlx_chains = [all_chains[0]]
                    self._log(f"{ccy}: falling back to {all_chains[0].exchange}")
                else:
                    continue

            chain = phlx_chains[0]

            # Filter expirations within max_maturity_days
            valid_exps = sorted(
                exp
                for exp in chain.expirations
                if datetime.strptime(exp, "%Y%m%d").date() <= max_date
                and datetime.strptime(exp, "%Y%m%d").date() > today
            )

            # Filter strikes around spot
            spot = spots[ccy]
            lo = spot * (1 - self.strike_filter_pct)
            hi = spot * (1 + self.strike_filter_pct)
            valid_strikes = sorted(s for s in chain.strikes if lo <= s <= hi)

            chains[ccy] = {
                "conId": con_id,
                "exchange": chain.exchange,
                "tradingClass": chain.tradingClass,
                "multiplier": chain.multiplier,
                "expirations": valid_exps,
                "strikes": valid_strikes,
                "all_strikes": len(chain.strikes),
                "all_expirations": len(chain.expirations),
            }

            self._log(
                f"{ccy}: {len(valid_exps)} expirations, "
                f"{len(valid_strikes)} strikes (of {len(chain.strikes)} total) "
                f"on {chain.exchange}, class={chain.tradingClass}"
            )

        return chains

    # ------------------------------------------------------------------
    # Contract building
    # ------------------------------------------------------------------

    def _build_option_contracts(
        self,
        currency: str,
        chain: dict,
        batch_qualify_size: int = 50,
    ) -> List[Option]:
        """Build and qualify Option contracts for one currency.

        Returns list of qualified Option contracts.
        """
        exchange = chain["exchange"]
        trading_class = chain["tradingClass"]
        multiplier = chain.get("multiplier", "100")

        raw_contracts = []
        for exp in chain["expirations"]:
            for strike in chain["strikes"]:
                for right in ["C", "P"]:
                    opt = Option(
                        symbol=currency,
                        lastTradeDateOrContractMonth=exp,
                        strike=strike,
                        right=right,
                        exchange=exchange,
                        multiplier=str(multiplier),
                        tradingClass=trading_class,
                    )
                    raw_contracts.append(opt)

        if not raw_contracts:
            return []

        # Qualify in batches
        qualified = []
        for i in range(0, len(raw_contracts), batch_qualify_size):
            batch = raw_contracts[i : i + batch_qualify_size]
            try:
                result = self.ib.qualifyContracts(*batch)
                qualified.extend([c for c in result if c.conId > 0])
            except Exception as e:
                self._log(f"{currency}: qualification error batch {i}: {e}")
            time.sleep(0.5)

        self._log(
            f"{currency}: {len(qualified)} contracts qualified "
            f"(of {len(raw_contracts)} attempted)"
        )
        return qualified

    # ------------------------------------------------------------------
    # Greek extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_greeks(ticker) -> dict:
        """Extract Greeks from a ticker object.

        Prefers modelGreeks, falls back to bidGreeks/askGreeks.
        Returns dict with impliedVol, delta, gamma, vega, theta.
        """
        nan = float("nan")
        result = {
            "impliedVol": nan,
            "delta": nan,
            "gamma": nan,
            "vega": nan,
            "theta": nan,
        }

        greeks = None

        # Prefer model greeks
        if hasattr(ticker, "modelGreeks") and ticker.modelGreeks:
            greeks = ticker.modelGreeks
        elif hasattr(ticker, "lastGreeks") and ticker.lastGreeks:
            greeks = ticker.lastGreeks
        elif hasattr(ticker, "bidGreeks") and ticker.bidGreeks:
            greeks = ticker.bidGreeks
        elif hasattr(ticker, "askGreeks") and ticker.askGreeks:
            greeks = ticker.askGreeks

        if greeks:
            for field in result:
                val = getattr(greeks, field, nan)
                if val is not None:
                    result[field] = val

        return result

    # ------------------------------------------------------------------
    # Snapshot collection for a single currency
    # ------------------------------------------------------------------

    def _collect_currency_snapshot(
        self,
        currency: str,
        contracts: List[Option],
        spot_price: float,
    ) -> pd.DataFrame:
        """Collect market data snapshot for all option contracts of one currency.

        Requests tickers in batches, extracts prices and Greeks.
        Returns DataFrame with full option surface.
        """
        if not contracts:
            return pd.DataFrame()

        now = datetime.now()
        rows = []
        total_batches = math.ceil(len(contracts) / self.batch_size)

        # Request frozen+delayed data (works outside market hours)
        self.ib.reqMarketDataType(4)

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(contracts))
            batch = contracts[start:end]

            self._log(
                f"{currency}: batch {batch_idx + 1}/{total_batches} "
                f"({len(batch)} contracts)"
            )

            try:
                # Use reqMktData for each contract, then ib.sleep to process
                tickers = []
                for con in batch:
                    ticker = self.ib.reqMktData(con, "", False, False)
                    tickers.append(ticker)

                # Wait for data to populate (ib.sleep processes IB events)
                self.ib.sleep(self.snapshot_timeout)

                for ticker in tickers:
                    contract = ticker.contract
                    greeks = self._extract_greeks(ticker)

                    # Extract prices with validation
                    bid = ticker.bid if self._valid_price(ticker.bid) else float("nan")
                    ask = ticker.ask if self._valid_price(ticker.ask) else float("nan")
                    last = ticker.last if self._valid_price(ticker.last) else float("nan")
                    close = ticker.close if self._valid_price(ticker.close) else float("nan")

                    mid = float("nan")
                    if not math.isnan(bid) and not math.isnan(ask):
                        mid = (bid + ask) / 2
                    elif not math.isnan(last):
                        mid = last
                    elif not math.isnan(close):
                        mid = close

                    volume = ticker.volume if self._valid_price(ticker.volume) else 0

                    # Calculate DTE
                    exp_str = contract.lastTradeDateOrContractMonth
                    exp_date = datetime.strptime(exp_str[:8], "%Y%m%d")
                    dte = (exp_date.date() - now.date()).days

                    rows.append(
                        {
                            "timestamp": now,
                            "currency": currency,
                            "expiration": exp_str[:8],
                            "dte": dte,
                            "strike": contract.strike,
                            "right": contract.right,
                            "bid": bid,
                            "ask": ask,
                            "last": last,
                            "mid": mid,
                            "volume": int(volume),
                            "impliedVol": greeks["impliedVol"],
                            "delta": greeks["delta"],
                            "gamma": greeks["gamma"],
                            "vega": greeks["vega"],
                            "theta": greeks["theta"],
                            "underlyingPrice": spot_price,
                            "moneyness": contract.strike / spot_price
                            if spot_price > 0
                            else float("nan"),
                        }
                    )

                # Cancel market data for this batch
                for con in batch:
                    try:
                        self.ib.cancelMktData(con)
                    except Exception:
                        pass

            except Exception as e:
                self._log(f"{currency}: batch {batch_idx + 1} error: {e}")

            # Pause between batches
            if batch_idx < total_batches - 1:
                time.sleep(self.batch_pause)

        df = pd.DataFrame(rows)

        # Count non-NaN implied vols
        if not df.empty:
            iv_count = df["impliedVol"].notna().sum()
            delta_count = df["delta"].notna().sum()
            self._log(
                f"{currency}: {len(df)} rows, "
                f"{iv_count} with IV, {delta_count} with delta"
            )

        return df

    # ------------------------------------------------------------------
    # Save snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(self, currency: str, df: pd.DataFrame) -> str:
        """Append snapshot to per-currency parquet file.

        Reads existing file, concatenates new data, overwrites.
        Returns log message.
        """
        path = self.cache_dir / f"{currency}.parquet"

        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, df], ignore_index=True)
            except Exception:
                combined = df
        else:
            combined = df

        combined.to_parquet(path, index=False)
        return f"{currency}: saved {len(df)} new rows (total {len(combined)} in file)"

    # ------------------------------------------------------------------
    # Single currency collection
    # ------------------------------------------------------------------

    def collect_currency(
        self, currency: str
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """Collect option surface for a single currency.

        Returns (DataFrame, log_message) or (None, error_message).
        """
        # Get spot price
        spots = self._get_spot_prices()
        if currency not in spots:
            return None, f"{currency}: no spot price available"

        spot = spots[currency]

        # Get option chain
        chains = self._get_option_chains(spots)
        if currency not in chains:
            return None, f"{currency}: no option chain available"

        chain = chains[currency]

        # Build contracts
        contracts = self._build_option_contracts(currency, chain)
        if not contracts:
            return None, f"{currency}: no contracts qualified"

        # Collect snapshot
        df = self._collect_currency_snapshot(currency, contracts, spot)
        if df.empty:
            return None, f"{currency}: empty snapshot"

        # Save
        msg = self._save_snapshot(currency, df)
        self._log(msg)

        # Log manifest
        self._append_manifest(
            {
                "currency": currency,
                "status": "ok",
                "rows": len(df),
                "contracts": len(contracts),
                "iv_count": int(df["impliedVol"].notna().sum()),
                "delta_count": int(df["delta"].notna().sum()),
                "spot": spot,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return df, msg

    # ------------------------------------------------------------------
    # Main entry point: collect all currencies
    # ------------------------------------------------------------------

    def collect_all(
        self,
        skip_current: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect FX option surfaces for all configured currencies.

        Returns (combined_surface_df, results_df).
        """
        print("=" * 60)
        print("FX OPTION SURFACE COLLECTION")
        print("=" * 60)
        start_time = time.time()

        # Phase 1: Scan cache
        self._log_summary("")
        self._log_summary("Phase 1: Scanning cache...")
        cache_states = {}
        for ccy in self.currencies:
            state, last_ts = self._check_cache(ccy)
            cache_states[ccy] = (state, last_ts)
            ts_str = last_ts.strftime("%Y-%m-%d %H:%M") if last_ts else "never"
            self._log(f"{ccy}: {state} (last: {ts_str})")

        to_collect = [
            ccy
            for ccy in self.currencies
            if not (skip_current and cache_states[ccy][0] == "current")
        ]
        skipped = [ccy for ccy in self.currencies if ccy not in to_collect]

        self._log_summary("")
        self._log_summary(
            f"  CURRENT (skip): {len(skipped)} [{', '.join(skipped) or 'none'}]"
        )
        self._log_summary(
            f"  TO COLLECT:     {len(to_collect)} [{', '.join(to_collect) or 'none'}]"
        )

        if not to_collect:
            self._log_summary("")
            self._log_summary("Nothing to collect — all currencies current.")
            return self.build_surface_matrix(), pd.DataFrame()

        # Phase 2: Get spot prices
        self._log_summary("")
        self._log_summary("Phase 2: Getting spot prices...")
        spots = self._get_spot_prices()

        # Phase 3: Get option chains
        self._log_summary("")
        self._log_summary("Phase 3: Discovering option chains...")
        chains = self._get_option_chains(spots)

        # Phase 4: Collect snapshots
        self._log_summary("")
        self._log_summary("Phase 4: Collecting option surfaces...")
        results = []
        all_dfs = []

        for i, ccy in enumerate(to_collect):
            self._log_summary("")
            self._log(f"--- {ccy} ({i + 1}/{len(to_collect)}) ---")

            if ccy not in spots:
                self._log(f"{ccy}: SKIP (no spot price)")
                results.append(
                    {"currency": ccy, "status": "no_spot", "rows": 0}
                )
                continue

            if ccy not in chains:
                self._log(f"{ccy}: SKIP (no option chain)")
                results.append(
                    {"currency": ccy, "status": "no_chain", "rows": 0}
                )
                continue

            spot = spots[ccy]
            chain = chains[ccy]

            # Build contracts
            contracts = self._build_option_contracts(ccy, chain)
            if not contracts:
                self._log(f"{ccy}: SKIP (no contracts qualified)")
                results.append(
                    {"currency": ccy, "status": "no_contracts", "rows": 0}
                )
                continue

            # Collect snapshot
            try:
                df = self._collect_currency_snapshot(ccy, contracts, spot)

                if df.empty:
                    results.append(
                        {"currency": ccy, "status": "empty", "rows": 0}
                    )
                    continue

                # Save
                msg = self._save_snapshot(ccy, df)
                self._log(msg)
                all_dfs.append(df)

                results.append(
                    {
                        "currency": ccy,
                        "status": "ok",
                        "rows": len(df),
                        "contracts": len(contracts),
                        "iv_count": int(df["impliedVol"].notna().sum()),
                        "delta_count": int(df["delta"].notna().sum()),
                        "spot": spot,
                    }
                )

                # Log manifest
                self._append_manifest(
                    {
                        "currency": ccy,
                        "status": "ok",
                        "rows": len(df),
                        "contracts": len(contracts),
                        "iv_count": int(df["impliedVol"].notna().sum()),
                        "delta_count": int(df["delta"].notna().sum()),
                        "spot": spot,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                self._log(f"{ccy}: ERROR - {e}")
                results.append(
                    {"currency": ccy, "status": f"error: {e}", "rows": 0}
                )

            if progress_callback:
                progress_callback(i + 1, len(to_collect), ccy)

        # Summary
        elapsed = time.time() - start_time
        results_df = pd.DataFrame(results)

        self._log_summary("")
        print("=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        self._log_summary(f"  Time: {elapsed / 60:.1f} minutes")
        self._log_summary(f"  Currencies collected: {len(all_dfs)}")
        self._log_summary(f"  Skipped (current): {len(skipped)}")
        total_rows = sum(len(df) for df in all_dfs)
        total_iv = sum(df["impliedVol"].notna().sum() for df in all_dfs) if all_dfs else 0
        total_delta = sum(df["delta"].notna().sum() for df in all_dfs) if all_dfs else 0
        self._log_summary(f"  Total rows: {total_rows}")
        self._log_summary(f"  Rows with IV: {total_iv}")
        self._log_summary(f"  Rows with delta: {total_delta}")

        # Combine all new snapshots
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        return combined, results_df

    # ------------------------------------------------------------------
    # Build combined surface from cache
    # ------------------------------------------------------------------

    def build_surface_matrix(self) -> pd.DataFrame:
        """Load all cached per-currency parquets and combine.

        Returns a single DataFrame with all currencies and snapshots.
        """
        dfs = []
        for path in sorted(self.cache_dir.glob("*.parquet")):
            if path.stem == "manifest":
                continue
            try:
                df = pd.read_parquet(path)
                dfs.append(df)
            except Exception as e:
                self._log(f"Warning: could not read {path.name}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        self._log_summary(
            f"Surface matrix: {len(combined)} rows across "
            f"{combined['currency'].nunique()} currencies"
        )
        return combined
