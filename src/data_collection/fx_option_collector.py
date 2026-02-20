"""
FX Option Surface Collector

Captures point-in-time snapshots of CME FX futures option surfaces from
Interactive Brokers Gateway. Collects mid prices and Greeks
(delta, gamma, vega, theta, implied volatility) for all major currency
pairs across all available maturities.

Uses options on CME currency futures (EUR→6E, GBP→6B, etc.) traded on CME.
Queries multiple futures months to build a full term-structure surface.

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
    Uses reqMktData (market data snapshots), not reqHistoricalData.
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
from ib_insync import IB, Future, FuturesOption

# ------------------------------------------------------------------
# Currency pair configuration — CME FX futures
#
# IB uses the currency code as symbol (EUR, not 6E).
# The localSymbol (6EH6, 6BM6, etc.) is assigned by IB automatically.
# ------------------------------------------------------------------

FX_PAIRS = {
    "EUR": {"symbol": "EUR", "exchange": "CME", "multiplier": 125_000},
    "GBP": {"symbol": "GBP", "exchange": "CME", "multiplier": 62_500},
    "AUD": {"symbol": "AUD", "exchange": "CME", "multiplier": 100_000},
    "CAD": {"symbol": "CAD", "exchange": "CME", "multiplier": 100_000},
    "CHF": {"symbol": "CHF", "exchange": "CME", "multiplier": 125_000},
    "JPY": {"symbol": "JPY", "exchange": "CME", "multiplier": 12_500_000},
}

ALL_CURRENCIES = list(FX_PAIRS.keys())


class FXOptionCollector:
    """Snapshot collector for CME FX futures option surfaces."""

    def __init__(
        self,
        ib: IB,
        cache_dir: str = str(
            Path.home() / "trade_data" / "ETFTrader" / "fx_options"
        ),
        batch_size: int = 90,
        batch_pause: float = 2.0,
        snapshot_timeout: float = 10,
        max_maturity_days: int = 730,
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
    # Logging
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
    # Price validation
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

    # ------------------------------------------------------------------
    # Underlying futures discovery and pricing
    # ------------------------------------------------------------------

    def _get_futures_months(self, currency: str) -> List[dict]:
        """Discover futures contract months for a currency.

        Returns list of dicts with contract, conId, expiry, price
        for each month within max_maturity_days.
        """
        pair_info = FX_PAIRS[currency]
        sym = pair_info["symbol"]
        exchange = pair_info["exchange"]
        today = datetime.now().date()
        max_date = today + timedelta(days=self.max_maturity_days)

        generic = Future(symbol=sym, exchange=exchange)
        try:
            details_list = self.ib.reqContractDetails(generic)
        except Exception as e:
            self._log(f"{currency}: failed to get futures details: {e}")
            return []

        if not details_list:
            self._log(f"{currency}: no futures contracts found for {sym} on {exchange}")
            return []

        # Filter to contracts expiring within range
        months = []
        for d in details_list:
            exp_str = d.contract.lastTradeDateOrContractMonth
            try:
                exp_date = datetime.strptime(exp_str[:8], "%Y%m%d").date()
            except ValueError:
                try:
                    exp_date = datetime.strptime(exp_str[:6], "%Y%m").date()
                except ValueError:
                    continue

            if today < exp_date <= max_date:
                months.append({
                    "contract": d.contract,
                    "conId": d.contract.conId,
                    "expiry": exp_date,
                    "localSymbol": d.contract.localSymbol,
                    "price": None,
                })

        months.sort(key=lambda x: x["expiry"])
        self._log(
            f"{currency}: {len(months)} futures months "
            f"(of {len(details_list)} total)"
        )

        # Fetch prices for all months
        if months:
            self.ib.reqMarketDataType(4)  # Frozen+delayed

            tickers = {}
            for m in months:
                ticker = self.ib.reqMktData(m["contract"], "", False, False)
                tickers[m["conId"]] = (m, ticker)

            self.ib.sleep(3)

            for con_id, (m, ticker) in tickers.items():
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
                    m["price"] = mid

                self.ib.cancelMktData(m["contract"])

            # Keep only months with prices
            priced = [m for m in months if m["price"] is not None]
            if priced:
                self._log(
                    f"{currency}: {len(priced)} months with prices, "
                    f"front={priced[0]['localSymbol']} @ {priced[0]['price']:.5f}"
                )
            else:
                self._log(f"{currency}: WARNING - no futures prices obtained")
            return priced

        return months

    # ------------------------------------------------------------------
    # Option chain discovery (across multiple futures months)
    # ------------------------------------------------------------------

    def _get_option_chains(
        self, currency: str, futures_months: List[dict]
    ) -> List[dict]:
        """Get option chains for a currency across all futures months.

        For each futures month, calls reqSecDefOptParams and picks the
        chain with the most strikes (standard monthly, typically EUU-class).

        Returns list of chain dicts, each with:
            futures_conId, futures_price, futures_expiry,
            exchange, tradingClass, multiplier,
            expirations, strikes
        """
        pair_info = FX_PAIRS[currency]
        sym = pair_info["symbol"]
        exchange = pair_info["exchange"]
        today = datetime.now().date()
        max_date = today + timedelta(days=self.max_maturity_days)

        all_chains = []

        for m in futures_months:
            raw_chains = self.ib.reqSecDefOptParams(
                sym, exchange, "FUT", m["conId"]
            )

            if not raw_chains:
                continue

            # Pick the standard chain (most strikes = widest strike range)
            best = max(raw_chains, key=lambda c: len(c.strikes))

            # Filter expirations
            valid_exps = sorted(
                exp for exp in best.expirations
                if datetime.strptime(exp, "%Y%m%d").date() > today
                and datetime.strptime(exp, "%Y%m%d").date() <= max_date
            )

            if not valid_exps:
                continue

            # Filter strikes around futures price
            price = m["price"]
            lo = price * (1 - self.strike_filter_pct)
            hi = price * (1 + self.strike_filter_pct)
            valid_strikes = sorted(
                s for s in best.strikes if lo <= s <= hi
            )

            if not valid_strikes:
                continue

            all_chains.append({
                "futures_conId": m["conId"],
                "futures_price": m["price"],
                "futures_expiry": m["expiry"],
                "futures_localSymbol": m["localSymbol"],
                "exchange": best.exchange,
                "tradingClass": best.tradingClass,
                "multiplier": best.multiplier,
                "expirations": valid_exps,
                "strikes": valid_strikes,
            })

        total_exps = sum(len(c["expirations"]) for c in all_chains)
        total_strikes = max(
            (len(c["strikes"]) for c in all_chains), default=0
        )
        self._log(
            f"{currency}: {len(all_chains)} futures months with options, "
            f"{total_exps} total expirations, "
            f"up to {total_strikes} strikes"
        )
        return all_chains

    # ------------------------------------------------------------------
    # Contract building
    # ------------------------------------------------------------------

    def _build_option_contracts(
        self,
        currency: str,
        chains: List[dict],
        batch_qualify_size: int = 50,
    ) -> List[Tuple[FuturesOption, float]]:
        """Build and qualify FuturesOption contracts for one currency.

        Returns list of (qualified_contract, underlying_price) tuples.
        """
        pair_info = FX_PAIRS[currency]
        sym = pair_info["symbol"]

        raw_contracts = []  # (contract, underlying_price)
        for chain in chains:
            for exp in chain["expirations"]:
                for strike in chain["strikes"]:
                    for right in ["C", "P"]:
                        opt = FuturesOption(
                            symbol=sym,
                            lastTradeDateOrContractMonth=exp,
                            strike=strike,
                            right=right,
                            exchange=chain["exchange"],
                            multiplier=str(chain["multiplier"]),
                            tradingClass=chain["tradingClass"],
                        )
                        raw_contracts.append(
                            (opt, chain["futures_price"])
                        )

        if not raw_contracts:
            return []

        # Qualify in batches
        qualified = []
        for i in range(0, len(raw_contracts), batch_qualify_size):
            batch = raw_contracts[i : i + batch_qualify_size]
            contracts_only = [c for c, _ in batch]
            price_map = {id(c): p for c, p in batch}
            try:
                result = self.ib.qualifyContracts(*contracts_only)
                for c in result:
                    if c.conId > 0:
                        # Find matching price from original batch
                        idx = contracts_only.index(c)
                        qualified.append((c, batch[idx][1]))
            except Exception as e:
                self._log(
                    f"{currency}: qualification error batch "
                    f"{i // batch_qualify_size + 1}: {e}"
                )
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
        contracts_with_prices: List[Tuple[FuturesOption, float]],
    ) -> pd.DataFrame:
        """Collect market data snapshot for all option contracts of one currency.

        Args:
            contracts_with_prices: List of (contract, underlying_price) tuples.

        Returns DataFrame with full option surface.
        """
        if not contracts_with_prices:
            return pd.DataFrame()

        now = datetime.now()
        rows = []
        total_batches = math.ceil(
            len(contracts_with_prices) / self.batch_size
        )

        self.ib.reqMarketDataType(4)

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(contracts_with_prices))
            batch = contracts_with_prices[start:end]

            self._log(
                f"{currency}: batch {batch_idx + 1}/{total_batches} "
                f"({len(batch)} contracts)"
            )

            try:
                tickers = []
                for con, _ in batch:
                    ticker = self.ib.reqMktData(con, "", False, False)
                    tickers.append(ticker)

                self.ib.sleep(self.snapshot_timeout)

                for (con, underlying_price), ticker in zip(batch, tickers):
                    contract = ticker.contract
                    greeks = self._extract_greeks(ticker)

                    # Compute mid price: prefer bid/ask midpoint,
                    # fall back to last, then close
                    mid = float("nan")
                    bid = ticker.bid if self._valid_price(ticker.bid) else None
                    ask = ticker.ask if self._valid_price(ticker.ask) else None
                    if bid is not None and ask is not None:
                        mid = (bid + ask) / 2
                    elif self._valid_price(ticker.last):
                        mid = ticker.last
                    elif self._valid_price(ticker.close):
                        mid = ticker.close

                    volume = ticker.volume if self._valid_price(ticker.volume) else 0

                    exp_str = contract.lastTradeDateOrContractMonth
                    exp_date = datetime.strptime(exp_str[:8], "%Y%m%d")
                    dte = (exp_date.date() - now.date()).days

                    rows.append({
                        "timestamp": now,
                        "currency": currency,
                        "expiration": exp_str[:8],
                        "dte": dte,
                        "strike": contract.strike,
                        "right": contract.right,
                        "mid": mid,
                        "volume": int(volume),
                        "impliedVol": greeks["impliedVol"],
                        "delta": greeks["delta"],
                        "gamma": greeks["gamma"],
                        "vega": greeks["vega"],
                        "theta": greeks["theta"],
                        "underlyingPrice": underlying_price,
                        "moneyness": contract.strike / underlying_price
                        if underlying_price > 0
                        else float("nan"),
                    })

                for con, _ in batch:
                    try:
                        self.ib.cancelMktData(con)
                    except Exception:
                        pass

            except Exception as e:
                self._log(f"{currency}: batch {batch_idx + 1} error: {e}")

            if batch_idx < total_batches - 1:
                time.sleep(self.batch_pause)

        df = pd.DataFrame(rows)

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
        """Append snapshot to per-currency parquet file."""
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
        return (
            f"{currency}: saved {len(df)} new rows "
            f"(total {len(combined)} in file)"
        )

    # ------------------------------------------------------------------
    # Single currency collection
    # ------------------------------------------------------------------

    def collect_currency(
        self, currency: str
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """Collect option surface for a single currency.

        Returns (DataFrame, log_message) or (None, error_message).
        """
        futures_months = self._get_futures_months(currency)
        if not futures_months:
            return None, f"{currency}: no futures months available"

        chains = self._get_option_chains(currency, futures_months)
        if not chains:
            return None, f"{currency}: no option chains available"

        contracts = self._build_option_contracts(currency, chains)
        if not contracts:
            return None, f"{currency}: no contracts qualified"

        df = self._collect_currency_snapshot(currency, contracts)
        if df.empty:
            return None, f"{currency}: empty snapshot"

        msg = self._save_snapshot(currency, df)
        self._log(msg)

        front_price = futures_months[0]["price"]
        self._append_manifest({
            "currency": currency,
            "status": "ok",
            "rows": len(df),
            "contracts": len(contracts),
            "iv_count": int(df["impliedVol"].notna().sum()),
            "delta_count": int(df["delta"].notna().sum()),
            "spot": front_price,
            "timestamp": datetime.now().isoformat(),
        })

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
        print("FX OPTION SURFACE COLLECTION (CME Futures Options)")
        print("=" * 60)
        start_time = time.time()

        # Phase 1: Scan cache
        self._log_summary("")
        self._log_summary("Phase 1: Scanning cache...")
        cache_states = {}
        for ccy in self.currencies:
            state, last_ts = self._check_cache(ccy)
            cache_states[ccy] = (state, last_ts)
            ts_str = (
                last_ts.strftime("%Y-%m-%d %H:%M") if last_ts else "never"
            )
            self._log(f"{ccy}: {state} (last: {ts_str})")

        to_collect = [
            ccy
            for ccy in self.currencies
            if not (skip_current and cache_states[ccy][0] == "current")
        ]
        skipped = [
            ccy for ccy in self.currencies if ccy not in to_collect
        ]

        self._log_summary("")
        self._log_summary(
            f"  CURRENT (skip): {len(skipped)} "
            f"[{', '.join(skipped) or 'none'}]"
        )
        self._log_summary(
            f"  TO COLLECT:     {len(to_collect)} "
            f"[{', '.join(to_collect) or 'none'}]"
        )

        if not to_collect:
            self._log_summary("")
            self._log_summary(
                "Nothing to collect — all currencies current."
            )
            return self.build_surface_matrix(), pd.DataFrame()

        # Phase 2-4: Collect each currency
        results = []
        all_dfs = []

        for i, ccy in enumerate(to_collect):
            self._log_summary("")
            self._log(f"--- {ccy} ({i + 1}/{len(to_collect)}) ---")

            try:
                # Phase 2: Futures months + prices
                futures_months = self._get_futures_months(ccy)
                if not futures_months:
                    results.append({
                        "currency": ccy, "status": "no_futures", "rows": 0,
                    })
                    continue

                # Phase 3: Option chains
                chains = self._get_option_chains(ccy, futures_months)
                if not chains:
                    results.append({
                        "currency": ccy, "status": "no_chain", "rows": 0,
                    })
                    continue

                # Build contracts
                contracts = self._build_option_contracts(ccy, chains)
                if not contracts:
                    results.append({
                        "currency": ccy, "status": "no_contracts",
                        "rows": 0,
                    })
                    continue

                # Phase 4: Collect snapshot
                df = self._collect_currency_snapshot(ccy, contracts)

                if df.empty:
                    results.append({
                        "currency": ccy, "status": "empty", "rows": 0,
                    })
                    continue

                msg = self._save_snapshot(ccy, df)
                self._log(msg)
                all_dfs.append(df)

                front_price = futures_months[0]["price"]
                results.append({
                    "currency": ccy,
                    "status": "ok",
                    "rows": len(df),
                    "contracts": len(contracts),
                    "iv_count": int(df["impliedVol"].notna().sum()),
                    "delta_count": int(df["delta"].notna().sum()),
                    "spot": front_price,
                })

                self._append_manifest({
                    "currency": ccy,
                    "status": "ok",
                    "rows": len(df),
                    "contracts": len(contracts),
                    "iv_count": int(df["impliedVol"].notna().sum()),
                    "delta_count": int(df["delta"].notna().sum()),
                    "spot": front_price,
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as e:
                self._log(f"{ccy}: ERROR - {e}")
                results.append({
                    "currency": ccy, "status": f"error: {e}", "rows": 0,
                })

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
        total_iv = (
            sum(df["impliedVol"].notna().sum() for df in all_dfs)
            if all_dfs else 0
        )
        total_delta = (
            sum(df["delta"].notna().sum() for df in all_dfs)
            if all_dfs else 0
        )
        self._log_summary(f"  Total rows: {total_rows}")
        self._log_summary(f"  Rows with IV: {total_iv}")
        self._log_summary(f"  Rows with delta: {total_delta}")

        combined = (
            pd.concat(all_dfs, ignore_index=True)
            if all_dfs else pd.DataFrame()
        )
        return combined, results_df

    # ------------------------------------------------------------------
    # Build combined surface from cache
    # ------------------------------------------------------------------

    def build_surface_matrix(self) -> pd.DataFrame:
        """Load all cached per-currency parquets and combine."""
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
