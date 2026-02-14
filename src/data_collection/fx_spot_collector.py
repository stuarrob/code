"""
FX Spot Data Collector

Collects daily FX spot rates for major currency pairs via IB Gateway
using reqHistoricalData with MIDPOINT. Stores per-pair parquet files
that are appended over time.

Usage:
    from ib_insync import IB
    from data_collection.fx_spot_collector import FXSpotCollector

    ib = IB()
    ib.connect("127.0.0.1", 4001, clientId=7, readonly=True)

    collector = FXSpotCollector(ib)
    spot_df = collector.collect_all()
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Forex

# Same pairs as FX option collector
FX_PAIRS = {
    "EUR": "EURUSD",
    "GBP": "GBPUSD",
    "AUD": "AUDUSD",
    "CAD": "USDCAD",
    "CHF": "USDCHF",
    "JPY": "USDJPY",
}


class FXSpotCollector:
    """Collects daily FX spot midpoint rates via IB historical data."""

    def __init__(
        self,
        ib: IB,
        cache_dir: str = str(Path.home() / "trade_data" / "ETFTrader" / "fx_spot"),
        duration: str = "1 Y",
        currencies: Optional[List[str]] = None,
    ):
        """
        Args:
            ib: Connected IB instance.
            cache_dir: Directory for per-pair parquet files.
            duration: IB duration string for historical data request.
            currencies: List of currency codes, or None for all 6 majors.
        """
        self.ib = ib
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration
        self.currencies = currencies or list(FX_PAIRS.keys())

    def collect_pair(self, currency: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Collect daily spot history for a single currency pair.

        Returns:
            (DataFrame with date/close/high/low/open/volume columns, status message)
        """
        pair = FX_PAIRS.get(currency)
        if not pair:
            return None, f"{currency}: unknown pair"

        contract = Forex(pair, exchange="IDEALPRO")
        try:
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return None, f"{pair}: failed to qualify"
        except Exception as e:
            return None, f"{pair}: qualify error — {e}"

        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=self.duration,
                barSizeSetting="1 day",
                whatToShow="MIDPOINT",
                useRTH=True,
                formatDate=1,
            )
        except Exception as e:
            return None, f"{pair}: reqHistoricalData error — {e}"

        if not bars:
            return None, f"{pair}: no bars returned"

        df = pd.DataFrame([{
            "date": b.date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": getattr(b, "volume", 0),
        } for b in bars])

        df["date"] = pd.to_datetime(df["date"])
        df["currency"] = currency
        df["pair"] = pair

        # Save per-pair parquet (merge with existing)
        path = self.cache_dir / f"{pair}.parquet"
        df = self._merge_and_save(df, path)

        return df, f"{pair}: {len(df)} bars ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})"

    def collect_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect spot history for all configured currency pairs.

        Returns:
            (combined DataFrame, results summary DataFrame)
        """
        print(f"\n{'='*60}")
        print(f"FX Spot Collection — {len(self.currencies)} pairs")
        print(f"{'='*60}\n")

        all_dfs = []
        results = []

        for ccy in self.currencies:
            df, msg = self.collect_pair(ccy)
            print(f"  {msg}")
            results.append({
                "currency": ccy,
                "pair": FX_PAIRS.get(ccy, ""),
                "bars": len(df) if df is not None else 0,
                "status": "ok" if df is not None else "failed",
            })
            if df is not None:
                all_dfs.append(df)
            time.sleep(1)  # Brief pause between requests

        results_df = pd.DataFrame(results)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            print(f"\nTotal: {len(combined)} bars across {len(all_dfs)} pairs")
            return combined, results_df

        print("\nNo data collected")
        return pd.DataFrame(), results_df

    def build_spot_matrix(self) -> pd.DataFrame:
        """Load all cached parquets and build a date x currency matrix.

        Returns:
            DataFrame with date index and currency columns (close prices).
        """
        dfs = []
        for ccy, pair in FX_PAIRS.items():
            if ccy not in self.currencies:
                continue
            path = self.cache_dir / f"{pair}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df["date"] = pd.to_datetime(df["date"])
                dfs.append(df[["date", "currency", "close"]].copy())

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        matrix = combined.pivot_table(index="date", columns="currency", values="close")
        matrix = matrix.sort_index()
        return matrix

    def _merge_and_save(self, new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """Merge new data with existing parquet, dedup by date, save."""
        if path.exists():
            existing = pd.read_parquet(path)
            existing["date"] = pd.to_datetime(existing["date"])
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
        else:
            combined = new_df.sort_values("date").reset_index(drop=True)

        combined.to_parquet(path, index=False)
        return combined
