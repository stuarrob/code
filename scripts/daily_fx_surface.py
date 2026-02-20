#!/usr/bin/env python3
"""
Daily FX Option Surface Collection + SABR Calibration

Connects to IB Gateway, collects the full CME FX futures option surface
for all 6 major currencies, fits SABR parameters per maturity, and
collects spot rates. Designed to run once daily via cron.

Outputs:
    ~/trade_data/ETFTrader/fx_options/{CCY}.parquet   — raw surface (appended)
    ~/trade_data/ETFTrader/fx_spot/{PAIR}.parquet     — spot rates (appended)
    ~/trade_data/ETFTrader/fx_sabr/sabr_params.parquet — SABR params (appended)
    ~/trade_data/ETFTrader/fx_sabr/sabr_latest.csv    — today's params (overwritten)

Idempotent: skips currencies already collected today (via skip_current).

Usage:
    python scripts/daily_fx_surface.py              # collect + fit
    python scripts/daily_fx_surface.py --fit-only   # refit from cached surfaces
    python scripts/daily_fx_surface.py --dry-run    # show what would be collected
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fx_options.sabr import calibrate_surface, assign_tenor_bucket

DATA_DIR = Path.home() / "trade_data" / "ETFTrader"
FX_OPTIONS_DIR = DATA_DIR / "fx_options"
FX_SPOT_DIR = DATA_DIR / "fx_spot"
FX_SABR_DIR = DATA_DIR / "fx_sabr"

IB_HOST = "127.0.0.1"
IB_PORT = 4001
IB_CLIENT_ID = 20  # Dedicated client ID for daily job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_fx")


def collect_surfaces(skip_current: bool = True) -> pd.DataFrame:
    """Collect FX option surfaces for all currencies via IB."""
    from ib_wait import wait_for_ib
    from data_collection.fx_option_collector import FXOptionCollector

    ib = wait_for_ib(IB_HOST, IB_PORT, IB_CLIENT_ID)

    collector = FXOptionCollector(
        ib, cache_dir=str(FX_OPTIONS_DIR),
    )
    surface, results = collector.collect_all(skip_current=skip_current)

    ib.disconnect()
    return surface


def collect_spots() -> pd.DataFrame:
    """Collect FX spot rates for all currencies via IB."""
    from ib_wait import wait_for_ib
    from data_collection.fx_spot_collector import FXSpotCollector

    ib = wait_for_ib(IB_HOST, IB_PORT, IB_CLIENT_ID + 1)

    collector = FXSpotCollector(ib, cache_dir=str(FX_SPOT_DIR), duration="1 M")
    spots, results = collector.collect_all()

    ib.disconnect()
    return spots


def load_todays_surface() -> pd.DataFrame:
    """Load today's surface data from the per-currency cache files."""
    today = datetime.now().date()
    dfs = []
    for path in sorted(FX_OPTIONS_DIR.glob("*.parquet")):
        if path.stem == "manifest":
            continue
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                today_df = df[df["timestamp"].dt.date == today]
                if not today_df.empty:
                    dfs.append(today_df)
        except Exception as e:
            log.warning("Could not read %s: %s", path.name, e)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def fit_sabr(surface: pd.DataFrame) -> pd.DataFrame:
    """Fit SABR parameters to each maturity slice and add nu*sqrt(T)."""
    if surface.empty:
        log.warning("Empty surface — nothing to fit")
        return pd.DataFrame()

    params = calibrate_surface(surface, min_strikes=5)

    if params.empty:
        log.warning("No slices had enough strikes for calibration")
        return pd.DataFrame()

    # Add time-scaled nu and T
    params["T"] = params["dte"].clip(lower=1) / 365.0
    params["nu_sqrt_T"] = params["nu"] * np.sqrt(params["T"])

    return params


def save_sabr(params: pd.DataFrame):
    """Append SABR params to history and overwrite latest CSV."""
    FX_SABR_DIR.mkdir(parents=True, exist_ok=True)

    # Append to parquet history
    history_path = FX_SABR_DIR / "sabr_params.parquet"
    if history_path.exists():
        existing = pd.read_parquet(history_path)
        combined = pd.concat([existing, params], ignore_index=True)
        # Dedup: keep latest per (date, currency, expiration)
        combined["date"] = pd.to_datetime(combined["timestamp"]).dt.date
        combined = combined.drop_duplicates(
            subset=["date", "currency", "expiration"], keep="last"
        )
        combined = combined.drop(columns=["date"])
    else:
        combined = params

    combined.to_parquet(history_path, index=False)
    log.info("SABR history: %d total rows in %s", len(combined), history_path)

    # Overwrite latest CSV (human-readable)
    latest_path = FX_SABR_DIR / "sabr_latest.csv"
    params.to_csv(latest_path, index=False)
    log.info("SABR latest: %s", latest_path)


def print_summary(params: pd.DataFrame):
    """Print a readable summary of today's SABR calibration."""
    if params.empty:
        return

    print(f"\n{'='*72}")
    print(f"SABR CALIBRATION SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*72}")

    for ccy in sorted(params["currency"].unique()):
        ccy_df = params[params["currency"] == ccy].sort_values("dte")
        print(f"\n  {ccy}:")
        print(f"  {'DTE':>5s}  {'Tenor':>5s}  {'Fwd':>8s}  "
              f"{'ATM IV':>7s}  {'Alpha':>7s}  {'Rho':>7s}  "
              f"{'Nu':>6s}  {'Nu√T':>5s}  {'RMSE':>7s}  {'n':>3s}")
        print(f"  {'-'*68}")
        for _, row in ccy_df.iterrows():
            print(f"  {row['dte']:>5.0f}  {row['tenor_bucket']:>5s}  "
                  f"{row['forward']:>8.4f}  "
                  f"{row['atm_iv']*100:>6.1f}%  "
                  f"{row['alpha']*100:>6.2f}%  "
                  f"{row['rho']:>7.4f}  "
                  f"{row['nu']:>6.3f}  "
                  f"{row['nu_sqrt_T']:>5.3f}  "
                  f"{row['rmse']*100:>6.3f}%  "
                  f"{row['n_strikes']:>3.0f}")

    # Nu*sqrt(T) stability check
    valid = params[params["dte"] >= 30]
    if not valid.empty:
        for ccy in sorted(valid["currency"].unique()):
            v = valid[valid["currency"] == ccy]["nu_sqrt_T"]
            print(f"\n  {ccy} ν√T:  mean={v.mean():.3f}  std={v.std():.3f}  "
                  f"CV={v.std()/v.mean()*100:.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="Daily FX surface collection + SABR fit")
    parser.add_argument("--fit-only", action="store_true",
                        help="Skip collection, refit from cached surface data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be collected without doing it")
    parser.add_argument("--no-spots", action="store_true",
                        help="Skip spot rate collection")
    args = parser.parse_args()

    start = time.time()

    if args.dry_run:
        log.info("DRY RUN — checking cache state only")
        from data_collection.fx_option_collector import FXOptionCollector
        collector = FXOptionCollector.__new__(FXOptionCollector)
        collector.cache_dir = FX_OPTIONS_DIR
        collector.currencies = ["EUR", "GBP", "AUD", "CAD", "CHF", "JPY"]
        for ccy in collector.currencies:
            state, last_ts = collector._check_cache(ccy)
            ts_str = last_ts.strftime("%Y-%m-%d %H:%M") if last_ts else "never"
            log.info("  %s: %s (last: %s)", ccy, state, ts_str)
        return

    # Step 1: Collect surfaces (or load from cache)
    if args.fit_only:
        log.info("Loading today's cached surface data...")
        surface = load_todays_surface()
        log.info("Loaded %d rows from cache", len(surface))
    else:
        log.info("Collecting FX option surfaces...")
        surface = collect_surfaces(skip_current=True)

    if surface is None or surface.empty:
        log.error("No surface data — aborting SABR fit")
        return

    # Step 2: Collect spot rates
    if not args.fit_only and not args.no_spots:
        log.info("Collecting spot rates...")
        try:
            collect_spots()
        except Exception as e:
            log.warning("Spot collection failed: %s (continuing with SABR fit)", e)

    # Step 3: Fit SABR
    log.info("Fitting SABR parameters...")
    params = fit_sabr(surface)

    if params.empty:
        log.error("SABR calibration produced no results")
        return

    # Step 4: Save and summarize
    save_sabr(params)
    print_summary(params)

    elapsed = time.time() - start
    log.info("Done in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
