"""
Step 8: FX Option Surface Collection

Two modes:
  1. Live snapshot via IB Gateway (daily collection, ~15 min)
  2. Historical backfill via Databento (3 years, ~30 min)

Usage (from notebook):
    from scripts.s8_fx_options import collect_fx_options, collect_fx_options_historical
    surface = collect_fx_options(FX_CACHE_DIR, PROCESSED_DIR)          # IB live
    surface = collect_fx_options_historical(FX_DB_DIR, start, end)     # Databento
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


def collect_fx_options(
    fx_cache_dir: Path,
    processed_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 6,
    currencies: list = None,
    run_collection: bool = True,
    skip_current: bool = True,
) -> pd.DataFrame:
    """Collect/update FX option surface snapshots.

    Args:
        fx_cache_dir: Directory for per-currency parquet files.
        processed_dir: Directory for combined latest snapshot.
        ib_host: IB Gateway host.
        ib_port: IB Gateway port.
        ib_client_id: Client ID (use 6 to avoid conflict with ETF collector).
        currencies: List of currency codes, or None for all 6 majors.
        run_collection: If False, only load from cache.
        skip_current: Skip currencies already collected today.

    Returns:
        DataFrame with FX option surface data.
    """
    surface = None
    latest_path = processed_dir / "fx_option_surface_latest.parquet"

    # Check for existing data
    if latest_path.exists():
        surface = pd.read_parquet(latest_path)
        print(f"Loaded existing FX surface: {len(surface)} rows")
        if "timestamp" in surface.columns:
            ts = surface["timestamp"].max()
            print(f"  Latest snapshot: {ts}")

    if not run_collection:
        if surface is None:
            print("No cached FX option data and collection disabled.")
        return surface

    # Run collection via IB
    try:
        from data_collection.fx_option_collector import FXOptionCollector
        from ib_insync import IB

        ib = IB()
        ib.connect(
            ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10
        )
        accounts = ib.managedAccounts()
        print(f"Connected to IB: {accounts[0] if accounts else 'unknown'}\n")

        collector = FXOptionCollector(
            ib=ib,
            cache_dir=str(fx_cache_dir),
            currencies=currencies,
        )

        new_surface, results = collector.collect_all(skip_current=skip_current)
        ib.disconnect()

        # Save combined latest snapshot
        if new_surface is not None and not new_surface.empty:
            processed_dir.mkdir(parents=True, exist_ok=True)
            new_surface.to_parquet(latest_path, index=False)
            print(f"\nSaved latest surface: {latest_path}")
            surface = new_surface

    except Exception as e:
        print(f"\nIB connection failed: {e}")
        if surface is not None:
            print(f"Using existing cached surface ({len(surface)} rows).")

    return surface


def collect_fx_options_historical(
    fx_db_dir: Path,
    start: str = "2023-02-20",
    end: str = "2026-02-20",
    currencies: list = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Collect historical FX option surface data via Databento.

    Fetches CME FX futures option settlement prices, computes implied
    vols via Black-76, and returns a DataFrame compatible with the
    SABR calibration pipeline.

    Args:
        fx_db_dir: Cache directory for Databento FX data.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        currencies: List of currency codes, or None for all 6 majors.
        use_cache: If True, load from cache if available.

    Returns:
        DataFrame with historical FX option surface data.
    """
    from data_collection.databento_fx_collector import DatabentoFXCollector

    # Check cache first
    if use_cache:
        dfs = []
        all_currencies = currencies or ["EUR", "GBP", "AUD", "CAD", "JPY"]
        for ccy in all_currencies:
            path = Path(fx_db_dir) / f"{ccy}_surface.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                dfs.append(df)
                print(f"  {ccy}: loaded {len(df)} rows from cache")

        if dfs:
            surface = pd.concat(dfs, ignore_index=True)
            print(f"\nLoaded historical surface: {len(surface)} rows, "
                  f"{surface['currency'].nunique()} currencies")
            return surface

    # Fetch from Databento
    collector = DatabentoFXCollector(cache_dir=str(fx_db_dir))
    surface = collector.fetch_history(
        currencies=currencies,
        start=start,
        end=end,
        use_cache=use_cache,
    )

    return surface
