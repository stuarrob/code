"""
Step 8: FX Option Surface Collection

Snapshot collector for PHLX FX option prices and Greeks.
Captures the full volatility surface (all strikes x expirations x put/call)
for EUR, GBP, AUD, CAD, CHF, JPY vs USD.

Usage (from notebook):
    from scripts.s8_fx_options import collect_fx_options
    surface = collect_fx_options(FX_CACHE_DIR, PROCESSED_DIR)
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
