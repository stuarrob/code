"""
Step 9: FX Spot Data Collection

Collects daily FX spot rates for major currency pairs via IB Gateway.
Stores per-pair parquets and a combined spot matrix.

Usage (from notebook):
    from scripts.s9_fx_spot import collect_fx_spot
    spot_matrix = collect_fx_spot(FX_SPOT_DIR, PROCESSED_DIR)
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


def collect_fx_spot(
    fx_spot_dir: Path,
    processed_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 7,
    currencies: list = None,
    duration: str = "1 Y",
    run_collection: bool = True,
    years: int = 1,
) -> pd.DataFrame:
    """Collect/update FX spot history.

    Args:
        fx_spot_dir: Directory for per-pair parquet files.
        processed_dir: Directory for combined spot matrix.
        ib_host: IB Gateway host.
        ib_port: IB Gateway port.
        ib_client_id: Client ID (use 7 to avoid conflict with other collectors).
        currencies: List of currency codes, or None for all 6 majors.
        duration: IB duration string (e.g. "1 Y", "6 M").
        run_collection: If False, only load from cache.

    Returns:
        DataFrame with date x currency spot matrix.
    """
    matrix = None
    matrix_path = processed_dir / "fx_spot_history.parquet"

    # Check for existing data
    if matrix_path.exists():
        matrix = pd.read_parquet(matrix_path)
        print(f"Loaded existing FX spot matrix: {matrix.shape}")
        if hasattr(matrix.index, 'max'):
            print(f"  Latest date: {matrix.index.max()}")

    if not run_collection:
        if matrix is None:
            print("No cached FX spot data and collection disabled.")
        return matrix

    # Run collection via IB
    try:
        from data_collection.fx_spot_collector import FXSpotCollector
        from ib_insync import IB

        ib = IB()
        ib.connect(
            ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10
        )
        accounts = ib.managedAccounts()
        print(f"Connected to IB: {accounts[0] if accounts else 'unknown'}\n")

        collector = FXSpotCollector(
            ib=ib,
            cache_dir=str(fx_spot_dir),
            duration=duration,
            currencies=currencies,
        )

        if years > 1:
            combined_df, results = collector.collect_all_extended(years=years)
        else:
            combined_df, results = collector.collect_all()
        new_matrix = collector.build_spot_matrix()
        ib.disconnect()

        # Save combined matrix
        if new_matrix is not None and not new_matrix.empty:
            processed_dir.mkdir(parents=True, exist_ok=True)
            new_matrix.to_parquet(matrix_path)
            print(f"\nSaved spot matrix: {matrix_path}")
            matrix = new_matrix

    except Exception as e:
        print(f"\nIB connection failed: {e}")
        if matrix is not None:
            print(f"Using existing cached matrix ({matrix.shape}).")

    return matrix
