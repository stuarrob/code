"""
Step 10: SABR Calibration & Signal Generation

Calibrates simplified SABR model to FX option surfaces and builds
time series of parameter-based signals.

Usage (from notebook):
    from scripts.s10_fx_sabr import calibrate_and_build_signals
    signals = calibrate_and_build_signals(FX_CACHE_DIR, FX_SABR_DIR, PROCESSED_DIR)
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


def calibrate_and_build_signals(
    fx_option_dir: Path,
    fx_sabr_dir: Path,
    processed_dir: Path,
    fx_spot_dir: Path = None,
    beta: float = 0.5,
    tenors: list = None,
) -> pd.DataFrame:
    """Calibrate SABR to option surface and build signals.

    Args:
        fx_option_dir: Directory with per-currency option parquets.
        fx_sabr_dir: Directory for SABR parameter and signal output.
        processed_dir: Directory with latest option surface parquet.
        fx_spot_dir: Optional directory with FX spot parquets (for targets).
        beta: Fixed SABR beta (default 0.5 for FX).
        tenors: Tenor buckets for signals. Default: ["1M", "3M", "6M", "1Y"].

    Returns:
        DataFrame of SABR-based signals.
    """
    from fx_options.sabr import calibrate_surface
    from fx_options.signals import build_signals

    fx_sabr_dir = Path(fx_sabr_dir)
    fx_sabr_dir.mkdir(parents=True, exist_ok=True)

    # Load option surface
    surface_path = processed_dir / "fx_option_surface_latest.parquet"
    if not surface_path.exists():
        # Try building from per-currency files
        surface = _load_all_surfaces(fx_option_dir)
        if surface is None:
            print("No FX option surface data found.")
            return pd.DataFrame()
    else:
        surface = pd.read_parquet(surface_path)

    print(f"Option surface: {len(surface)} rows, {surface['currency'].nunique()} currencies")

    # Calibrate SABR
    print("\nCalibrating SABR model (beta={beta})...")
    params_df = calibrate_surface(surface, beta=beta)

    if params_df.empty:
        print("Calibration produced no results.")
        return pd.DataFrame()

    # Save SABR parameters
    params_path = fx_sabr_dir / "sabr_params.parquet"
    params_df = _merge_and_save(params_df, params_path)
    n_success = params_df["calibration_success"].sum()
    print(f"SABR parameters: {len(params_df)} slices, {n_success} successful")
    print(f"  Median RMSE: {params_df['rmse'].median():.6f}")
    print(f"  Saved: {params_path}")

    # Load spot history for targets (optional)
    spot_history = None
    if fx_spot_dir is not None:
        spot_path = Path(fx_spot_dir)
        if spot_path.exists():
            spot_history = _load_spot_history(spot_path)
            if spot_history is not None:
                print(f"\nSpot history: {len(spot_history)} rows")

    # Build signals
    print("\nBuilding signals...")
    signals = build_signals(params_df, spot_history=spot_history, tenors=tenors)

    if not signals.empty:
        signals_path = fx_sabr_dir / "sabr_signals.parquet"
        signals.to_parquet(signals_path, index=False)
        print(f"Signals: {len(signals)} rows, {len(signals.columns)} columns")
        print(f"  Saved: {signals_path}")

    return signals


def _load_all_surfaces(fx_option_dir: Path) -> pd.DataFrame:
    """Load and combine all per-currency option parquets."""
    fx_option_dir = Path(fx_option_dir)
    if not fx_option_dir.exists():
        return None

    dfs = []
    for f in fx_option_dir.glob("*.parquet"):
        if f.stem == "manifest":
            continue
        df = pd.read_parquet(f)
        dfs.append(df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _load_spot_history(spot_dir: Path) -> pd.DataFrame:
    """Load per-pair spot parquets into a long-format DataFrame."""
    dfs = []
    for f in spot_dir.glob("*.parquet"):
        df = pd.read_parquet(f)
        if "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if "currency" not in df.columns:
                # Infer from filename (e.g. EURUSD.parquet -> EUR)
                pair = f.stem
                df["currency"] = pair[:3]
            dfs.append(df[["date", "currency", "close"]].rename(columns={"close": "spot"}))

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _merge_and_save(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge new SABR params with existing, dedup, save."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["timestamp", "currency", "expiration"], keep="last"
        )
        combined = combined.sort_values(["timestamp", "currency", "expiration"]).reset_index(drop=True)
    else:
        combined = new_df.sort_values(["timestamp", "currency", "expiration"]).reset_index(drop=True)

    combined.to_parquet(path, index=False)
    return combined
