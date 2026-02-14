"""
SABR Signal Construction

Builds tradeable signals from calibrated SABR parameters:
- Rho: level, changes, momentum across tenor buckets
- Alpha: relative (alpha/ATM_IV) and z-scored, changes, momentum
- Nu: z-scored level
- Term structure slopes: short vs long tenor differences
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tenors used for signal construction (subset of all available)
SIGNAL_TENORS = ["1M", "3M", "6M", "1Y"]

# Change windows (business days)
CHANGE_WINDOWS = {"1d": 1, "5d": 5, "21d": 21}

# Z-score rolling window
ZSCORE_WINDOW = 63  # ~1 quarter


def build_signals(
    sabr_params: pd.DataFrame,
    spot_history: pd.DataFrame = None,
    tenors: list = None,
    zscore_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Build signal DataFrame from calibrated SABR parameters.

    Args:
        sabr_params: Output of calibrate_surface(). Must have columns:
            timestamp, currency, tenor_bucket, alpha, rho, nu, atm_iv.
        spot_history: Optional DataFrame with columns: date, currency, spot.
            Used to compute forward-looking return targets.
        tenors: Tenor buckets to include. Default: ["1M", "3M", "6M", "1Y"].
        zscore_window: Rolling window for z-score calculations.

    Returns:
        DataFrame of signals indexed by (date, currency).
    """
    if tenors is None:
        tenors = SIGNAL_TENORS

    required = {"timestamp", "currency", "tenor_bucket", "alpha", "rho", "nu", "atm_iv"}
    missing = required - set(sabr_params.columns)
    if missing:
        raise ValueError(f"Missing columns in sabr_params: {missing}")

    df = sabr_params.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

    # Filter to requested tenors
    df = df[df["tenor_bucket"].isin(tenors)].copy()

    if df.empty:
        logger.warning("No data for tenors %s", tenors)
        return pd.DataFrame()

    # Pivot: one row per (date, currency), columns for each tenor's params
    signals = _pivot_params(df, tenors)

    # Add change and momentum features
    signals = _add_changes(signals, tenors, zscore_window)

    # Add term structure slopes
    signals = _add_term_structure(signals, tenors)

    # Add spot return targets if spot history provided
    if spot_history is not None:
        signals = _add_spot_targets(signals, spot_history)

    # Sort and clean
    signals = signals.sort_values(["date", "currency"]).reset_index(drop=True)

    logger.info(
        "Built signals: %d rows, %d columns, %d currencies, %s to %s",
        len(signals),
        len(signals.columns),
        signals["currency"].nunique(),
        signals["date"].min().strftime("%Y-%m-%d"),
        signals["date"].max().strftime("%Y-%m-%d"),
    )

    return signals


def _pivot_params(df: pd.DataFrame, tenors: list) -> pd.DataFrame:
    """Pivot SABR params to wide format: one row per (date, currency)."""
    records = []

    for (date, ccy), group in df.groupby(["date", "currency"]):
        row = {"date": date, "currency": ccy}

        for _, r in group.iterrows():
            tenor = r["tenor_bucket"]
            if tenor not in tenors:
                continue

            # Raw parameters
            row[f"rho_{tenor}"] = r["rho"]
            row[f"alpha_{tenor}"] = r["alpha"]
            row[f"nu_{tenor}"] = r["nu"]
            row[f"atm_iv_{tenor}"] = r["atm_iv"]

            # Relative alpha: alpha / ATM_IV (size-invariant)
            if r["atm_iv"] > 0:
                row[f"alpha_rel_{tenor}"] = r["alpha"] / r["atm_iv"]
            else:
                row[f"alpha_rel_{tenor}"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


def _add_changes(df: pd.DataFrame, tenors: list, zscore_window: int) -> pd.DataFrame:
    """Add change, momentum, and z-score features per currency."""
    result_frames = []

    for ccy, group in df.groupby("currency"):
        group = group.sort_values("date").copy()

        for tenor in tenors:
            rho_col = f"rho_{tenor}"
            alpha_rel_col = f"alpha_rel_{tenor}"
            nu_col = f"nu_{tenor}"

            if rho_col not in group.columns:
                continue

            # Rho changes and momentum
            for label, window in CHANGE_WINDOWS.items():
                group[f"rho_{tenor}_chg_{label}"] = group[rho_col].diff(window)

            # Rho momentum: change of the 5d change
            group[f"rho_{tenor}_mom_21d"] = group[f"rho_{tenor}_chg_5d"].diff(21)

            # Alpha relative changes
            if alpha_rel_col in group.columns:
                for label, window in CHANGE_WINDOWS.items():
                    group[f"alpha_rel_{tenor}_chg_{label}"] = group[alpha_rel_col].diff(window)

            # Z-scored alpha (rolling)
            alpha_col = f"alpha_{tenor}"
            if alpha_col in group.columns:
                rolling_mean = group[alpha_col].rolling(zscore_window, min_periods=20).mean()
                rolling_std = group[alpha_col].rolling(zscore_window, min_periods=20).std()
                group[f"alpha_z_{tenor}"] = (group[alpha_col] - rolling_mean) / rolling_std.replace(0, np.nan)

            # Z-scored nu
            if nu_col in group.columns:
                rolling_mean = group[nu_col].rolling(zscore_window, min_periods=20).mean()
                rolling_std = group[nu_col].rolling(zscore_window, min_periods=20).std()
                group[f"nu_z_{tenor}"] = (group[nu_col] - rolling_mean) / rolling_std.replace(0, np.nan)

        result_frames.append(group)

    return pd.concat(result_frames, ignore_index=True)


def _add_term_structure(df: pd.DataFrame, tenors: list) -> pd.DataFrame:
    """Add term structure slope features (short minus long tenor)."""
    if len(tenors) < 2:
        return df

    short_tenor = tenors[0]   # e.g. "1M"
    long_tenor = tenors[-1]   # e.g. "1Y"

    short_rho = f"rho_{short_tenor}"
    long_rho = f"rho_{long_tenor}"
    if short_rho in df.columns and long_rho in df.columns:
        df["rho_slope"] = df[short_rho] - df[long_rho]

    short_alpha = f"alpha_rel_{short_tenor}"
    long_alpha = f"alpha_rel_{long_tenor}"
    if short_alpha in df.columns and long_alpha in df.columns:
        df["alpha_rel_slope"] = df[short_alpha] - df[long_alpha]

    short_nu = f"nu_{short_tenor}"
    long_nu = f"nu_{long_tenor}"
    if short_nu in df.columns and long_nu in df.columns:
        df["nu_slope"] = df[short_nu] - df[long_nu]

    return df


def _add_spot_targets(df: pd.DataFrame, spot_history: pd.DataFrame) -> pd.DataFrame:
    """Add forward-looking spot return columns as prediction targets.

    Args:
        df: Signals DataFrame with date, currency columns.
        spot_history: Must have columns: date, currency, spot.
    """
    spot = spot_history.copy()
    spot["date"] = pd.to_datetime(spot["date"]).dt.normalize()
    spot = spot.sort_values(["currency", "date"])

    # Compute forward returns
    for ccy, group in spot.groupby("currency"):
        for horizon_label, horizon_days in [("1d", 1), ("5d", 5)]:
            spot.loc[group.index, f"spot_return_{horizon_label}"] = (
                group["spot"].shift(-horizon_days) / group["spot"] - 1
            )

    # Merge returns into signals
    merge_cols = ["date", "currency"]
    return_cols = [c for c in spot.columns if c.startswith("spot_return_")]
    df = df.merge(
        spot[merge_cols + return_cols],
        on=merge_cols,
        how="left",
    )

    return df


def get_feature_columns(signals_df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes targets, date, currency)."""
    exclude = {"date", "currency"}
    exclude.update(c for c in signals_df.columns if c.startswith("spot_return_"))
    return [c for c in signals_df.columns if c not in exclude and signals_df[c].dtype in [np.float64, np.float32]]
