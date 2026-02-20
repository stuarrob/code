"""
FX Skew Divergence Signal
=========================
Signal: Short-dated rho has moved while longer-dated rho is quiet.
Rationale: Short-dated skew leads long-dated skew. The edge exists in the window
between the fast tenor moving and the slow tenor catching up.

Key finding: the optimal tenor pair differs by liquidity:
  - EUR (most liquid)  → 1W vs 1M  (fastest propagation)
  - GBP (liquid)       → 1M vs 3M  (3-day lookback)
  - JPY (liquid, USD/) → 1M vs 3M  (7-day lookback, bear direction)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAIR_MAP = {"EUR": "EURUSD", "GBP": "GBPUSD", "JPY": "USDJPY"}
CURRENCIES = ["EUR", "GBP", "JPY"]

# Sign convention: +1 = rho up → pair goes up (xxx/USD)
#                  -1 = rho up → pair goes down (USD/xxx)
SPOT_SIGN = {"EUR": +1, "GBP": +1, "JPY": -1}

TARGET_DTE = {"1W": 7, "2W": 14, "1M": 30, "2M": 60, "3M": 91, "6M": 182}
RHO_BOUNDARY = 0.98
FORWARD_HORIZONS = [1, 2, 3, 5, 7, 10, 14, 21]

# Per-pair tuned signal configuration
PAIR_CONFIGS = {
    "GBP": {
        "fast_tenor": "1M",
        "slow_tenor": "3M",
        "fast_method": "chg",
        "fast_window": 3,
        "fast_threshold_q": 0.25,
        "slow_method": "dev",
        "slow_window": 5,
        "quiet_q": 0.50,
        "hold_days": 5,
        "trade_direction": "bull",
    },
    "EUR": {
        "fast_tenor": "1W",
        "slow_tenor": "1M",
        "fast_method": "chg",
        "fast_window": 3,
        "fast_threshold_q": 0.25,
        "slow_method": "dev",
        "slow_window": 10,
        "quiet_q": 0.50,
        "hold_days": 5,
        "trade_direction": "bull",
    },
    "JPY": {
        "fast_tenor": "1M",
        "slow_tenor": "3M",
        "fast_method": "chg",
        "fast_window": 7,
        "fast_threshold_q": 0.25,
        "slow_method": "chg",
        "slow_window": 3,
        "quiet_q": 0.50,
        "hold_days": 10,
        "trade_direction": "bear",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pair_data(currency, data_dir=None, config=None):
    """Load SABR + spot data for one currency pair using its tuned config.

    Returns (df, mask_both, info).
    """
    if data_dir is None:
        data_dir = Path.home() / "trade_data" / "ETFTrader"
    data_dir = Path(data_dir)

    if config is None:
        config = PAIR_CONFIGS[currency]

    pair = PAIR_MAP[currency]
    fast_tenor = config["fast_tenor"]
    slow_tenor = config["slow_tenor"]
    tenors = [fast_tenor, slow_tenor]

    # --- SABR ---
    sabr_raw = pd.read_parquet(data_dir / "fx_sabr" / "sabr_params_historical.parquet")
    sabr_raw["date"] = pd.to_datetime(sabr_raw["timestamp"]).dt.normalize()

    sabr = sabr_raw[
        (sabr_raw["currency"] == currency) & (sabr_raw["tenor_bucket"].isin(tenors))
    ].copy()

    n_raw = len(sabr)
    boundary = sabr["rho"].abs() > RHO_BOUNDARY
    n_boundary = boundary.sum()
    sabr = sabr[~boundary].copy()

    # Dedup: closest DTE to target per (date, tenor)
    sabr["target_dte"] = sabr["tenor_bucket"].map(TARGET_DTE)
    sabr["dte_diff"] = (sabr["dte"] - sabr["target_dte"]).abs()
    sabr = sabr.sort_values("dte_diff").drop_duplicates(["date", "tenor_bucket"], keep="first")
    sabr = sabr.drop(columns=["target_dte", "dte_diff"])

    # Pivot wide
    df = sabr.pivot_table(
        index="date", columns="tenor_bucket", values=["rho", "alpha", "nu"], aggfunc="first"
    )
    df.columns = [f"{p}_{t}" for p, t in df.columns]
    df = df.reset_index().sort_values("date")

    # --- Spot ---
    spot = pd.read_parquet(data_dir / "fx_spot" / f"{pair}.parquet")
    spot["date"] = pd.to_datetime(spot["date"]).dt.normalize()
    spot = spot.sort_values("date")

    for h in FORWARD_HORIZONS:
        spot[f"fwd_ret_{h}d"] = spot["close"].shift(-h) / spot["close"] - 1

    merge_cols = ["date", "close"] + [c for c in spot.columns if c.startswith("fwd_ret_")]
    df = df.merge(spot[merge_cols], on="date", how="left")
    df = df.rename(columns={"close": "spot"})

    rho_fast = f"rho_{fast_tenor}"
    rho_slow = f"rho_{slow_tenor}"
    mask_both = df[rho_fast].notna() & df[rho_slow].notna()

    info = {
        "currency": currency,
        "pair": pair,
        "fast_tenor": fast_tenor,
        "slow_tenor": slow_tenor,
        "n_raw": n_raw,
        "n_boundary": n_boundary,
        "boundary_pct": n_boundary / n_raw * 100 if n_raw > 0 else 0,
        "n_dates": len(df),
        "n_fast": df[rho_fast].notna().sum(),
        "n_slow": df[rho_slow].notna().sum(),
        "n_overlap": mask_both.sum(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
    }

    return df, mask_both, info


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def build_detection_metrics(df, tenor, windows=None):
    """Compute change and MA-deviation metrics for a tenor.

    Returns dict of (method, window) -> column_name.
    """
    if windows is None:
        windows = [3, 5, 7, 10]

    rho_col = f"rho_{tenor}"
    cols = {}

    for N in windows:
        col_chg = f"rho_{tenor}_chg{N}d"
        if col_chg not in df.columns:
            df[col_chg] = df[rho_col] - df[rho_col].shift(N)
        cols[("chg", N)] = col_chg

        col_dev = f"rho_{tenor}_dev{N}d"
        if col_dev not in df.columns:
            df[col_dev] = df[rho_col] - df[rho_col].rolling(N, min_periods=max(2, N // 2)).mean()
        cols[("dev", N)] = col_dev

    return cols


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def build_divergence_signal(df, mask_both, config):
    """Build the divergence signal using per-pair config.

    Returns (bull_mask, bear_mask, bull_fast_only, bear_fast_only, signal_info).
    """
    fast_tenor = config["fast_tenor"]
    slow_tenor = config["slow_tenor"]

    fast_cols = build_detection_metrics(df, fast_tenor, [config["fast_window"]])
    slow_cols = build_detection_metrics(df, slow_tenor, [config["slow_window"]])

    fast_col = fast_cols[(config["fast_method"], config["fast_window"])]
    slow_col = slow_cols[(config["slow_method"], config["slow_window"])]

    # Fast tenor thresholds (computed on overlap dates)
    fast_vals = df.loc[mask_both, fast_col].dropna()
    q = config["fast_threshold_q"]
    thresh_lo = fast_vals.quantile(q)
    thresh_hi = fast_vals.quantile(1 - q)

    # Slow tenor quiet threshold
    slow_vals = df.loc[mask_both, slow_col].dropna().abs()
    quiet_thresh = slow_vals.quantile(config["quiet_q"])

    quiet_mask = df[slow_col].abs() < quiet_thresh

    # Compose signals
    bull_mask = (df[fast_col] > thresh_hi) & quiet_mask & mask_both
    bear_mask = (df[fast_col] < thresh_lo) & quiet_mask & mask_both

    # Fast-only (no quiet filter) for ablation
    bull_fast_only = (df[fast_col] > thresh_hi) & mask_both
    bear_fast_only = (df[fast_col] < thresh_lo) & mask_both

    signal_info = {
        "fast_col": fast_col,
        "slow_col": slow_col,
        "thresh_lo": thresh_lo,
        "thresh_hi": thresh_hi,
        "quiet_thresh": quiet_thresh,
        "n_bull": bull_mask.sum(),
        "n_bear": bear_mask.sum(),
        "n_bull_fast_only": bull_fast_only.sum(),
        "n_bear_fast_only": bear_fast_only.sum(),
    }

    return bull_mask, bear_mask, bull_fast_only, bear_fast_only, signal_info


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------

def evaluate_signal(df, mask, sign, horizons=None):
    """Evaluate a single signal mask: t-test, hit rate, mean return.

    sign: +1 for bullish (expect pair to go up), -1 for bearish.
    """
    if horizons is None:
        horizons = ["5d", "10d"]

    results = {}
    for h in horizons:
        ret_col = f"fwd_ret_{h}"
        rets = df.loc[mask, ret_col].dropna() * sign
        n = len(rets)
        if n < 3:
            results[h] = {"n": n, "mean_bps": np.nan, "hit_rate": np.nan,
                          "t_stat": np.nan, "p_value": np.nan}
            continue
        t, p = stats.ttest_1samp(rets, 0)
        results[h] = {
            "n": n,
            "mean_bps": rets.mean() * 10000,
            "hit_rate": (rets > 0).mean(),
            "t_stat": t,
            "p_value": p,
        }
    return results


def evaluate_pair(df, mask_both, currency, config=None):
    """Full evaluation of the divergence signal for one currency pair.

    Returns a dict with all results needed for the cross-pair summary.
    """
    if config is None:
        config = PAIR_CONFIGS[currency]

    spot_sign = SPOT_SIGN[currency]
    bull_mask, bear_mask, bull_fo, bear_fo, sig_info = build_divergence_signal(
        df, mask_both, config
    )

    result = {
        "currency": currency,
        "pair": PAIR_MAP[currency],
        "config": config,
        "n_overlap": mask_both.sum(),
        "signal_info": sig_info,
    }

    # Evaluate divergence signals
    result["bull_div"] = evaluate_signal(df, bull_mask, +spot_sign)
    result["bear_div"] = evaluate_signal(df, bear_mask, -spot_sign)

    # Evaluate fast-only (ablation)
    result["bull_fast_only"] = evaluate_signal(df, bull_fo, +spot_sign)
    result["bear_fast_only"] = evaluate_signal(df, bear_fo, -spot_sign)

    # Sub-period consistency (use best horizon per pair)
    mid = df["date"].median()
    for label, mask, sign in [
        ("bull", bull_mask, +spot_sign),
        ("bear", bear_mask, -spot_sign),
    ]:
        halves = {}
        for half_label, half_filt in [("H1", df["date"] <= mid), ("H2", df["date"] > mid)]:
            combined = mask & half_filt
            for h in ["5d", "10d"]:
                rets = df.loc[combined, f"fwd_ret_{h}"].dropna() * sign
                halves[(half_label, h)] = rets.mean() * 10000 if len(rets) > 2 else np.nan
        for h in ["5d", "10d"]:
            h1 = halves[("H1", h)]
            h2 = halves[("H2", h)]
            consistent = (
                not np.isnan(h1) and not np.isnan(h2) and h1 * h2 > 0
            )
            result[f"{label}_H1_{h}_bps"] = h1
            result[f"{label}_H2_{h}_bps"] = h2
            result[f"{label}_{h}_consistent"] = consistent

    # Store masks for backtest
    result["bull_mask"] = bull_mask
    result["bear_mask"] = bear_mask

    return result


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    df, bull_mask, bear_mask,
    hold_days=5, spot_sign=1,
    cost_per_trade_pct=0.00005,
):
    """Backtest: position on signal, hold for N days.

    spot_sign: +1 for xxx/USD pairs (long=buy pair),
               -1 for USD/xxx pairs (bull rho → short pair).
    """
    bt = df[["date", "spot"]].dropna().copy()
    bt = bt.reset_index(drop=True)
    bt["daily_ret"] = bt["spot"].pct_change()
    bt["position"] = 0.0

    pos_col = bt.columns.get_loc("position")
    hold_until = -1
    for i in range(len(bt)):
        if i <= hold_until:
            bt.iloc[i, pos_col] = bt.iloc[i - 1, pos_col]
            continue
        date_i = bt["date"].iloc[i]
        if date_i in df["date"].values:
            df_idx = df.index[df["date"] == date_i]
            if len(df_idx) > 0:
                idx = df_idx[0]
                if bull_mask.get(idx, False):
                    bt.iloc[i, pos_col] = float(spot_sign)
                    hold_until = i + hold_days
                elif bear_mask.get(idx, False):
                    bt.iloc[i, pos_col] = float(-spot_sign)
                    hold_until = i + hold_days

    bt["signal_ret"] = bt["position"].shift(1) * bt["daily_ret"]
    bt["pos_change"] = bt["position"].diff().abs()
    bt["cost"] = bt["pos_change"] * cost_per_trade_pct
    bt["net_ret"] = bt["signal_ret"] - bt["cost"]
    bt["cum_gross"] = (1 + bt["signal_ret"].fillna(0)).cumprod()
    bt["cum_net"] = (1 + bt["net_ret"].fillna(0)).cumprod()

    n_trades = (bt["pos_change"] > 0).sum()
    active = (bt["position"] != 0).sum()
    sr_gross = (
        bt["signal_ret"].mean() / bt["signal_ret"].std() * np.sqrt(252)
        if bt["signal_ret"].std() > 0 else 0
    )
    sr_net = (
        bt["net_ret"].mean() / bt["net_ret"].std() * np.sqrt(252)
        if bt["net_ret"].std() > 0 else 0
    )
    max_dd = (bt["cum_net"] / bt["cum_net"].cummax() - 1).min()

    return {
        "n_trades": n_trades,
        "active_days": active,
        "total_days": len(bt),
        "sharpe_gross": sr_gross,
        "sharpe_net": sr_net,
        "total_return_gross": (bt["cum_gross"].iloc[-1] - 1) * 100,
        "total_return_net": (bt["cum_net"].iloc[-1] - 1) * 100,
        "max_drawdown": max_dd * 100,
        "bt": bt,
    }
