"""
SABR Model — Lognormal (beta=1) Implied Volatility & Calibration

With beta=1 the Hagan (2002) formula simplifies dramatically:
all (1-beta) terms vanish, the denominator correction A = 1,
and the smile is captured by a single compact expression.

Full formula (beta=1):
    sigma(K) = alpha * [z / x(z)] * [1 + (rho*nu*alpha/4 + (2-3*rho^2)*nu^2/24) * T]

Leading order (drops the small T correction):
    sigma(K) = alpha * z / x(z)

where:
    z    = (nu / alpha) * ln(F / K)
    x(z) = ln( (sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho) )

At ATM (K=F): sigma ≈ alpha.  Skew comes from rho, curvature from nu.

Reference: Hagan, Kumar, Lesniewski, Woodward (2002)
           "Managing Smile Risk", Wilmott Magazine.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Standard tenor buckets for grouping expirations
TENOR_BUCKETS = {
    "1W": 7,
    "2W": 14,
    "1M": 30,
    "2M": 60,
    "3M": 91,
    "6M": 182,
    "9M": 274,
    "1Y": 365,
}


def assign_tenor_bucket(dte: int) -> str:
    """Map days-to-expiration to nearest standard tenor bucket."""
    best_label = "1Y"
    best_diff = abs(dte - 365)
    for label, days in TENOR_BUCKETS.items():
        diff = abs(dte - days)
        if diff < best_diff:
            best_diff = diff
            best_label = label
    return best_label


# ------------------------------------------------------------------
# Lognormal SABR (beta = 1) — the clean FX formula
# ------------------------------------------------------------------


def _x_of_z(z: float, rho: float) -> float:
    """Compute x(z) = ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho)).

    This is the core function that maps the moneyness-adjusted variable z
    into the smile shape via the correlation parameter rho.
    """
    if abs(z) < 1e-10:
        return 1.0  # z / x(z) -> 1 as z -> 0
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    return np.log((sqrt_term + z - rho) / (1.0 - rho))


def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    rho: float,
    nu: float,
) -> float:
    """Lognormal SABR (beta=1) implied volatility.

    sigma(K) = alpha * [z / x(z)] * [1 + correction * T]

    where correction = rho*nu*alpha/4 + (2 - 3*rho^2)/24 * nu^2

    Args:
        F: Forward price.
        K: Strike price.
        T: Time to expiry in years.
        alpha: SABR alpha (≈ ATM implied vol).
        rho: SABR rho (skew, -1 < rho < 1).
        nu: SABR nu (vol-of-vol, > 0).

    Returns:
        Black implied volatility (annualised).
    """
    if T <= 0:
        return alpha

    # ATM case: z -> 0, z/x(z) -> 1
    if abs(F - K) < 1e-12 * F:
        correction = rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu
        return max(alpha * (1.0 + correction * T), 1e-10)

    # General case
    log_fk = np.log(F / K)
    z = (nu / alpha) * log_fk
    x_z = _x_of_z(z, rho)

    # z / x(z) ratio — the smile shape
    z_over_xz = z / x_z if abs(x_z) > 1e-15 else 1.0

    # Time correction (only 2 terms with beta=1, not 3)
    correction = rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu

    vol = alpha * z_over_xz * (1.0 + correction * T)
    return max(vol, 1e-10)


def sabr_implied_vol_vec(
    F: float,
    strikes: np.ndarray,
    T: float,
    alpha: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Vectorised lognormal SABR over an array of strikes."""
    return np.array([sabr_implied_vol(F, K, T, alpha, rho, nu) for K in strikes])


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------


def calibrate_sabr(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    F: float,
    T: float,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Calibrate lognormal SABR (alpha, rho, nu) to market implied vols.

    Args:
        strikes: Array of strike prices.
        market_vols: Array of market implied vols (same length as strikes).
        F: Forward price.
        T: Time to expiry in years.
        weights: Optional per-strike weights (e.g. vega weights).

    Returns:
        Dict with keys: alpha, rho, nu, rmse, n_strikes, success.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    # Filter out NaN / zero vols
    valid = np.isfinite(market_vols) & (market_vols > 0) & np.isfinite(strikes) & (strikes > 0)
    strikes = strikes[valid]
    market_vols = market_vols[valid]

    if len(strikes) < 3:
        logger.warning("Need at least 3 valid strike/vol pairs, got %d", len(strikes))
        return {
            "alpha": np.nan, "rho": np.nan, "nu": np.nan,
            "rmse": np.nan, "n_strikes": len(strikes), "success": False,
        }

    if weights is not None:
        weights = np.asarray(weights, dtype=float)[valid]
    else:
        weights = np.ones(len(strikes))

    # Initial guess: alpha ≈ ATM vol (beta=1 means alpha IS the ATM vol)
    atm_idx = np.argmin(np.abs(strikes - F))
    alpha0 = market_vols[atm_idx]
    x0 = np.array([alpha0, 0.0, 0.3])

    bounds = [
        (1e-4, 2.0),      # alpha
        (-0.999, 0.999),   # rho
        (1e-4, 20.0),      # nu
    ]

    def objective(params):
        a, r, n = params
        model_vols = sabr_implied_vol_vec(F, strikes, T, a, r, n)
        residuals = (model_vols - market_vols) * weights
        return np.sum(residuals ** 2)

    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    alpha_fit, rho_fit, nu_fit = result.x
    model_vols = sabr_implied_vol_vec(F, strikes, T, alpha_fit, rho_fit, nu_fit)
    rmse = np.sqrt(np.mean((model_vols - market_vols) ** 2))

    return {
        "alpha": alpha_fit,
        "rho": rho_fit,
        "nu": nu_fit,
        "rmse": rmse,
        "n_strikes": len(strikes),
        "success": result.success,
    }


# ------------------------------------------------------------------
# Surface-level calibration
# ------------------------------------------------------------------


def calibrate_surface(
    option_df: pd.DataFrame,
    min_strikes: int = 3,
    use_mid_iv: bool = True,
) -> pd.DataFrame:
    """Calibrate lognormal SABR for each (timestamp, currency, expiration) slice.

    Args:
        option_df: FX option surface DataFrame from FXOptionCollector.
            Required columns: timestamp, currency, expiration, dte, strike,
            impliedVol, underlyingPrice.
        min_strikes: Minimum strikes needed per slice.
        use_mid_iv: If True, average call and put IVs at each strike.

    Returns:
        DataFrame of calibrated SABR parameters.
    """
    required_cols = {"timestamp", "currency", "expiration", "dte", "strike", "impliedVol", "underlyingPrice"}
    missing = required_cols - set(option_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = option_df[option_df["impliedVol"].notna() & (option_df["impliedVol"] > 0)].copy()

    if use_mid_iv and "right" in df.columns:
        df = (
            df.groupby(["timestamp", "currency", "expiration", "dte", "strike", "underlyingPrice"])
            .agg(impliedVol=("impliedVol", "mean"))
            .reset_index()
        )

    results = []
    for (ts, ccy, exp), group in df.groupby(["timestamp", "currency", "expiration"]):
        strikes = group["strike"].values
        vols = group["impliedVol"].values
        spot = group["underlyingPrice"].iloc[0]
        dte = group["dte"].iloc[0]
        T = max(dte / 365.0, 1.0 / 365.0)

        if len(strikes) < min_strikes:
            continue

        F = spot  # spot as proxy for forward

        params = calibrate_sabr(strikes, vols, F, T)

        atm_idx = np.argmin(np.abs(strikes - F))
        atm_iv = vols[atm_idx]

        results.append({
            "timestamp": ts,
            "currency": ccy,
            "expiration": exp,
            "dte": dte,
            "tenor_bucket": assign_tenor_bucket(dte),
            "forward": F,
            "atm_iv": atm_iv,
            "alpha": params["alpha"],
            "rho": params["rho"],
            "nu": params["nu"],
            "rmse": params["rmse"],
            "n_strikes": params["n_strikes"],
            "calibration_success": params["success"],
        })

    if not results:
        logger.warning("No slices had enough strikes for calibration")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    n_success = result_df["calibration_success"].sum()
    n_total = len(result_df)
    logger.info(
        "Calibrated %d/%d slices (%.0f%% success), median RMSE: %.6f",
        n_success, n_total, 100 * n_success / n_total,
        result_df["rmse"].median(),
    )

    return result_df
