"""
Black-76 Model — Pricing & Implied Volatility for Futures Options

The Black-76 model prices European options on futures contracts:

    C = e^{-rT} [F * N(d1) - K * N(d2)]
    P = e^{-rT} [K * N(-d2) - F * N(-d1)]

where:
    d1 = [ln(F/K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

This is the standard model for CME futures options (FX, equity index, etc.)
and is used to extract implied volatility from settlement prices.

Reference: Black, F. (1976) "The pricing of commodity contracts",
           Journal of Financial Economics.
"""

import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------
# Pricing
# ------------------------------------------------------------------


def black76_price(
    F: float,
    K: float,
    T: float,
    sigma: float,
    is_call: bool,
    r: float = 0.0,
) -> float:
    """Black-76 European option price on a futures contract.

    Args:
        F: Futures price (forward).
        K: Strike price.
        T: Time to expiry in years.
        sigma: Implied volatility (annualised).
        is_call: True for call, False for put.
        r: Risk-free rate (default 0 — standard for futures options).

    Returns:
        Option price.
    """
    if T <= 0 or sigma <= 0:
        # At or past expiry: intrinsic value
        if is_call:
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    df = np.exp(-r * T)

    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_vega(
    F: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """Black-76 vega: dPrice/dSigma.

    Used as the derivative in the Newton-Raphson IV solver.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    df = np.exp(-r * T)

    return df * F * sqrt_T * norm.pdf(d1)


# ------------------------------------------------------------------
# Implied volatility solver
# ------------------------------------------------------------------


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    is_call: bool,
    r: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    """Extract implied volatility from an option price via Newton-Raphson.

    Uses the Brenner-Subrahmanyam (1988) approximation as initial guess:
        sigma_0 ≈ sqrt(2*pi/T) * price / F

    Args:
        price: Observed option price (mid or settlement).
        F: Futures price (forward).
        K: Strike price.
        T: Time to expiry in years.
        is_call: True for call, False for put.
        r: Risk-free rate.
        max_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance on price difference.

    Returns:
        Implied volatility, or NaN if solver fails.
    """
    if price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    # Check intrinsic value
    df = np.exp(-r * T)
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)

    if price < intrinsic - 1e-10:
        # Price below intrinsic — arbitrage, return NaN
        return np.nan

    if price <= intrinsic + 1e-10:
        # At intrinsic — essentially zero extrinsic value
        return 1e-6

    # Brenner-Subrahmanyam initial guess
    sigma = np.sqrt(2.0 * np.pi / T) * price / F
    sigma = np.clip(sigma, 0.01, 5.0)

    for _ in range(max_iter):
        p = black76_price(F, K, T, sigma, is_call, r)
        v = black76_vega(F, K, T, sigma, r)

        diff = p - price
        if abs(diff) < tol:
            return sigma

        if v < 1e-12:
            # Vega too small for Newton — use bisection
            return _bisection_iv(price, F, K, T, is_call, r)

        step = diff / v
        if abs(step) > 2.0 * sigma:
            # Step too large — use bisection
            return _bisection_iv(price, F, K, T, is_call, r)

        sigma -= step
        sigma = max(sigma, 1e-6)

    # Did not converge — try bisection
    return _bisection_iv(price, F, K, T, is_call, r)


def _bisection_iv(
    price: float,
    F: float,
    K: float,
    T: float,
    is_call: bool,
    r: float = 0.0,
    lo: float = 1e-4,
    hi: float = 5.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Bisection fallback for implied vol when Newton-Raphson fails."""
    p_lo = black76_price(F, K, T, lo, is_call, r)
    p_hi = black76_price(F, K, T, hi, is_call, r)

    if price < p_lo or price > p_hi:
        return np.nan

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p_mid = black76_price(F, K, T, mid, is_call, r)

        if abs(p_mid - price) < tol:
            return mid

        if p_mid < price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


# ------------------------------------------------------------------
# Vectorised interface
# ------------------------------------------------------------------


def implied_vol_vec(
    prices: np.ndarray,
    F: np.ndarray,
    strikes: np.ndarray,
    T: np.ndarray,
    is_call: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """Vectorised implied vol extraction for arrays of options.

    All array inputs must have the same length.

    Returns:
        Array of implied vols (NaN where solver fails).
    """
    n = len(prices)
    ivs = np.full(n, np.nan)

    for i in range(n):
        ivs[i] = implied_vol(
            float(prices[i]),
            float(F[i]),
            float(strikes[i]),
            float(T[i]),
            bool(is_call[i]),
            r,
        )

    return ivs
