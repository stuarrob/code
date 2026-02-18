"""
Momentum Timing Signals for Leveraged ETFs

Determines when to be IN (risk-on) vs OUT (risk-off) using:
1. 12-month momentum with 1-month skip (Antonacci dual momentum)
2. 200-day SMA crossover
3. Volatility regime filter (realised vol and/or VIX)

Combined signal: enter when BOTH agree, exit when EITHER turns negative.
Evaluated weekly (Friday close).
"""

from enum import Enum
import numpy as np
import pandas as pd


class SignalState(Enum):
    RISK_ON = "risk_on"      # Both signals positive → full leveraged allocation
    RISK_OFF = "risk_off"    # Both signals negative → 100% cash
    CAUTION = "caution"      # Mixed signals → reduced allocation


# Default parameters
MOMENTUM_LOOKBACK = 252   # 12 months
MOMENTUM_SKIP = 21        # skip recent 1 month
SMA_PERIOD = 200          # 200-day simple moving average


def compute_momentum(prices: pd.Series, lookback: int = MOMENTUM_LOOKBACK,
                     skip: int = MOMENTUM_SKIP) -> pd.Series:
    """12-month momentum with 1-month skip.

    momentum = price[t - skip] / price[t - lookback] - 1

    Positive = uptrend, negative = downtrend.
    """
    if len(prices) < lookback:
        return pd.Series(np.nan, index=prices.index)
    return prices.shift(skip) / prices.shift(lookback) - 1


def compute_sma_signal(prices: pd.Series, period: int = SMA_PERIOD) -> pd.Series:
    """True when price is above its N-day SMA."""
    sma = prices.rolling(period, min_periods=period).mean()
    return prices > sma


def compute_signals(
    reference_prices: pd.Series,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
    momentum_skip: int = MOMENTUM_SKIP,
    sma_period: int = SMA_PERIOD,
    signal_mode: str = "dual",
    vol_filter_threshold: float = 0.0,
    vol_filter_lookback: int = 20,
    vix_prices: pd.Series = None,
    vix_weight: float = 0.0,
    vix_exit_threshold: float = 0.0,
) -> pd.DataFrame:
    """Compute all timing signals for a reference index.

    Args:
        reference_prices: Daily close prices for the unleveraged
            reference (e.g. QQQ for TQQQ, SPY for UPRO).
        momentum_lookback: Lookback for momentum calculation.
        momentum_skip: Days to skip (avoid reversal).
        sma_period: SMA period for trend signal.
        signal_mode: "dual", "sma_only", or "momentum_only".
        vol_filter_threshold: If > 0, override to RISK_OFF when
            composite vol exceeds this. 0 = disabled.
            Typical: 0.20 aggressive, 0.25 moderate.
        vol_filter_lookback: Rolling window for vol filter.
        vix_prices: Optional VIX index daily closes. When
            provided with vix_weight > 0, blends VIX into
            the vol filter as a composite measure.
        vix_weight: Weight for VIX in composite vol filter
            (0-1). composite = w*VIX + (1-w)*ref_vol.
            0 = ref_vol only (default), 1 = VIX only.
        vix_exit_threshold: Emergency VIX exit level.
            If > 0, force RISK_OFF when VIX exceeds this
            regardless of other signals. 0 = disabled.

    Returns:
        DataFrame with columns: momentum, momentum_signal,
        sma_signal, ref_vol, vol_filter, signal_state.
        When VIX provided: also vix, composite_vol.
    """
    df = pd.DataFrame({"price": reference_prices})
    df.index = pd.to_datetime(df.index)

    # Momentum signal
    df["momentum"] = compute_momentum(
        df["price"], momentum_lookback, momentum_skip,
    )
    df["momentum_signal"] = df["momentum"] > 0

    # SMA signal
    sma = df["price"].rolling(
        sma_period, min_periods=sma_period,
    ).mean()
    df["sma"] = sma
    df["sma_signal"] = df["price"] > df["sma"]

    # Reference index realised vol
    ref_returns = df["price"].pct_change()
    df["ref_vol"] = (
        ref_returns
        .rolling(vol_filter_lookback, min_periods=10)
        .std() * np.sqrt(252)
    )

    # VIX integration (optional)
    if vix_prices is not None:
        vix_aligned = vix_prices.reindex(
            df.index, method="ffill",
        )
        df["vix"] = vix_aligned
        df["vix_annualised"] = vix_aligned / 100.0
    else:
        df["vix"] = np.nan
        df["vix_annualised"] = np.nan

    # Composite vol: blend ref_vol with VIX
    if vix_weight > 0 and vix_prices is not None:
        df["composite_vol"] = (
            vix_weight * df["vix_annualised"]
            + (1 - vix_weight) * df["ref_vol"]
        )
    else:
        df["composite_vol"] = df["ref_vol"]

    # Vol filter using composite vol
    df["vol_filter"] = True
    if vol_filter_threshold > 0:
        df["vol_filter"] = (
            df["composite_vol"] < vol_filter_threshold
        )

    # VIX emergency exit override
    df["vix_exit"] = False
    if vix_exit_threshold > 0 and vix_prices is not None:
        df["vix_exit"] = df["vix"] > vix_exit_threshold

    # Combined state
    df["signal_state"] = df.apply(
        lambda row: _classify_signal(row, signal_mode),
        axis=1,
    )

    return df


def _classify_signal(row, mode: str = "dual") -> str:
    """Classify from momentum, SMA, vol filter, and VIX."""
    mom = row.get("momentum_signal")
    sma = row.get("sma_signal")
    vol_ok = row.get("vol_filter", True)
    vix_exit = row.get("vix_exit", False)

    # VIX emergency exit overrides everything
    if vix_exit:
        return SignalState.RISK_OFF.value

    # Vol filter overrides trend signals
    if not vol_ok:
        return SignalState.RISK_OFF.value

    if mode == "sma_only":
        if pd.isna(sma):
            return SignalState.RISK_OFF.value
        return (SignalState.RISK_ON.value if sma
                else SignalState.RISK_OFF.value)

    if mode == "momentum_only":
        if pd.isna(mom):
            return SignalState.RISK_OFF.value
        return (SignalState.RISK_ON.value if mom
                else SignalState.RISK_OFF.value)

    # dual: require both
    if pd.isna(mom) or pd.isna(sma):
        return SignalState.RISK_OFF.value

    if mom and sma:
        return SignalState.RISK_ON.value
    elif not mom and not sma:
        return SignalState.RISK_OFF.value
    else:
        return SignalState.CAUTION.value


def get_weekly_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Resample signals to weekly (Friday close) for decision-making.

    Returns one row per week with the Friday signal state.
    """
    # Get Friday signals (or last trading day of week)
    weekly = signals_df.resample("W-FRI").last()
    return weekly.dropna(subset=["signal_state"])


def compute_vol_scalar(
    equity_prices: pd.Series,
    target_vol: float = 0.15,
    vol_lookback: int = 60,
) -> pd.Series:
    """Compute volatility-based position size scalar.

    scalar = target_vol / realised_vol
    Capped at 1.0 (never more than 100% of base allocation).

    Args:
        equity_prices: Daily close prices of the leveraged ETF itself.
        target_vol: Target annualised portfolio volatility.
        vol_lookback: Rolling window for realised vol calculation.

    Returns:
        Series of scalars (0 to 1.0).
    """
    returns = equity_prices.pct_change()
    realised_vol = returns.rolling(vol_lookback, min_periods=20).std() * np.sqrt(252)
    scalar = target_vol / realised_vol.replace(0, np.nan)
    return scalar.clip(upper=1.0).fillna(0.5)  # default 0.5 if insufficient data
