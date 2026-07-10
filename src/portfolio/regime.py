"""Regime overlay signal — SPY trend + VIX level with hysteresis and dwell.

Closes T2.1 in the adversarial-review roadmap. Applies the slowly-varying
design principle in three deliberate places:

  1. **Trend uses a 200-day SMA** — long-lookback, slowly moving. A shorter
     window (50d) whipsaws on ordinary corrections. 200d is the classic
     bull/bear separator (Faber 2007, Ang 2014).
  2. **Vol uses a raw VIX cutoff** with hysteresis, not a differenced or
     ranked signal. The VIX itself is noisy but a cutoff plus dwell is
     stable — flips only when the market genuinely re-rates volatility.
  3. **Dwell enforcement**: once we switch regimes, we hold for at least
     `min_dwell_days` (default 30) before we can switch again. Prevents
     regime flicker in choppy transition periods.

The signal itself is a `pd.Series` of {0, 1} indexed by date:
    1  = risk-on  → strategy runs at full equity allocation
    0  = risk-off → strategy reduces equity, holds more cash

The overlay applies as an equity multiplier in the backtest — see
`apply_regime_overlay_to_weights` for the composition rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    """Regime signal hyperparameters.

    Defaults chosen per the slowly-varying rule (memo: rule_no_whipsaw).
    Do not tune without recording a research note.

    Attributes:
        trend_sma_days: SPY moving-average window. 200 is the classic
            long-term trend filter.
        vix_threshold: VIX level below which vol is "supportive" for
            risk-on. 25 is the historic transition zone — above 25 is
            elevated, below 20 is calm. Between 20 and 25 the trend
            signal typically dominates.
        hysteresis_days: rolling-mean window applied to the raw AND-ed
            regime signal to smooth out single-day flips.
        min_dwell_days: minimum days a regime must hold before it can
            flip back.
        risk_off_equity_multiplier: fraction of target equity we hold
            when the signal reads risk-off. 0.60 means "reduce all
            position sizes to 60% and hold the remainder as cash".
    """
    trend_sma_days: int = 200
    vix_threshold: float = 25.0
    hysteresis_days: int = 10
    min_dwell_days: int = 30
    risk_off_equity_multiplier: float = 0.60


def compute_regime_signal(
    spy_close: pd.Series,
    vix_close: pd.Series,
    config: Optional[RegimeConfig] = None,
) -> pd.Series:
    """Compute the {0,1} risk-on/off signal.

    Args:
        spy_close: Daily SPY close prices, date-indexed ascending.
        vix_close: Daily VIX close, date-indexed ascending.
        config: Signal hyperparameters. Defaults per `RegimeConfig`.

    Returns:
        `pd.Series` of {0, 1} indexed by the intersection of the two
        input dates. First `trend_sma_days` entries are 1 (default
        risk-on) since the SMA is undefined — this defers to the
        strategy's natural behaviour before we have enough history.

    Signal construction (in order):
        raw_regime[t] = (SPY[t] > SPY_SMA[t]) AND (VIX[t] < vix_threshold)
        smoothed[t]   = rolling_mean(raw_regime, hysteresis_days) > 0.5
        final[t]      = enforce_min_dwell(smoothed, min_dwell_days)

    No look-ahead: every value at t uses only data at index <= t.
    """
    cfg = config or RegimeConfig()

    # Align on the common date index (inner join).
    idx = spy_close.index.intersection(vix_close.index)
    spy = spy_close.reindex(idx).astype(float)
    vix = vix_close.reindex(idx).astype(float)

    # Trend component: SPY above its N-day SMA.
    sma = spy.rolling(window=cfg.trend_sma_days, min_periods=cfg.trend_sma_days).mean()
    trend_ok = (spy > sma).astype(int)

    # Vol component: VIX below threshold.
    vol_ok = (vix < cfg.vix_threshold).astype(int)

    # Raw regime = both conditions satisfied.
    raw = (trend_ok & vol_ok).astype(int)

    # Hysteresis: smooth with a rolling mean and threshold at 0.5.
    # If more than half the last `hysteresis_days` were risk-on, we're on.
    smoothed_mean = raw.rolling(window=cfg.hysteresis_days, min_periods=1).mean()
    smoothed = (smoothed_mean > 0.5).astype(int)

    # Minimum-dwell enforcement.
    final = _enforce_min_dwell(smoothed, cfg.min_dwell_days)

    # Pre-SMA warmup: default to risk-on so early history doesn't get
    # spuriously flagged risk-off just because the SMA isn't computable yet.
    warmup_mask = sma.isna()
    final[warmup_mask] = 1

    return final


def _enforce_min_dwell(signal: pd.Series, min_dwell_days: int) -> pd.Series:
    """Force a regime to hold for at least `min_dwell_days` before flipping.

    Walk the series forward; when we see a proposed flip, only apply it
    if the current regime has been held for at least `min_dwell_days`.
    Otherwise, keep the current regime.

    Zero look-ahead — the decision at t uses only history <= t.
    """
    if min_dwell_days <= 1 or signal.empty:
        return signal.copy()

    out = np.array(signal.astype(int).to_numpy(), copy=True)
    current = int(out[0])
    dwell = 1
    for i in range(1, len(out)):
        proposed = int(out[i])
        if proposed != current and dwell < min_dwell_days:
            # Reject the flip; hold the current regime.
            out[i] = current
            dwell += 1
        elif proposed != current:
            # Accept the flip.
            current = proposed
            dwell = 1
        else:
            dwell += 1
    return pd.Series(out, index=signal.index, name=signal.name)


def apply_regime_overlay_to_weights(
    target_weights: pd.Series,
    regime_on: bool,
    config: Optional[RegimeConfig] = None,
) -> pd.Series:
    """Rescale target position weights by the regime multiplier.

    Args:
        target_weights: proposed portfolio weights (sum <= 1.0).
        regime_on: True (risk-on) → weights unchanged. False (risk-off)
            → weights multiplied by `risk_off_equity_multiplier`.
        config: hyperparameters.

    Returns:
        Rescaled Series. Cash allocation implicitly rises to fill the gap.
    """
    cfg = config or RegimeConfig()
    if regime_on:
        return target_weights.copy()
    return target_weights * cfg.risk_off_equity_multiplier
