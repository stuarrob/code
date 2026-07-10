"""Unit tests for the regime overlay signal.

Trading-logic module. Tests focus on the failure modes that would
silently corrupt the overlay:
  - Look-ahead in the SMA / hysteresis computation
  - Dwell not enforced (whipsaw returns)
  - Warmup rows accidentally flagged risk-off
  - Overlay multiplier applied in the wrong direction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.regime import (
    RegimeConfig,
    _enforce_min_dwell,
    apply_regime_overlay_to_weights,
    compute_regime_signal,
)


pytestmark = pytest.mark.unit


def _dates(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B")


class TestEnforceMinDwell:
    def test_single_flip_accepted(self):
        s = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=_dates(10))
        out = _enforce_min_dwell(s, min_dwell_days=3)
        # Dwell of 5 days on '1' before the flip → accepted.
        assert out.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    def test_short_dwell_rejected(self):
        # Flip after only 2 days should be rejected (dwell < 3).
        s = pd.Series([1, 1, 0, 1, 1, 1, 1, 1, 1, 1], index=_dates(10))
        out = _enforce_min_dwell(s, min_dwell_days=3)
        # The '0' at index 2 is rejected → still '1'.
        assert out.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_dwell_one_is_noop(self):
        s = pd.Series([1, 0, 1, 0, 1], index=_dates(5))
        out = _enforce_min_dwell(s, min_dwell_days=1)
        assert out.tolist() == [1, 0, 1, 0, 1]

    def test_empty_series(self):
        s = pd.Series([], dtype=int)
        out = _enforce_min_dwell(s, min_dwell_days=30)
        assert len(out) == 0


class TestComputeRegimeSignal:
    def test_warmup_defaults_to_risk_on(self):
        """Before the SMA is computable, signal must be 1 (risk-on)."""
        n = 250
        spy = pd.Series(100 + np.arange(n), index=_dates(n))
        vix = pd.Series(15.0, index=_dates(n))
        cfg = RegimeConfig(trend_sma_days=200)
        s = compute_regime_signal(spy, vix, cfg)
        # First 199 rows are pre-SMA warmup → all 1.
        assert (s.iloc[:199] == 1).all()

    def test_risk_off_when_spy_below_sma_and_vix_high(self):
        """Persistent bear (SPY < SMA and VIX > threshold) → risk-off."""
        n = 400
        # Build SPY that starts high, then drops for 100 days.
        spy_vals = np.concatenate([np.linspace(200, 220, 200),
                                    np.linspace(220, 180, 200)])
        spy = pd.Series(spy_vals, index=_dates(n))
        vix = pd.Series(np.concatenate([np.full(200, 15.0), np.full(200, 30.0)]),
                        index=_dates(n))
        cfg = RegimeConfig(trend_sma_days=100, vix_threshold=25.0,
                           hysteresis_days=10, min_dwell_days=30)
        s = compute_regime_signal(spy, vix, cfg)
        # By end of the down-leg, both trend and vol argue for risk-off.
        assert s.iloc[-1] == 0

    def test_dwell_prevents_flicker(self):
        """A single-day dip below SMA should not flip the regime."""
        n = 500
        spy = pd.Series(200.0, index=_dates(n))
        # Inject a single day where SPY dips below the SMA and VIX spikes.
        spy.iloc[250] = 100.0
        vix = pd.Series(15.0, index=_dates(n))
        vix.iloc[250] = 40.0
        cfg = RegimeConfig(trend_sma_days=100, vix_threshold=25.0,
                           hysteresis_days=10, min_dwell_days=30)
        s = compute_regime_signal(spy, vix, cfg)
        # The single-day dip is smoothed by hysteresis; regime stays risk-on.
        assert s.iloc[251] == 1

    def test_no_lookahead(self):
        """Signal at t must depend only on data at index <= t.

        Verified by extending the input series with a future spike and
        checking the earlier signal values are unchanged.
        """
        n = 400
        spy_base = pd.Series(200 + np.arange(n) * 0.01, index=_dates(n))
        vix_base = pd.Series(15.0, index=_dates(n))
        cfg = RegimeConfig(trend_sma_days=100)

        # Compute signal on the base series.
        s_base = compute_regime_signal(spy_base, vix_base, cfg)

        # Corrupt the LAST 50 days of both inputs (arbitrarily).
        spy_corrupt = spy_base.copy()
        spy_corrupt.iloc[-50:] = 50.0
        vix_corrupt = vix_base.copy()
        vix_corrupt.iloc[-50:] = 60.0
        s_corrupt = compute_regime_signal(spy_corrupt, vix_corrupt, cfg)

        # All values BEFORE the corruption must be identical.
        pd.testing.assert_series_equal(
            s_base.iloc[:-50], s_corrupt.iloc[:-50],
        )


class TestApplyOverlay:
    def test_on_leaves_weights_alone(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_regime_overlay_to_weights(w, regime_on=True)
        pd.testing.assert_series_equal(out, w)

    def test_off_scales_down(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        cfg = RegimeConfig(risk_off_equity_multiplier=0.60)
        out = apply_regime_overlay_to_weights(w, regime_on=False, config=cfg)
        assert out["A"] == pytest.approx(0.30)
        assert out["B"] == pytest.approx(0.30)
        # Implicit cash = 0.40
        assert 1.0 - out.sum() == pytest.approx(0.40)
