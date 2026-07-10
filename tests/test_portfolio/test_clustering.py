"""Unit tests for correlation-clustering module (T2.2).

Trading-logic module. Tests focus on the failure modes:
  - Look-ahead in correlation matrix computation
  - Greedy pick using wrong direction of correlation cap
  - NaN correlation not treated conservatively
  - Sector cap over-picks a hot sector
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.clustering import (
    ClusteringConfig,
    QuarterlyCorrelationCache,
    apply_sector_cluster_cap,
    compute_correlation_matrix,
    pick_with_correlation_cap,
)


pytestmark = pytest.mark.unit


def _dates(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="B")


class TestComputeCorrelationMatrix:
    def test_perfectly_correlated_pair(self):
        n = 150
        idx = _dates(n)
        # A rises 1% every day, B rises exactly the same → corr = 1.0.
        prices = pd.DataFrame({
            "A": (1.01 ** np.arange(n)) * 100,
            "B": (1.01 ** np.arange(n)) * 200,
        }, index=idx)
        corr = compute_correlation_matrix(prices, as_of=idx[-1], window_days=126)
        assert corr.at["A", "B"] == pytest.approx(1.0)

    def test_anticorrelated_pair(self):
        n = 150
        idx = _dates(n)
        rng = np.random.default_rng(42)
        r_a = rng.normal(0.001, 0.01, n)
        r_b = -r_a  # exact anticorrelation
        prices = pd.DataFrame({
            "A": 100 * np.cumprod(1 + r_a),
            "B": 100 * np.cumprod(1 + r_b),
        }, index=idx)
        corr = compute_correlation_matrix(prices, as_of=idx[-1], window_days=126)
        assert corr.at["A", "B"] == pytest.approx(-1.0)

    def test_no_lookahead(self):
        """Correlation at t must not depend on data after t."""
        n = 300
        idx = _dates(n)
        rng = np.random.default_rng(0)
        prices_base = pd.DataFrame({
            "A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
            "B": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
        }, index=idx)
        as_of = idx[200]
        corr_base = compute_correlation_matrix(prices_base, as_of, window_days=126)

        # Corrupt post-t prices arbitrarily.
        prices_corrupt = prices_base.copy()
        prices_corrupt.iloc[201:] = 999.0
        corr_corrupt = compute_correlation_matrix(prices_corrupt, as_of, window_days=126)

        # Correlations at as_of must be identical.
        pd.testing.assert_frame_equal(corr_base, corr_corrupt)

    def test_snaps_as_of_to_available_bar(self):
        """When as_of is a weekend, we snap to the preceding business day."""
        n = 150
        idx = _dates(n)
        prices = pd.DataFrame({"A": np.arange(n) + 100.0}, index=idx)
        # Sunday between two business days.
        weekend = idx[100] + pd.Timedelta(days=1)
        corr = compute_correlation_matrix(prices, as_of=weekend, window_days=50)
        assert "A" in corr.columns


class TestPickWithCorrelationCap:
    def _corr_frame(self, mapping: dict[tuple[str, str], float]) -> pd.DataFrame:
        """Build a full symmetric corr frame from a sparse mapping. Diagonal 1.0."""
        tickers = sorted({t for pair in mapping for t in pair})
        corr = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers,
                            dtype=float)
        for (a, b), v in mapping.items():
            corr.at[a, b] = v
            corr.at[b, a] = v
        return corr

    def test_no_filter_when_tau_is_one(self):
        corr = self._corr_frame({("A", "B"): 0.99, ("A", "C"): 0.99, ("B", "C"): 0.99})
        picked = pick_with_correlation_cap(["A", "B", "C"], corr,
                                            num_positions=3, tau=1.0)
        assert picked == ["A", "B", "C"]

    def test_high_correlation_pair_rejected(self):
        corr = self._corr_frame({("A", "B"): 0.95, ("A", "C"): 0.20, ("B", "C"): 0.20})
        picked = pick_with_correlation_cap(["A", "B", "C"], corr,
                                            num_positions=2, tau=0.80)
        # A picked first, B blocked (0.95 > 0.80), C picked.
        assert picked == ["A", "C"]

    def test_only_pair_over_tau_gets_dropped(self):
        corr = self._corr_frame({
            ("A", "B"): 0.30, ("A", "C"): 0.20, ("A", "D"): 0.20,
            ("B", "C"): 0.85, ("B", "D"): 0.20, ("C", "D"): 0.20,
        })
        picked = pick_with_correlation_cap(["A", "B", "C", "D"], corr,
                                            num_positions=4, tau=0.80)
        # A picked. B picked (only 0.30 with A). C blocked by 0.85 with B. D picked.
        assert picked == ["A", "B", "D"]

    def test_nan_correlation_treated_conservatively(self):
        """A NaN correlation should reject the candidate rather than silently accept."""
        corr = self._corr_frame({("A", "B"): 0.10})
        corr.at["A", "B"] = float("nan")
        corr.at["B", "A"] = float("nan")
        picked = pick_with_correlation_cap(["A", "B"], corr,
                                            num_positions=2, tau=0.80)
        # A picked. B has NaN with A → conservatively rejected.
        assert picked == ["A"]

    def test_abs_correlation_used(self):
        """Highly-anticorrelated pair (-0.95) should also be filtered."""
        corr = self._corr_frame({("A", "B"): -0.95})
        picked = pick_with_correlation_cap(["A", "B"], corr,
                                            num_positions=2, tau=0.80)
        assert picked == ["A"]

    def test_stops_at_num_positions(self):
        corr = self._corr_frame({
            ("A", "B"): 0.1, ("A", "C"): 0.1, ("A", "D"): 0.1,
            ("B", "C"): 0.1, ("B", "D"): 0.1, ("C", "D"): 0.1,
        })
        picked = pick_with_correlation_cap(["A", "B", "C", "D"], corr,
                                            num_positions=2, tau=0.80)
        assert picked == ["A", "B"]

    def test_missing_candidate_skipped(self):
        corr = self._corr_frame({("A", "B"): 0.1})
        picked = pick_with_correlation_cap(["A", "MISSING", "B"], corr,
                                            num_positions=3, tau=0.80)
        assert picked == ["A", "B"]


class TestQuarterlyCache:
    def test_same_quarter_uses_cached_matrix(self):
        n = 300
        idx = _dates(n, start="2024-01-01")
        rng = np.random.default_rng(1)
        prices = pd.DataFrame({
            "A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
            "B": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
        }, index=idx)
        cache = QuarterlyCorrelationCache()
        # Two dates within the same quarter should share a cached matrix.
        d1 = pd.Timestamp("2024-02-15")
        d2 = pd.Timestamp("2024-03-15")
        c1 = cache.get(prices, d1, window_days=100)
        c2 = cache.get(prices, d2, window_days=100)
        assert cache.n_quarters_cached == 1
        pd.testing.assert_frame_equal(c1, c2)

    def test_different_quarters_recompute(self):
        n = 300
        idx = _dates(n, start="2024-01-01")
        rng = np.random.default_rng(2)
        prices = pd.DataFrame({
            "A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
            "B": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
        }, index=idx)
        cache = QuarterlyCorrelationCache()
        cache.get(prices, pd.Timestamp("2024-02-15"), window_days=100)
        cache.get(prices, pd.Timestamp("2024-05-15"), window_days=100)
        cache.get(prices, pd.Timestamp("2024-08-15"), window_days=100)
        assert cache.n_quarters_cached == 3


class TestSectorCap:
    def test_cap_enforced(self):
        sector_map = {
            "AAA": "tech", "BBB": "tech", "CCC": "tech", "DDD": "tech",
            "EEE": "value", "FFF": "value",
        }
        # tech dominates the ranking but only 2 allowed.
        picked = apply_sector_cluster_cap(
            ["AAA", "BBB", "CCC", "EEE", "DDD", "FFF"],
            sector_map, num_positions=4, max_per_sector=2,
        )
        # AAA, BBB accepted (tech=2). CCC blocked. EEE accepted (value=1).
        # DDD blocked. FFF accepted (value=2).
        assert picked == ["AAA", "BBB", "EEE", "FFF"]

    def test_unknown_ticker_is_own_sector(self):
        picked = apply_sector_cluster_cap(
            ["X", "Y"], sector_map={}, num_positions=2, max_per_sector=1,
        )
        # Both unknown → different sectors → both accepted.
        assert picked == ["X", "Y"]
