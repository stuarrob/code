"""Correlation-based clustering for top-N portfolio selection (T2.2).

Closes T2.2 in the adversarial-review roadmap. Replaces "top-N by score"
with "top-N by score, subject to a pairwise-correlation cap" so the
portfolio's effective diversification matches its nominal position count.

Design principles applied (slowly-varying rule):
  - **126-day rolling correlation window** — ~6 months. Short windows
    overfit to recent noise (whipsaw); long windows lag regime shifts.
  - **Quarterly refresh cadence** — the correlation matrix is computed
    once per quarter and held constant between refreshes. Weekly re-
    computation would introduce whipsaw of its own.
  - **Greedy pick with pairwise cap** — take the top-ranked candidate;
    skip subsequent candidates whose max abs correlation with any
    already-picked position exceeds `tau`. Simpler and less parameter-
    sensitive than k-means/hierarchical with a fixed k.

Baseline for comparison: `tau = 1.0` (no filtering) = the current
top-N-by-score behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusteringConfig:
    """Correlation-clustering hyperparameters.

    Attributes:
        tau: pairwise absolute-correlation cap. 1.0 disables clustering
            entirely; 0.60 forces aggressive spread; 0.80 is a
            reasonable central value.
        corr_window_days: rolling window used to compute the correlation
            matrix. 126 ~ 6 months.
        max_candidates_multiplier: when picking top-N with clustering we
            need more than N ranked candidates because some will be
            rejected. Draw from top (N * multiplier). Default 3 → for a
            30-position basket we look at top 90.
    """
    tau: float = 0.80
    corr_window_days: int = 126
    max_candidates_multiplier: int = 3


def compute_correlation_matrix(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int = 126,
) -> pd.DataFrame:
    """Compute the pairwise return correlation matrix over the trailing window.

    Args:
        prices: wide daily-price DataFrame indexed by date, columns are
            tickers.
        as_of: last date included in the window (inclusive).
        window_days: number of trading days in the window.

    Returns:
        DataFrame indexed and columned by ticker; NaN for pairs with
        insufficient overlap in the window.

    No look-ahead: uses only prices at index <= as_of.
    """
    if as_of not in prices.index:
        # Snap to the latest available bar at or before as_of.
        preceding = prices.index[prices.index <= as_of]
        if len(preceding) == 0:
            return pd.DataFrame()
        as_of = preceding[-1]

    hist = prices.loc[:as_of].tail(window_days + 1)
    if len(hist) < 2:
        return pd.DataFrame()
    rets = hist.pct_change().dropna(how="all")
    # min_periods = 0.6 of window forces reasonable coverage before we
    # trust a correlation. NaNs for thin pairs propagate through the
    # greedy filter as "cannot assess" → treated conservatively as
    # "cannot rule out high correlation" (skip).
    min_periods = max(2, int(0.60 * window_days))
    corr = rets.corr(min_periods=min_periods)
    return corr


def pick_with_correlation_cap(
    ranked_candidates: Iterable[str],
    corr_matrix: pd.DataFrame,
    num_positions: int,
    tau: float,
) -> list[str]:
    """Greedy top-N with correlation-cap filtering.

    Walks the ranked list in order. Accepts a candidate if its absolute
    correlation with every already-picked position is <= tau. Skips it
    otherwise. NaN correlations are treated as "unknown" and CONSERVATIVELY
    rejected — better to skip a candidate than accidentally load up on a
    hidden pair.

    Args:
        ranked_candidates: tickers in descending order of factor score.
        corr_matrix: symmetric correlation frame (from
            `compute_correlation_matrix`). Must contain each candidate as
            both a row and a column.
        num_positions: target number of positions to fill.
        tau: max allowed abs correlation with any already-picked. 1.0
            disables filtering (all candidates pass unless self-loop).

    Returns:
        list of picked tickers, in the order they were picked. May be
        shorter than num_positions if the candidate list is exhausted.
    """
    picked: list[str] = []
    if tau >= 1.0:
        # Fast path — no filtering.
        for c in ranked_candidates:
            picked.append(c)
            if len(picked) == num_positions:
                break
        return picked

    for candidate in ranked_candidates:
        if candidate in picked:
            continue
        if candidate not in corr_matrix.index:
            # No correlation data for this candidate — reject
            # conservatively. Downstream telemetry should log this.
            continue

        # Check candidate against every already-picked position.
        conflict = False
        for p in picked:
            if p not in corr_matrix.columns:
                # Same treatment: unknown → skip conservatively.
                conflict = True
                break
            val = corr_matrix.at[candidate, p]
            if pd.isna(val):
                conflict = True
                break
            if abs(val) > tau:
                conflict = True
                break

        if not conflict:
            picked.append(candidate)
            if len(picked) == num_positions:
                break

    return picked


class QuarterlyCorrelationCache:
    """Caches the correlation matrix on a quarterly cadence.

    Rebalance dates may be monthly or bimonthly; the correlation matrix
    is refreshed only once per calendar quarter. Between refreshes the
    same matrix is reused across all rebalances — the slowly-varying
    doctrine applied to the diversification model itself.

    Usage:
        cache = QuarterlyCorrelationCache()
        for date in rebalance_dates:
            corr = cache.get(prices, date, window_days=126)
    """

    def __init__(self) -> None:
        self._by_quarter_start: dict[pd.Timestamp, pd.DataFrame] = {}

    def get(self, prices: pd.DataFrame, as_of: pd.Timestamp,
            window_days: int = 126) -> pd.DataFrame:
        quarter_key = pd.Timestamp(as_of).to_period("Q").to_timestamp()
        if quarter_key not in self._by_quarter_start:
            self._by_quarter_start[quarter_key] = compute_correlation_matrix(
                prices, as_of, window_days=window_days,
            )
        return self._by_quarter_start[quarter_key]

    @property
    def n_quarters_cached(self) -> int:
        return len(self._by_quarter_start)


def apply_sector_cluster_cap(
    ranked_candidates: Iterable[str],
    sector_map: dict[str, str],
    num_positions: int,
    max_per_sector: int,
) -> list[str]:
    """Sector-cap variant — a simpler alternative to the correlation cap.

    Uses FMP `etf/info` sector data (mapped to a single dominant sector
    per ETF, or a hash-mapped bucket for multi-sector funds) to enforce
    a maximum ETFs per sector. Cheaper than correlation clustering and
    more interpretable, at the cost of not capturing cross-sector
    correlation clusters (e.g. a semiconductor ETF is technically "tech"
    but correlates heavily with certain industrial ETFs).

    Args:
        ranked_candidates: tickers in descending factor-score order.
        sector_map: ticker → sector-label. Tickers missing from the map
            are treated as their own sector (no constraint).
        num_positions: target holding count.
        max_per_sector: cap on ETFs sharing a sector label.

    Returns:
        list of picked tickers.
    """
    picked: list[str] = []
    sector_counts: dict[str, int] = {}
    for candidate in ranked_candidates:
        sector = sector_map.get(candidate, f"__unknown_{candidate}")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        picked.append(candidate)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(picked) == num_positions:
            break
    return picked
