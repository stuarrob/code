#!/usr/bin/env python3
"""Factor-weight diagnostic — grid search, multi-window stability, sensitivity.

Answers three questions the user posed on 2026-07-09:

  1. Are the factor weights in `configs/etf_smart_beta.toml` (momentum 35%,
     quality 30%, volatility 20%, value 15%) justified by out-of-sample
     performance, or just picked by intuition?

  2. Across a grid of weight vectors, which give the best risk-adjusted
     returns on our cached ETF history?

  3. Do the "best" weights hold across different look-back windows
     (3-year, 5-year, full, plus rolling 3-year slices)? If they wander,
     the invariance claim of factor investing is empirically weakened for
     this universe.

Plus a bonus:

  4. Sensitivity of Sharpe and Sortino to each factor's weight, evaluated
     at the current policy point. Answers "if I nudge momentum by 5%,
     what happens?"

Outputs a dated markdown report + PNG plots under
``docs/research/YYYY-MM_factor_weight_diagnostic.md`` (and a sibling
``_plots/`` directory).

Runs against whatever the freshest cache has — safe to re-run after each
weekly refresh to see how the picture evolves.

Design notes
------------
- Uses a simplified strategy simulator (top-N by integrated rank,
  exponential weights, monthly rebalance) rather than the full MVO
  backtest engine. This isolates the factor-weight effect from optimizer
  behaviour; keeps the grid search fast (~150 weight vectors × 4 windows
  runs in a few minutes rather than an hour).
- Rolling factor scores are cached to disk (``~/trade_data/ETFTrader/
  processed/rolling_factor_scores.parquet``) so re-runs only recompute
  when the underlying price data has advanced.
- All strategy metrics are computed on OUT-OF-SAMPLE returns (score at
  rebalance date t → hold portfolio until t+1). No look-ahead.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless PNG rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_collection.etf_filters import filter_leveraged_etfs
from src.factors import (
    MomentumFactor,
    QualityFactor,
    VolatilityFactor,
)
from src.portfolio.pipeline import DEFAULT_PROCESSED_DIR, collect_prices
from src.portfolio.policy import DEFAULT_POLICY_PATH, SmartBetaPolicy, load_policy


logger = logging.getLogger(__name__)

FACTORS = ("momentum", "quality", "volatility", "value")

DEFAULT_ROLLING_SCORES_PATH = (
    Path.home() / "trade_data" / "ETFTrader" / "processed" / "rolling_factor_scores.parquet"
)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "research"


# ────────────────────────────────────────────────────────────────
# Rolling factor scores
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RollingScores:
    """Per-factor time series of cross-sectional scores.

    Attributes:
        by_factor: {factor_name -> DataFrame indexed by rebalance date,
                    columns are tickers. Values are raw factor scores
                    (z-scores or fractional ranks per factor's calculate).
        rebalance_dates: The dates at which scores were computed.
        universe: All tickers present in at least one date's score frame.
        source_hash: Marker of the underlying price data used, so re-runs
                     can invalidate the cache when data changes.
    """

    by_factor: dict[str, pd.DataFrame]
    rebalance_dates: pd.DatetimeIndex
    universe: tuple[str, ...]
    source_hash: str


def _price_data_hash(prices: pd.DataFrame) -> str:
    """Cheap deterministic hash: shape + first / last date + column count."""
    return (
        f"{prices.shape[0]}x{prices.shape[1]}_"
        f"{prices.index.min():%Y%m%d}_{prices.index.max():%Y%m%d}"
    )


def compute_rolling_scores(
    prices: pd.DataFrame,
    policy: SmartBetaPolicy,
    rebalance_freq: str = "ME",
    warmup_days: int = 252,
    cache_path: Path = DEFAULT_ROLLING_SCORES_PATH,
    force: bool = False,
) -> RollingScores:
    """Compute cross-sectional factor scores at every rebalance date.

    Skips the value factor: it's proxied only by expense ratio, has no
    time-series interpretation, and would dominate the grid results with
    an artefact rather than a real signal. The diagnostic runs over three
    real factors (momentum / quality / volatility); after we identify the
    best real-factor blend, we can add value back as a small residual.
    """
    src_hash = _price_data_hash(prices)
    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        cache_hash = cached.attrs.get("source_hash") if hasattr(cached, "attrs") else None
        # Fallback: hash was written as a special row/col — check the file's mtime
        # against the price data. Simplest robust check: compare rebalance date
        # coverage to what we'd compute now.
        cached_dates = pd.DatetimeIndex(cached.index.get_level_values(0).unique())
        expected_dates = _rebalance_dates(prices, rebalance_freq, warmup_days)
        if set(expected_dates).issubset(set(cached_dates)):
            logger.info("Using cached rolling scores from %s", cache_path)
            return _unpack_cache(cached)

    logger.info("Computing rolling factor scores from scratch (this takes a few minutes)")
    dates = _rebalance_dates(prices, rebalance_freq, warmup_days)

    momentum_factor = MomentumFactor(
        lookback=policy.factor_lookbacks.momentum,
        skip_recent=policy.factor_lookbacks.momentum_skip_recent,
    )
    quality_factor = QualityFactor(lookback=policy.factor_lookbacks.quality)
    volatility_factor = VolatilityFactor(lookback=policy.factor_lookbacks.volatility)

    by_factor: dict[str, dict] = {"momentum": {}, "quality": {}, "volatility": {}}
    universe: set[str] = set()

    t0 = time.perf_counter()
    for i, date in enumerate(dates, 1):
        hist = prices.loc[:date]
        by_factor["momentum"][date] = momentum_factor.calculate(hist)
        by_factor["quality"][date] = quality_factor.calculate(hist)
        by_factor["volatility"][date] = volatility_factor.calculate(hist)
        universe.update(hist.columns)
        if i % 6 == 0 or i == len(dates):
            elapsed = time.perf_counter() - t0
            rate = i / max(elapsed, 1e-3)
            eta = (len(dates) - i) / max(rate, 1e-3)
            logger.info(
                "  %d/%d rebalance dates scored (%.1fs elapsed, ETA %.0fs)",
                i, len(dates), elapsed, eta,
            )

    scores = RollingScores(
        by_factor={
            k: pd.DataFrame(v).T.reindex(sorted(universe), axis=1)
            for k, v in by_factor.items()
        },
        rebalance_dates=pd.DatetimeIndex(dates),
        universe=tuple(sorted(universe)),
        source_hash=src_hash,
    )

    _persist_cache(scores, cache_path)
    return scores


def _rebalance_dates(
    prices: pd.DataFrame,
    freq: str,
    warmup_days: int,
) -> pd.DatetimeIndex:
    """Month-end rebalance dates, skipping the warmup window."""
    first_scorable = prices.index[0] + pd.Timedelta(days=int(warmup_days * 1.5))
    schedule = pd.date_range(first_scorable, prices.index[-1], freq=freq)
    # Snap to nearest available trading day.
    return pd.DatetimeIndex(
        [prices.index[prices.index.searchsorted(d, side="left") - 1]
         for d in schedule if d > prices.index[0]]
    ).unique()


def _persist_cache(scores: RollingScores, path: Path) -> None:
    """Serialize the RollingScores as a single tidy parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for factor_name, df in scores.by_factor.items():
        long = df.stack().rename("score").reset_index()
        long.columns = ["date", "ticker", "score"]
        long["factor"] = factor_name
        frames.append(long)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index(["date", "ticker", "factor"])
    combined.attrs["source_hash"] = scores.source_hash
    combined.to_parquet(path)


def _unpack_cache(cached: pd.DataFrame) -> RollingScores:
    """Reverse of _persist_cache."""
    cached = cached.reset_index()
    by_factor = {}
    universe = set()
    for factor_name, sub in cached.groupby("factor"):
        wide = sub.pivot(index="date", columns="ticker", values="score")
        by_factor[factor_name] = wide
        universe.update(wide.columns)
    dates = pd.DatetimeIndex(
        sorted(next(iter(by_factor.values())).index.unique())
    )
    return RollingScores(
        by_factor=by_factor,
        rebalance_dates=dates,
        universe=tuple(sorted(universe)),
        source_hash=cached.attrs.get("source_hash", "unknown"),
    )


# ────────────────────────────────────────────────────────────────
# Strategy simulator
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyMetrics:
    """Out-of-sample performance metrics for one (weights, window) run."""

    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate_monthly: float  # % of months with positive return
    turnover_avg: float  # portfolio-level avg per-rebalance turnover


def _integrated_scores(
    rolling: RollingScores,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Weighted-geometric-mean integration on percentile ranks."""
    active_factors = [f for f in ("momentum", "quality", "volatility") if weights.get(f, 0) > 0]
    if not active_factors:
        raise ValueError("Weight vector has zero on all real factors")

    ranks_by_factor = {
        f: rolling.by_factor[f].rank(axis=1, pct=True).fillna(0.5)
        for f in active_factors
    }
    total_weight = sum(weights[f] for f in active_factors)
    normalised = {f: weights[f] / total_weight for f in active_factors}

    integrated = None
    for f in active_factors:
        term = ranks_by_factor[f] ** normalised[f]
        integrated = term if integrated is None else integrated * term
    return integrated  # type: ignore[return-value]


def simulate_strategy(
    integrated_scores: pd.DataFrame,
    prices: pd.DataFrame,
    num_positions: int = 20,
    weighting_scheme: str = "exponential",
    txn_cost_bps: float = 5.0,
    window: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    rf_annual: float = 0.04,
) -> StrategyMetrics:
    """Top-N by integrated score, rebalance monthly, evaluate OOS returns.

    Args:
        integrated_scores: rebalance_date × ticker (fractional ranks).
        prices: daily price frame.
        num_positions: 20 per policy.
        weighting_scheme: 'exponential' (matches production RankBased) or 'equal'.
        txn_cost_bps: bps applied to turnover per rebalance (5 bps ≈ IB retail).
        window: (start, end) inclusive; None ⇒ use full integrated_scores range.
        rf_annual: annual risk-free rate for Sharpe / Sortino.
    """
    if window is not None:
        start, end = window
        integrated_scores = integrated_scores.loc[
            (integrated_scores.index >= start) & (integrated_scores.index <= end)
        ]

    if len(integrated_scores) < 2:
        return StrategyMetrics(
            cagr=float("nan"), volatility=float("nan"), sharpe=float("nan"),
            sortino=float("nan"), max_drawdown=float("nan"),
            hit_rate_monthly=float("nan"), turnover_avg=float("nan"),
        )

    # Build the sequence of (rebalance_date → target weights).
    target_weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    for date, row in integrated_scores.iterrows():
        eligible = row.dropna()
        if len(eligible) < num_positions:
            continue
        top = eligible.nlargest(num_positions)
        if weighting_scheme == "exponential":
            ranks = pd.Series(range(1, num_positions + 1), index=top.index)
            w = np.exp(-ranks / num_positions)
            w = w / w.sum()
        else:  # equal
            w = pd.Series(1.0 / num_positions, index=top.index)
        target_weights_by_date[date] = w

    if len(target_weights_by_date) < 2:
        return StrategyMetrics(
            cagr=float("nan"), volatility=float("nan"), sharpe=float("nan"),
            sortino=float("nan"), max_drawdown=float("nan"),
            hit_rate_monthly=float("nan"), turnover_avg=float("nan"),
        )

    rebalance_dates = sorted(target_weights_by_date.keys())

    # Compute daily portfolio returns.
    daily_rets = prices.pct_change()
    portfolio_dailies: list[pd.Series] = []
    turnovers: list[float] = []
    prev_weights = pd.Series(dtype=float)

    for i in range(len(rebalance_dates) - 1):
        d = rebalance_dates[i]
        d_next = rebalance_dates[i + 1]
        w = target_weights_by_date[d]
        # Trading-day slice: after d up to d_next
        mask = (daily_rets.index > d) & (daily_rets.index <= d_next)
        period = daily_rets.loc[mask, w.index.intersection(daily_rets.columns)]
        if period.empty:
            continue
        aligned = w.reindex(period.columns).fillna(0.0)
        r = (period * aligned).sum(axis=1)
        # Deduct txn cost on the first day of this holding period
        if not prev_weights.empty:
            combined_idx = w.index.union(prev_weights.index)
            turnover = float(
                (w.reindex(combined_idx).fillna(0)
                 - prev_weights.reindex(combined_idx).fillna(0)).abs().sum() / 2.0
            )
        else:
            turnover = 1.0  # initial build-up
        turnovers.append(turnover)
        if len(r) > 0:
            first = r.index[0]
            r.loc[first] = r.loc[first] - turnover * txn_cost_bps / 10_000.0
        portfolio_dailies.append(r)
        prev_weights = w

    if not portfolio_dailies:
        return StrategyMetrics(
            cagr=float("nan"), volatility=float("nan"), sharpe=float("nan"),
            sortino=float("nan"), max_drawdown=float("nan"),
            hit_rate_monthly=float("nan"), turnover_avg=float("nan"),
        )

    all_rets = pd.concat(portfolio_dailies)
    # Metrics
    total_years = (all_rets.index[-1] - all_rets.index[0]).days / 365.25
    cum = (1 + all_rets).prod()
    cagr = cum ** (1 / max(total_years, 1e-6)) - 1
    vol = all_rets.std() * np.sqrt(252)
    daily_rf = rf_annual / 252
    excess = all_rets - daily_rf
    sharpe = excess.mean() / (all_rets.std() + 1e-12) * np.sqrt(252)
    downside = all_rets[all_rets < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else float("nan")
    sortino = (excess.mean() * 252) / (downside_vol + 1e-12) if downside_vol == downside_vol else float("nan")
    cumret = (1 + all_rets).cumprod()
    peak = cumret.cummax()
    mdd = (cumret / peak - 1.0).min()
    monthly = (1 + all_rets).resample("ME").prod() - 1
    hit_rate = float((monthly > 0).mean()) if len(monthly) > 0 else float("nan")

    return StrategyMetrics(
        cagr=float(cagr), volatility=float(vol), sharpe=float(sharpe),
        sortino=float(sortino), max_drawdown=float(mdd),
        hit_rate_monthly=hit_rate, turnover_avg=float(np.mean(turnovers)) if turnovers else float("nan"),
    )


# ────────────────────────────────────────────────────────────────
# Grid + windows + sensitivity
# ────────────────────────────────────────────────────────────────

def generate_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    """All 3-factor weight vectors summing to 1.0 within ``step`` increments.

    Excludes value (proxied by expense ratio, no time-series interpretation)
    from the grid. Momentum / quality / volatility only.
    """
    steps = int(round(1.0 / step))
    out: list[dict[str, float]] = []
    for a in range(0, steps + 1):
        for b in range(0, steps + 1 - a):
            c = steps - a - b
            if c < 0:
                continue
            w = {
                "momentum": round(a * step, 6),
                "quality": round(b * step, 6),
                "volatility": round(c * step, 6),
                "value": 0.0,
            }
            # Skip degenerate corners with everything on one factor — noisy.
            nonzero = sum(v > 0 for v in w.values())
            if nonzero >= 2:
                out.append(w)
    return out


def named_policies() -> dict[str, dict[str, float]]:
    """Reference weight vectors to always evaluate."""
    return {
        "current_policy": {"momentum": 0.35, "quality": 0.30, "volatility": 0.20, "value": 0.15},
        "current_no_value": {"momentum": 0.412, "quality": 0.353, "volatility": 0.235, "value": 0.0},
        "equal_weight": {"momentum": 1/3, "quality": 1/3, "volatility": 1/3, "value": 0.0},
        "momentum_heavy": {"momentum": 0.60, "quality": 0.20, "volatility": 0.20, "value": 0.0},
        "quality_heavy": {"momentum": 0.20, "quality": 0.60, "volatility": 0.20, "value": 0.0},
        "defensive": {"momentum": 0.15, "quality": 0.35, "volatility": 0.50, "value": 0.0},
        "trend": {"momentum": 0.70, "quality": 0.15, "volatility": 0.15, "value": 0.0},
    }


def define_windows(rebalance_dates: pd.DatetimeIndex) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping and rolling 3-year windows."""
    last = rebalance_dates.max()
    first = rebalance_dates.min()
    windows = {"full": (first, last)}
    if last - pd.DateOffset(years=5) > first:
        windows["5yr"] = (last - pd.DateOffset(years=5), last)
    if last - pd.DateOffset(years=3) > first:
        windows["3yr"] = (last - pd.DateOffset(years=3), last)
    # Rolling 3-year windows every 12 months back from `last`.
    for step in (1, 2):
        end = last - pd.DateOffset(years=step)
        start = end - pd.DateOffset(years=3)
        if start < first:
            continue
        windows[f"roll_3yr_ending_{end:%Y-%m}"] = (start, end)
    return windows


def compute_sensitivity(
    rolling: RollingScores,
    prices: pd.DataFrame,
    base_weights: dict[str, float],
    steps: int = 11,
) -> pd.DataFrame:
    """Marginal Sharpe / Sortino as each factor's weight ranges 0 → 0.6.

    Other factors are held at their base weights and renormalised so the
    total sums to 1.0.
    """
    rows = []
    for factor in ("momentum", "quality", "volatility"):
        for x in np.linspace(0.0, 0.6, steps):
            base_others = {
                k: v for k, v in base_weights.items()
                if k != factor and k in ("momentum", "quality", "volatility")
            }
            total_others = sum(base_others.values())
            if total_others == 0:
                continue
            scale = (1.0 - x) / total_others
            weights = {factor: float(x)}
            for k, v in base_others.items():
                weights[k] = float(v * scale)
            weights["value"] = 0.0
            if any(v < 0 for v in weights.values()):
                continue
            integrated = _integrated_scores(rolling, weights)
            metrics = simulate_strategy(integrated, prices)
            rows.append({
                "factor": factor,
                "weight": x,
                "sharpe": metrics.sharpe,
                "sortino": metrics.sortino,
                "cagr": metrics.cagr,
                "max_drawdown": metrics.max_drawdown,
            })
    return pd.DataFrame(rows)


def compute_heatmap(
    rolling: RollingScores,
    prices: pd.DataFrame,
    axis1: str,
    axis2: str,
    third: str,
    steps: int = 11,
) -> pd.DataFrame:
    """Sharpe heatmap over (axis1, axis2) with third factor held at 0-remainder."""
    grid_vals = np.linspace(0.0, 1.0, steps)
    rows = []
    for a in grid_vals:
        for b in grid_vals:
            c = 1.0 - a - b
            if c < 0 or c > 1.0:
                continue
            weights = {axis1: float(a), axis2: float(b), third: float(c), "value": 0.0}
            integrated = _integrated_scores(rolling, weights)
            metrics = simulate_strategy(integrated, prices)
            rows.append({axis1: a, axis2: b, "sharpe": metrics.sharpe, "sortino": metrics.sortino})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────

def _plot_marginal_sensitivity(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for metric, ax in zip(("sharpe", "sortino"), axes):
        for factor in df["factor"].unique():
            sub = df[df["factor"] == factor].sort_values("weight")
            ax.plot(sub["weight"], sub[metric], marker="o", label=factor)
        ax.set_xlabel("factor weight")
        ax.set_ylabel(metric)
        ax.set_title(f"Marginal {metric} sensitivity (others held at base + renormalised)")
        ax.axvline(0.35, color="gray", linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_heatmaps(pairs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
    for (name, df), ax in zip(pairs.items(), axes[0]):
        axis1, axis2 = name.split(" vs ")
        pivot = df.pivot(index=axis2, columns=axis1, values="sharpe")
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0, cbar_kws={"label": "Sharpe"})
        ax.set_title(f"Sharpe: {axis1} vs {axis2}")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_stability(
    grid_results: pd.DataFrame,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    out_path: Path,
) -> None:
    """For each window, show the top-Sharpe weight vector's factor mix."""
    windows_ordered = list(windows.keys())
    tops = []
    for w in windows_ordered:
        sub = grid_results[grid_results["window"] == w].dropna(subset=["sharpe"])
        if sub.empty:
            continue
        best = sub.sort_values("sharpe", ascending=False).iloc[0]
        tops.append({
            "window": w,
            "momentum": best["w_momentum"],
            "quality": best["w_quality"],
            "volatility": best["w_volatility"],
            "sharpe": best["sharpe"],
        })
    if not tops:
        return
    df = pd.DataFrame(tops)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["momentum"], width, label="momentum")
    ax.bar(x, df["quality"], width, label="quality")
    ax.bar(x + width, df["volatility"], width, label="volatility")
    ax.set_xticks(x)
    ax.set_xticklabels(df["window"], rotation=30, ha="right")
    ax.set_ylabel("weight in top-Sharpe vector")
    ax.set_title("Weight-vector stability across windows (invariance check)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _min_max_robust(grid_results: pd.DataFrame, windows: list[str]) -> pd.Series:
    """Weight vector with the highest *worst-case* Sharpe across all windows."""
    key_cols = ["w_momentum", "w_quality", "w_volatility"]
    pivot = grid_results.pivot_table(
        index=key_cols, columns="window", values="sharpe",
    )
    pivot = pivot.reindex(columns=windows)
    pivot["worst"] = pivot.min(axis=1)
    best = pivot.sort_values("worst", ascending=False).iloc[0]
    return pd.Series({
        "momentum": best.name[0],
        "quality": best.name[1],
        "volatility": best.name[2],
        "worst_case_sharpe": best["worst"],
    })


def write_report(
    grid_results: pd.DataFrame,
    sensitivity: pd.DataFrame,
    heatmaps: dict[str, pd.DataFrame],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    plots_dir: Path,
    report_path: Path,
    policy: SmartBetaPolicy,
    price_load_source: str,
    price_load_range: tuple[pd.Timestamp, pd.Timestamp],
) -> None:
    """Produce the markdown research note with embedded plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    marg_png = plots_dir / "marginal_sensitivity.png"
    heat_png = plots_dir / "heatmaps.png"
    stab_png = plots_dir / "window_stability.png"
    _plot_marginal_sensitivity(sensitivity, marg_png)
    _plot_heatmaps(heatmaps, heat_png)
    _plot_stability(grid_results, windows, stab_png)

    # Top-Sharpe vector per window
    top_by_window = (
        grid_results
        .dropna(subset=["sharpe"])
        .sort_values("sharpe", ascending=False)
        .groupby("window")
        .head(5)
        .reset_index(drop=True)
    )

    robust = _min_max_robust(grid_results, list(windows.keys()))

    # Current-policy performance per window
    current = grid_results[grid_results["label"] == "current_policy"].copy()

    lines = [
        f"# Factor-weight diagnostic — {pd.Timestamp.now():%Y-%m-%d}",
        "",
        "Level-1 diagnostic per `docs/RESEARCH_BACKLOG.md` #1. Grid-search of",
        "factor weight vectors across multiple time windows, plus marginal",
        "sensitivity at the current policy point.",
        "",
        "## Inputs",
        "",
        f"- **Universe**: cached ETF price matrix from `{price_load_source}`",
        f"- **History**: {price_load_range[0]:%Y-%m-%d} → {price_load_range[1]:%Y-%m-%d}",
        f"- **Rebalance**: month-end, top-{policy.num_positions} by integrated rank, "
        "exponential weights (matches production `RankBasedOptimizer`)",
        "- **Transaction cost**: 5 bps per rebalance turnover",
        "- **Risk-free rate**: 4.0% p.a.",
        "- **Value factor excluded from the search** (proxied only by expense",
        "  ratio in the current codebase — no meaningful time-series signal).",
        "  Weights below are on the 3-factor simplex: momentum + quality + volatility.",
        "",
        "## Current policy performance by window",
        "",
        "| Window | Sharpe | Sortino | CAGR | MaxDD | Turnover |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in current.iterrows():
        lines.append(
            f"| {row['window']} | "
            f"{row['sharpe']:.3f} | {row['sortino']:.3f} | "
            f"{row['cagr']:.2%} | {row['max_drawdown']:.2%} | "
            f"{row['turnover_avg']:.2%} |"
        )
    lines += [
        "",
        "## Top-5 weight vectors by window (in-sample Sharpe)",
        "",
    ]
    for w_name in windows:
        sub = top_by_window[top_by_window["window"] == w_name]
        if sub.empty:
            continue
        lines += [f"### {w_name}", ""]
        lines += ["| momentum | quality | volatility | Sharpe | Sortino | CAGR | MaxDD |"]
        lines += ["|---|---|---|---|---|---|---|"]
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['w_momentum']:.2f} | {row['w_quality']:.2f} | {row['w_volatility']:.2f} | "
                f"{row['sharpe']:.3f} | {row['sortino']:.3f} | "
                f"{row['cagr']:.2%} | {row['max_drawdown']:.2%} |"
            )
        lines.append("")

    lines += [
        "## Robust weights (worst-case Sharpe across windows)",
        "",
        f"The weight vector whose *worst-case* Sharpe across all windows is highest —",
        f"the min-max robust choice under the invariance framing:",
        "",
        f"- momentum   = **{robust['momentum']:.2f}**",
        f"- quality    = **{robust['quality']:.2f}**",
        f"- volatility = **{robust['volatility']:.2f}**",
        f"- worst-case Sharpe = **{robust['worst_case_sharpe']:.3f}**",
        "",
        "If this vector differs materially from `configs/etf_smart_beta.toml`,",
        "the current weights are not invariance-robust and should be updated.",
        "",
        "## Weight-vector stability across windows",
        "",
        f"![stability]({stab_png.relative_to(report_path.parent)})",
        "",
        "Bars show the *top-Sharpe* weight vector for each window. If the bars",
        "swing dramatically between windows, the invariance claim is empirically",
        "false for this universe and history.",
        "",
        "## Marginal sensitivity at the current policy point",
        "",
        f"![marginal]({marg_png.relative_to(report_path.parent)})",
        "",
        "Sharpe / Sortino as each factor's weight varies with the other two held",
        "at their base ratio (renormalised). Dashed vertical line at 0.35 marks",
        "the current momentum weight.",
        "",
        "## 2D sensitivity heatmaps",
        "",
        f"![heatmaps]({heat_png.relative_to(report_path.parent)})",
        "",
        "## Provisional recommendation",
        "",
        "Update `configs/etf_smart_beta.toml` to the robust weights above,",
        "citing this note. Re-run after the next cache refresh to confirm",
        "the picture is stable; if drift is material, treat the current policy",
        "as informed-prior only and consider a regime-conditional variant",
        "(ADR-0002 candidate — see `docs/RESEARCH_BACKLOG.md` #1 level 3).",
        "",
        "## Method",
        "",
        f"- Script: `scripts/factor_weight_diagnostic.py`",
        f"- Rolling scores cache: `{DEFAULT_ROLLING_SCORES_PATH}`",
        f"- Value factor deliberately excluded from the grid — proxied only by",
        "  expense ratio, contributes no meaningful cross-sectional signal for",
        "  this exercise. If a real value factor (P/E, P/B, div yield) is",
        "  added later, re-run with a 4-factor grid.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", report_path)


# ────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-recompute-scores", action="store_true",
                        help="Ignore the rolling-score cache and recompute.")
    parser.add_argument("--grid-step", type=float, default=0.10,
                        help="Weight grid increment (default 0.10 → ~66 vectors on 3-factor simplex).")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Where to write the markdown + PNGs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
    )

    policy = load_policy()
    logger.info("Loaded policy: %s (v%d)", policy.name, policy.version)

    load = collect_prices(policy, processed_dir=DEFAULT_PROCESSED_DIR)
    logger.info(
        "Loaded prices: %s (%d tickers, %s → %s)",
        load.source, load.n_tickers, load.start_date.date(), load.end_date.date(),
    )
    all_tickers = load.prices.columns.tolist()
    basic = filter_leveraged_etfs(all_tickers)
    prices = load.prices[basic]
    logger.info("After leveraged/inverse exclusion: %d tickers", prices.shape[1])

    scores = compute_rolling_scores(
        prices, policy, force=args.force_recompute_scores,
    )
    logger.info("Rolling scores ready: %d rebalance dates, %d factors",
                len(scores.rebalance_dates), len(scores.by_factor))

    weight_grid = generate_weight_grid(step=args.grid_step)
    named = named_policies()
    logger.info("Evaluating %d grid + %d named = %d vectors",
                len(weight_grid), len(named), len(weight_grid) + len(named))

    windows = define_windows(scores.rebalance_dates)
    logger.info("Windows: %s", ", ".join(windows.keys()))

    rows = []

    # Named policies first
    for label, weights in named.items():
        integrated = _integrated_scores(scores, weights)
        for w_name, w_range in windows.items():
            metrics = simulate_strategy(integrated, prices, window=w_range)
            rows.append({
                "label": label,
                "window": w_name,
                "w_momentum": weights["momentum"],
                "w_quality": weights["quality"],
                "w_volatility": weights["volatility"],
                "sharpe": metrics.sharpe,
                "sortino": metrics.sortino,
                "cagr": metrics.cagr,
                "volatility": metrics.volatility,
                "max_drawdown": metrics.max_drawdown,
                "hit_rate_monthly": metrics.hit_rate_monthly,
                "turnover_avg": metrics.turnover_avg,
            })

    # Grid
    t0 = time.perf_counter()
    for i, weights in enumerate(weight_grid, 1):
        integrated = _integrated_scores(scores, weights)
        for w_name, w_range in windows.items():
            metrics = simulate_strategy(integrated, prices, window=w_range)
            rows.append({
                "label": "grid",
                "window": w_name,
                "w_momentum": weights["momentum"],
                "w_quality": weights["quality"],
                "w_volatility": weights["volatility"],
                "sharpe": metrics.sharpe,
                "sortino": metrics.sortino,
                "cagr": metrics.cagr,
                "volatility": metrics.volatility,
                "max_drawdown": metrics.max_drawdown,
                "hit_rate_monthly": metrics.hit_rate_monthly,
                "turnover_avg": metrics.turnover_avg,
            })
        if i % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / max(elapsed, 1e-3)
            eta = (len(weight_grid) - i) / max(rate, 1e-3)
            logger.info(
                "  grid %d/%d (%.1fs elapsed, ETA %.0fs)",
                i, len(weight_grid), elapsed, eta,
            )

    grid_results = pd.DataFrame(rows)

    logger.info("Sensitivity at current policy point...")
    base = {"momentum": 0.35, "quality": 0.30, "volatility": 0.20}
    sensitivity = compute_sensitivity(scores, prices, base)

    logger.info("Heatmaps...")
    heat = {
        "momentum vs quality": compute_heatmap(scores, prices, "momentum", "quality", "volatility"),
        "momentum vs volatility": compute_heatmap(scores, prices, "momentum", "volatility", "quality"),
    }

    report_dir = args.report_dir
    plots_dir = report_dir / f"{pd.Timestamp.now():%Y-%m}_plots"
    report_path = report_dir / f"{pd.Timestamp.now():%Y-%m}_factor_weight_diagnostic.md"
    write_report(
        grid_results=grid_results,
        sensitivity=sensitivity,
        heatmaps=heat,
        windows=windows,
        plots_dir=plots_dir,
        report_path=report_path,
        policy=policy,
        price_load_source=load.source,
        price_load_range=(load.start_date, load.end_date),
    )

    # Also dump the raw grid as CSV so the applet or future exercises can reload.
    grid_results.to_csv(report_dir / f"{pd.Timestamp.now():%Y-%m}_grid_results.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
