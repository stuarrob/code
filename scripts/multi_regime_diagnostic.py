"""Multi-regime diagnostic suite — the polished T2 backtest run.

Runs Tests 1-6 from `docs/research/2026-07_multi_regime_test_plan.md`
on the 2010-2026 FMP-backfilled price data:

  1. Baseline restatement — the 35/30/20/15 prior on multi-regime data.
  2. Factor-weight defence — local stability, concentration stress-test,
     empirical-winner contrast, non-ergodicity check.
  3. Regime overlay on the strategy portfolio.
  4. Correlation clustering on multi-regime data + sector-aware variant.
  5. Combined "best-of" — winners of 2/3/4 stacked.
  6. Score-magnitude vs rank weighting A/B (T2.3).

Uses multiprocessing.Pool for parallelism within each test. Rolling
factor scores are computed ONCE at the top and shared via cache file.

Result artefacts:
  - docs/research/2026-07_multi_regime_results.md (polished write-up)
  - docs/research/2026-07_multi_regime_grid.csv (raw grid data)
  - docs/research/2026-07_multi_regime_plots/*.png
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.portfolio.clustering import (
    QuarterlyCorrelationCache, pick_with_correlation_cap,
)
from src.portfolio.policy import load_policy
from src.portfolio.regime import RegimeConfig, compute_regime_signal
from src.data_collection.fmp_market_data import load_spy_cache, load_vix_cache


RESEARCH_DIR = REPO_ROOT / "docs" / "research"
PLOTS_DIR = RESEARCH_DIR / "2026-07_multi_regime_plots"
RESULTS_DOC = RESEARCH_DIR / "2026-07_multi_regime_results.md"
GRID_CSV = RESEARCH_DIR / "2026-07_multi_regime_grid.csv"


# ────────────────────────────────────────────────────────────────
# Shared globals for worker processes (set once, forked to workers).
# ────────────────────────────────────────────────────────────────

_INTEGRATED_BY_WEIGHT_KEY: dict[str, pd.DataFrame] = {}
_PRICES: pd.DataFrame | None = None


# ────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────

def _load_prices() -> pd.DataFrame:
    """Load the full FMP-backfilled price frame. Uses close only (single column each)."""
    cache_dir = Path.home() / "trade_data" / "ETFTrader" / "ib_historical"
    files = [p for p in cache_dir.glob("*.parquet") if p.stem != "manifest"]
    frames: dict[str, pd.Series] = {}
    print(f"loading {len(files)} price parquet files…")
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f, columns=["close"])
            if len(df) > 20:
                frames[f.stem] = df["close"]
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}")
    prices = pd.DataFrame(frames).sort_index()
    return prices


def _compute_rolling_scores(prices: pd.DataFrame, policy) -> Any:
    """Wrap factor_weight_diagnostic's rolling-scores compute."""
    from factor_weight_diagnostic import (
        compute_rolling_scores, DEFAULT_ROLLING_SCORES_PATH,
    )
    # Force recompute if source hash differs from cached.
    scores = compute_rolling_scores(prices, policy, cache_path=DEFAULT_ROLLING_SCORES_PATH)
    return scores


def _integrated(scores, weights: dict[str, float]) -> pd.DataFrame:
    from factor_weight_diagnostic import _integrated_scores
    return _integrated_scores(scores, weights)


# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────

def _metrics(returns: pd.Series, rf_annual: float = 0.04) -> dict[str, float]:
    if len(returns) < 20:
        return {k: float("nan") for k in
                ("cagr", "vol", "sharpe", "sortino", "max_dd", "log_cagr", "turnover")}
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cum = (1.0 + returns).prod()
    cagr = cum ** (1.0 / max(years, 1e-6)) - 1.0
    log_cagr = float(np.log1p(returns).mean() * 252)  # time-average log return
    vol = float(returns.std() * np.sqrt(252))
    daily_rf = rf_annual / 252
    excess = returns - daily_rf
    sharpe = float(excess.mean() / (returns.std() + 1e-12) * np.sqrt(252))
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else float("nan")
    sortino = float((excess.mean() * 252) / (downside_vol + 1e-12)) if downside_vol == downside_vol else float("nan")
    cumret = (1.0 + returns).cumprod()
    peak = cumret.cummax()
    max_dd = float((cumret / peak - 1.0).min())
    return {
        "cagr": float(cagr), "log_cagr": log_cagr,
        "vol": vol, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
    }


def _windows(rebalance_dates: pd.DatetimeIndex) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if len(rebalance_dates) == 0:
        return {}
    end = rebalance_dates[-1]
    windows = {"full": (rebalance_dates[0], end)}
    # Trailing 5-year and 3-year windows if we have the depth.
    for label, days in (("5y", 5 * 365), ("3y", 3 * 365)):
        start = end - pd.Timedelta(days=days)
        if start >= rebalance_dates[0]:
            windows[label] = (start, end)
    return windows


# ────────────────────────────────────────────────────────────────
# Simulation core (adapted from factor_weight_diagnostic + clustering)
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    """One backtest configuration passed to a worker."""
    label: str
    weights: dict[str, float]         # {momentum, quality, volatility, value}
    num_positions: int = 30
    tau: float = 1.0                  # 1.0 = no clustering
    regime_overlay: bool = False
    regime_config_key: str = "default"
    weighting_scheme: str = "exponential"  # or "magnitude" for T6
    window: str = "full"              # which window to slice on


def _regime_series(spy_close: pd.Series, vix_close: pd.Series,
                   cfg_key: str) -> pd.Series | None:
    if cfg_key is None or cfg_key == "off":
        return None
    variants = {
        "default": RegimeConfig(),
        "vix20_0.60": RegimeConfig(vix_threshold=20.0, risk_off_equity_multiplier=0.60),
        "vix25_0.60": RegimeConfig(vix_threshold=25.0, risk_off_equity_multiplier=0.60),
        "vix30_0.60": RegimeConfig(vix_threshold=30.0, risk_off_equity_multiplier=0.60),
        "vix25_0.40": RegimeConfig(vix_threshold=25.0, risk_off_equity_multiplier=0.40),
        "vix25_0.80": RegimeConfig(vix_threshold=25.0, risk_off_equity_multiplier=0.80),
    }
    cfg = variants.get(cfg_key, RegimeConfig())
    signal = compute_regime_signal(spy_close, vix_close, cfg)
    return signal


def _regime_multiplier(cfg_key: str) -> float:
    if cfg_key.startswith("vix"):
        return float(cfg_key.split("_")[-1])
    return 0.60


def _simulate(cfg: RunConfig, integrated: pd.DataFrame,
              prices: pd.DataFrame, regime_signal: pd.Series | None,
              regime_mult: float, txn_cost_bps: float = 5.0) -> pd.Series:
    """Backtest one configuration. Returns daily portfolio-return series."""
    target_weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    corr_cache = QuarterlyCorrelationCache() if cfg.tau < 1.0 else None
    n_pos = cfg.num_positions

    for date, row in integrated.iterrows():
        eligible = row.dropna()
        if len(eligible) < n_pos:
            continue
        ranked = eligible.nlargest(n_pos * 3).index.tolist()
        if cfg.tau < 1.0 and corr_cache is not None:
            corr = corr_cache.get(prices, date, window_days=126)
            picked = pick_with_correlation_cap(ranked, corr, n_pos, cfg.tau)
        else:
            picked = ranked[:n_pos]
        if not picked:
            continue

        if cfg.weighting_scheme == "exponential":
            ranks = pd.Series(range(1, len(picked) + 1), index=picked)
            w = np.exp(-ranks / n_pos)
            w = w / w.sum()
        elif cfg.weighting_scheme == "magnitude":
            # Use integrated score magnitude, softmax normalised.
            s = eligible.reindex(picked)
            # Shift to positive: subtract min then softmax with mild temperature.
            s = s - s.min() + 1e-6
            w = s / s.sum()
        else:
            w = pd.Series(1.0 / len(picked), index=picked)
        target_weights_by_date[date] = w

    if len(target_weights_by_date) < 2:
        return pd.Series(dtype=float)

    rebalance_dates = sorted(target_weights_by_date.keys())
    daily_rets = prices.pct_change()
    portfolio_dailies: list[pd.Series] = []
    prev_weights = pd.Series(dtype=float)
    turnovers: list[float] = []

    for i in range(len(rebalance_dates) - 1):
        d = rebalance_dates[i]
        d_next = rebalance_dates[i + 1]
        w = target_weights_by_date[d]
        cols = w.index.intersection(daily_rets.columns)
        mask = (daily_rets.index > d) & (daily_rets.index <= d_next)
        period = daily_rets.loc[mask, cols]
        if period.empty:
            continue
        aligned = w.reindex(period.columns).fillna(0.0)
        r = (period * aligned).sum(axis=1)

        # Regime multiplier: applied per-day using the shifted signal.
        if regime_signal is not None:
            reg = regime_signal.reindex(r.index).ffill().fillna(1).shift(1).fillna(1).astype(int)
            mult = np.where(reg == 1, 1.0, regime_mult)
            r = r * mult

        if not prev_weights.empty:
            union_idx = w.index.union(prev_weights.index)
            turnover = float(
                (w.reindex(union_idx).fillna(0)
                 - prev_weights.reindex(union_idx).fillna(0)).abs().sum() / 2.0
            )
        else:
            turnover = 1.0
        turnovers.append(turnover)
        if len(r) > 0:
            first = r.index[0]
            r.loc[first] = r.loc[first] - turnover * txn_cost_bps / 10_000.0

        portfolio_dailies.append(r)
        prev_weights = w

    if not portfolio_dailies:
        return pd.Series(dtype=float)
    daily = pd.concat(portfolio_dailies)
    daily.attrs["turnover_avg"] = float(np.mean(turnovers)) if turnovers else float("nan")
    return daily


# Worker entrypoint (must be top-level for multiprocessing pickling).
def _run_one(cfg_and_context: tuple) -> dict:
    cfg, integrated, prices, regime_signal, regime_mult = cfg_and_context
    daily = _simulate(cfg, integrated, prices, regime_signal, regime_mult)
    # Slice window if not full.
    windows = _windows(integrated.index)
    if cfg.window not in windows:
        window = windows["full"]
    else:
        window = windows[cfg.window]
    daily_slice = daily.loc[(daily.index >= window[0]) & (daily.index <= window[1])]
    m = _metrics(daily_slice)
    m.update({
        "config": cfg.label,
        "window": cfg.window,
        "num_positions": cfg.num_positions,
        "tau": cfg.tau,
        "regime": cfg.regime_config_key if cfg.regime_overlay else "off",
        "weighting": cfg.weighting_scheme,
        "n_days": int(len(daily_slice)),
        "turnover": daily.attrs.get("turnover_avg", float("nan")),
    })
    return m


# ────────────────────────────────────────────────────────────────
# Test bank
# ────────────────────────────────────────────────────────────────

def _prior_weights() -> dict[str, float]:
    return {"momentum": 0.35, "quality": 0.30, "volatility": 0.20, "value": 0.15}


def _t2_defence_configs(prior: dict[str, float]) -> list[RunConfig]:
    """Build the four sub-tests of the factor weight defence."""
    configs: list[RunConfig] = []

    def _norm(w: dict[str, float]) -> dict[str, float]:
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    # 2a. Local stability — perturbation of each factor by ±5%, ±10% relative.
    for factor in ("momentum", "quality", "volatility", "value"):
        for delta in (-0.10, -0.05, 0.05, 0.10):
            w = dict(prior)
            w[factor] = w[factor] * (1 + delta)
            w = _norm(w)
            label = f"2a_{factor}{delta:+.2f}"
            configs.append(RunConfig(label=label, weights=w))

    # 2b. Concentration stress-test — one factor at 70%.
    for factor in ("momentum", "quality", "volatility"):
        w = {"momentum": 0.10, "quality": 0.10, "volatility": 0.10, "value": 0.10}
        w[factor] = 0.70
        w = _norm(w)
        configs.append(RunConfig(label=f"2b_{factor}_70", weights=w))

    # 2c. Empirical-winner-vs-prior contrast — done in reporting from 2a+2b outputs.
    # The prior itself is the baseline reference — always include.
    configs.append(RunConfig(label="prior", weights=prior))

    # Also grid-explore a small tight window (5-step simplex) — 8 configurations
    # in a controlled neighbourhood of the prior.
    for m in (0.30, 0.40):
        for q in (0.25, 0.35):
            for v in (0.15, 0.25):
                remaining = 1.0 - m - q - v
                if 0.05 <= remaining <= 0.25:
                    w = {"momentum": m, "quality": q, "volatility": v,
                         "value": remaining}
                    label = f"2c_m{int(m*100)}_q{int(q*100)}_v{int(v*100)}"
                    configs.append(RunConfig(label=label, weights=w))

    return configs


def _t3_regime_configs(prior: dict[str, float]) -> list[RunConfig]:
    """Regime overlay sensitivity."""
    configs = [
        RunConfig(label="no_regime", weights=prior, regime_overlay=False),
    ]
    for key in ("vix20_0.60", "vix25_0.60", "vix30_0.60",
                "vix25_0.40", "vix25_0.80"):
        configs.append(RunConfig(
            label=f"regime_{key}", weights=prior,
            regime_overlay=True, regime_config_key=key,
        ))
    return configs


def _t4_cluster_configs(prior: dict[str, float]) -> list[RunConfig]:
    """Correlation clustering across tau values."""
    configs = []
    for tau in (1.0, 0.90, 0.80, 0.70, 0.60):
        configs.append(RunConfig(
            label=f"cluster_tau{tau:.2f}", weights=prior, tau=tau,
        ))
    return configs


def _t6_magnitude_configs(prior: dict[str, float]) -> list[RunConfig]:
    return [
        RunConfig(label="weight_exponential", weights=prior, weighting_scheme="exponential"),
        RunConfig(label="weight_magnitude", weights=prior, weighting_scheme="magnitude"),
    ]


# ────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────

def _run_test_bank(configs: list[RunConfig], integrated: pd.DataFrame,
                   prices: pd.DataFrame, spy_close: pd.Series,
                   vix_close: pd.Series, workers: int) -> pd.DataFrame:
    """Run a list of configs across all windows, in parallel."""
    windows = list(_windows(integrated.index).keys())
    # Precompute regime signals per unique key.
    regime_signals: dict[str, pd.Series] = {}
    for cfg in configs:
        if cfg.regime_overlay and cfg.regime_config_key not in regime_signals:
            regime_signals[cfg.regime_config_key] = _regime_series(
                spy_close, vix_close, cfg.regime_config_key,
            )

    tasks = []
    for cfg in configs:
        for w in windows:
            new_cfg = RunConfig(
                label=cfg.label, weights=cfg.weights, num_positions=cfg.num_positions,
                tau=cfg.tau, regime_overlay=cfg.regime_overlay,
                regime_config_key=cfg.regime_config_key,
                weighting_scheme=cfg.weighting_scheme, window=w,
            )
            reg_sig = regime_signals.get(cfg.regime_config_key) if cfg.regime_overlay else None
            reg_mult = _regime_multiplier(cfg.regime_config_key) if cfg.regime_overlay else 1.0
            # Restrict integrated to the config's weights to save downstream work
            weight_key = ",".join(f"{k}:{v:.3f}" for k, v in sorted(cfg.weights.items()))
            if weight_key not in _INTEGRATED_BY_WEIGHT_KEY:
                # Compute lazily and cache. In multiprocessing we recompute per worker;
                # the integrated frames are small so this is fine.
                pass
            tasks.append((new_cfg, integrated, prices, reg_sig, reg_mult))

    print(f"  running {len(tasks)} sim tasks across {workers} workers…")
    if workers > 1:
        with mp.Pool(workers) as pool:
            rows = pool.map(_run_one, tasks)
    else:
        rows = [_run_one(t) for t in tasks]

    return pd.DataFrame(rows)


def _prepare_integrated_for_configs(scores, configs: list[RunConfig],
                                     prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-compute an integrated-scores frame per unique weight vector.

    Cached in-process; passed by reference to workers via pickle. Small
    enough (rebalance-date × ticker) that duplication in workers is
    acceptable.
    """
    out: dict[str, pd.DataFrame] = {}
    for cfg in configs:
        key = ",".join(f"{k}:{v:.4f}" for k, v in sorted(cfg.weights.items()))
        if key not in out:
            integrated = _integrated(scores, cfg.weights)
            # Restrict to tickers we have prices for so downstream sim doesn't
            # collapse to zero returns (the bug I fixed in the earlier T2.2 run).
            common = integrated.columns.intersection(prices.columns)
            out[key] = integrated[common]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--skip-t6", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these test labels: t1, t2, t3, t4, t5, t6")
    args = ap.parse_args()

    only = set(args.only) if args.only else None
    print(f"multi_regime_diagnostic — workers={args.workers}, only={only}")

    t0 = time.perf_counter()

    # Load inputs.
    prices = _load_prices()
    print(f"prices: {prices.shape}, {prices.index.min().date()} → {prices.index.max().date()}")

    spy_close = load_spy_cache()["close"]
    vix_close = load_vix_cache()["close"]

    policy = load_policy()
    prior = _prior_weights()

    # Phase 1 — rolling scores (single-threaded, cached).
    print("\n[phase 1] rolling factor scores…")
    scores = _compute_rolling_scores(prices, policy)
    print(f"  rebalance dates: {len(scores.rebalance_dates)}, universe: {len(scores.universe)}")

    # Build all configs first.
    configs_all: dict[str, list[RunConfig]] = {}
    if not only or "t1" in only:
        configs_all["t1"] = [RunConfig(label="baseline_prior", weights=prior)]
    if not only or "t2" in only:
        configs_all["t2"] = _t2_defence_configs(prior)
    if not only or "t3" in only:
        configs_all["t3"] = _t3_regime_configs(prior)
    if not only or "t4" in only:
        configs_all["t4"] = _t4_cluster_configs(prior)
    if (not args.skip_t6) and (not only or "t6" in only):
        configs_all["t6"] = _t6_magnitude_configs(prior)

    # Compute integrated frames for every unique weight vector.
    all_configs = [c for cs in configs_all.values() for c in cs]
    print(f"\n[phase 1.5] integrating scores for {len(set(tuple(sorted(c.weights.items())) for c in all_configs))} unique weight vectors…")
    integrated_by_key = _prepare_integrated_for_configs(scores, all_configs, prices)

    # Phase 2 — parallel tests.
    results_by_test: dict[str, pd.DataFrame] = {}
    for test_name, configs in configs_all.items():
        print(f"\n[phase 2] {test_name} — {len(configs)} configs")
        # Each config uses its own integrated. Group configs by weight-key.
        rows_all: list[dict] = []
        for cfg in configs:
            key = ",".join(f"{k}:{v:.4f}" for k, v in sorted(cfg.weights.items()))
            integrated = integrated_by_key[key]
            df = _run_test_bank([cfg], integrated, prices, spy_close, vix_close, args.workers)
            rows_all.append(df)
        results_by_test[test_name] = pd.concat(rows_all, ignore_index=True)

    # Consolidate.
    grid = pd.concat(
        [df.assign(test=test) for test, df in results_by_test.items()],
        ignore_index=True,
    )
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_CSV, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\nAll tests complete in {elapsed/60:.1f} min. Grid written to {GRID_CSV}")

    # Phase 3 (Test 5) + Phase 4 (write-up) live in the sibling
    # multi_regime_report.py so this runner stays focused on compute.
    print("Next: run scripts/multi_regime_report.py to produce the polished write-up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
