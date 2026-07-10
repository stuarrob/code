"""MVO prior calibration — is a calibrated MVO better than the current RankBased?

Motivation
----------
The multi-regime backtest (2026-07-10) validated the 35/30/20/15 factor
weights against a RankBased optimiser using exponential rank weights. The
alternative MeanVarianceOptimizer has three additional priors —
`risk_aversion`, `axioma_penalty`, `shrinkage_strength` — that were never
calibrated. `robustness_penalty` and `turnover_penalty` in
`configs/etf_smart_beta.toml` are dead code and are not read by the MVO
implementation.

This script grid-searches over the three real MVO priors and reports:

1. The empirical winner within the MVO family (min-Sharpe robust).
2. The head-to-head vs the RankBased baseline (CAGR 13.04%, Sharpe 0.51,
   MaxDD -37% from the 2026-07 multi-regime run).
3. A pre-registered criterion for switching to MVO: MVO-best beats
   RankBased baseline on min-Sharpe across three windows by >=0.10 AND
   wins the mean-log-CAGR (time-average, non-ergodicity-friendly).

Design decisions
----------------
- **n_resample = 0** for the calibration pass (Michaud disabled). This
  keeps runtime at ~0.4s per rebalance date per config. If a config
  looks like a winner we re-run it with n_resample=50 (Michaud enabled)
  to confirm robustness in a Phase 2 pass.
- **Fixed universe screen** — identical to `multi_regime_diagnostic.py`
  (599-ticker curated list) so results are directly comparable.
- **Fixed factor weights** — 35/30/20/15 prior. MVO changes the sizing,
  not the ranking. Value factor still ignored in integration (matches
  RankBased comparison).

Writes to docs/research/2026-07_mvo_calibration_grid.csv and prints a
summary.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from src.portfolio.optimizer import MeanVarianceOptimizer
from src.data_collection.etf_filters import filter_universe


GRID_CSV = REPO_ROOT / "docs" / "research" / "2026-07_mvo_calibration_grid.csv"

# Pre-registered acceptance criterion (matches the multi-regime plan).
RANKBASED_BASELINE_MIN_SHARPE = 0.511  # from 2026-07 multi-regime results


@dataclass(frozen=True)
class MvoConfig:
    risk_aversion: float
    axioma_penalty: float
    shrinkage_strength: float
    n_resample: int = 0

    @property
    def label(self) -> str:
        return (f"mvo_r{self.risk_aversion:.2f}_"
                f"a{self.axioma_penalty:.3f}_"
                f"s{self.shrinkage_strength:.2f}_"
                f"n{self.n_resample}")


# ────────────────────────────────────────────────────────────────
# Data loading (matches multi_regime_diagnostic setup)
# ────────────────────────────────────────────────────────────────

def _load_inputs():
    from factor_weight_diagnostic import (
        _unpack_cache, DEFAULT_ROLLING_SCORES_PATH, _integrated_scores,
    )
    scores = _unpack_cache(pd.read_parquet(DEFAULT_ROLLING_SCORES_PATH))

    cache = Path.home() / "trade_data" / "ETFTrader" / "ib_historical"
    files = [p for p in cache.glob("*.parquet") if p.stem != "manifest"]
    kept = set(filter_universe([f.stem for f in files]))
    frames: dict[str, pd.Series] = {}
    for f in files:
        if f.stem not in kept:
            continue
        try:
            df = pd.read_parquet(f, columns=["close"])
            if len(df) > 20:
                frames[f.stem] = df["close"]
        except Exception:
            continue
    prices = pd.DataFrame(frames).sort_index()

    integrated = _integrated_scores(scores, {
        "momentum": 0.35, "quality": 0.30, "volatility": 0.20, "value": 0.15,
    })
    common = integrated.columns.intersection(prices.columns)
    integrated = integrated[common]
    return integrated, prices


# ────────────────────────────────────────────────────────────────
# Simulation
# ────────────────────────────────────────────────────────────────

def _metrics(returns: pd.Series, rf_annual: float = 0.04) -> dict:
    if len(returns) < 20:
        return {k: float("nan") for k in
                ("cagr", "vol", "sharpe", "sortino", "max_dd", "log_cagr")}
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cum = (1.0 + returns).prod()
    cagr = cum ** (1.0 / max(years, 1e-6)) - 1.0
    log_cagr = float(np.log1p(returns).mean() * 252)
    vol = float(returns.std() * np.sqrt(252))
    daily_rf = rf_annual / 252
    excess = returns - daily_rf
    sharpe = float(excess.mean() / (returns.std() + 1e-12) * np.sqrt(252))
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else float("nan")
    sortino = (float((excess.mean() * 252) / (downside_vol + 1e-12))
                if downside_vol == downside_vol else float("nan"))
    cumret = (1.0 + returns).cumprod()
    peak = cumret.cummax()
    max_dd = float((cumret / peak - 1.0).min())
    return {"cagr": float(cagr), "log_cagr": log_cagr, "vol": vol,
            "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd}


def _windows(rebalance_dates: pd.DatetimeIndex) -> dict:
    end = rebalance_dates[-1]
    out = {"full": (rebalance_dates[0], end)}
    for label, days in (("5y", 5 * 365), ("3y", 3 * 365)):
        start = end - pd.Timedelta(days=days)
        if start >= rebalance_dates[0]:
            out[label] = (start, end)
    return out


def _simulate_mvo(cfg: MvoConfig,
                   integrated: pd.DataFrame,
                   prices: pd.DataFrame,
                   num_positions: int = 30,
                   min_weight: float = 0.02,
                   max_weight: float = 0.15,
                   lookback: int = 60,
                   candidate_multiplier: int = 3,
                   txn_cost_bps: float = 5.0) -> pd.Series:
    """Backtest one MVO config. Returns daily portfolio-return series."""
    target_by_date: dict[pd.Timestamp, pd.Series] = {}

    for date, row in integrated.iterrows():
        eligible = row.dropna()
        if len(eligible) < num_positions:
            continue
        ranked = eligible.nlargest(num_positions * candidate_multiplier).index.tolist()
        hist = prices[ranked].loc[:date].tail(lookback + 1)
        if len(hist) < lookback:
            continue
        opt = MeanVarianceOptimizer(
            num_positions=num_positions,
            lookback=lookback,
            risk_aversion=cfg.risk_aversion,
            axioma_penalty=cfg.axioma_penalty,
            shrinkage_strength=cfg.shrinkage_strength,
            n_resample=cfg.n_resample,
            min_weight=min_weight,
            max_weight=max_weight,
            use_factor_scores_as_alpha=True,
        )
        try:
            w = opt.optimize(row[ranked], hist)
        except Exception:
            continue
        if w is None or w.empty or w.sum() < 0.5:
            continue
        target_by_date[date] = w[w > 0]

    if len(target_by_date) < 2:
        return pd.Series(dtype=float)

    dates = sorted(target_by_date.keys())
    daily_rets = prices.pct_change()
    parts: list[pd.Series] = []
    prev = pd.Series(dtype=float)

    for i in range(len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        w = target_by_date[d]
        cols = w.index.intersection(daily_rets.columns)
        mask = (daily_rets.index > d) & (daily_rets.index <= d_next)
        period = daily_rets.loc[mask, cols]
        if period.empty:
            continue
        aligned = w.reindex(period.columns).fillna(0.0)
        r = (period * aligned).sum(axis=1)

        if not prev.empty:
            union_idx = w.index.union(prev.index)
            turnover = float(
                (w.reindex(union_idx).fillna(0)
                 - prev.reindex(union_idx).fillna(0)).abs().sum() / 2.0
            )
        else:
            turnover = 1.0
        if len(r) > 0:
            r.loc[r.index[0]] -= turnover * txn_cost_bps / 10_000.0

        parts.append(r)
        prev = w

    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts)


# ────────────────────────────────────────────────────────────────
# Worker
# ────────────────────────────────────────────────────────────────

# Globals populated in worker init; keeps large frames out of task pickling.
_INTEGRATED: pd.DataFrame | None = None
_PRICES: pd.DataFrame | None = None


def _init_worker(integrated: pd.DataFrame, prices: pd.DataFrame) -> None:
    global _INTEGRATED, _PRICES
    _INTEGRATED = integrated
    _PRICES = prices


def _run_one(args: tuple[MvoConfig, str, pd.Timestamp, pd.Timestamp]) -> dict:
    cfg, window_label, start, end = args
    daily = _simulate_mvo(cfg, _INTEGRATED, _PRICES)
    if daily.empty:
        return {"config": cfg.label, "window": window_label,
                "risk_aversion": cfg.risk_aversion,
                "axioma_penalty": cfg.axioma_penalty,
                "shrinkage_strength": cfg.shrinkage_strength,
                "n_resample": cfg.n_resample,
                **{k: float("nan") for k in
                   ("cagr", "log_cagr", "vol", "sharpe", "sortino", "max_dd")}}
    sliced = daily.loc[(daily.index >= start) & (daily.index <= end)]
    m = _metrics(sliced)
    return {"config": cfg.label, "window": window_label,
            "risk_aversion": cfg.risk_aversion,
            "axioma_penalty": cfg.axioma_penalty,
            "shrinkage_strength": cfg.shrinkage_strength,
            "n_resample": cfg.n_resample, **m}


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--n-resample", type=int, default=0,
                    help="0 = fast survey (default); 50 = Michaud on winners")
    ap.add_argument("--configs", type=Path, default=None,
                    help="If set, run only the configs (label list) in this CSV.")
    args = ap.parse_args()

    print(f"mvo_calibration_diagnostic — workers={args.workers}, "
          f"n_resample={args.n_resample}")
    print("Loading inputs...")
    integrated, prices = _load_inputs()
    print(f"  integrated {integrated.shape}, prices {prices.shape}")

    windows = _windows(integrated.index)
    print(f"  windows: {list(windows.keys())}")

    # Grid.
    ra_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    ax_grid = [0.001, 0.01, 0.05, 0.1]
    sh_grid = [0.2, 0.5, 0.8]

    if args.configs and args.configs.exists():
        prior_configs = pd.read_csv(args.configs)
        configs = [
            MvoConfig(
                risk_aversion=r["risk_aversion"],
                axioma_penalty=r["axioma_penalty"],
                shrinkage_strength=r["shrinkage_strength"],
                n_resample=args.n_resample,
            )
            for _, r in prior_configs.iterrows()
        ]
    else:
        configs = [
            MvoConfig(risk_aversion=r, axioma_penalty=a,
                       shrinkage_strength=s, n_resample=args.n_resample)
            for r, a, s in itertools.product(ra_grid, ax_grid, sh_grid)
        ]

    tasks = [
        (cfg, w_label, w_start, w_end)
        for cfg in configs
        for w_label, (w_start, w_end) in windows.items()
    ]
    print(f"  {len(configs)} configs x {len(windows)} windows = {len(tasks)} tasks")

    t0 = time.perf_counter()
    with mp.Pool(args.workers, initializer=_init_worker,
                  initargs=(integrated, prices)) as pool:
        rows = []
        for i, res in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            rows.append(res)
            if i % 10 == 0 or i == len(tasks):
                elapsed = time.perf_counter() - t0
                eta = (len(tasks) - i) / max(i, 1) * elapsed
                print(f"  {i}/{len(tasks)}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    df = pd.DataFrame(rows)
    GRID_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GRID_CSV, index=False)
    print(f"\nGrid written to {GRID_CSV}")

    # Quick headline summary.
    print("\n" + "=" * 80)
    print("Top 10 configs by min-Sharpe across windows:")
    rob = df.groupby("config").agg(
        min_sharpe=("sharpe", "min"), mean_sharpe=("sharpe", "mean"),
        mean_cagr=("cagr", "mean"), mean_log_cagr=("log_cagr", "mean"),
        worst_dd=("max_dd", "min"),
        risk_aversion=("risk_aversion", "first"),
        axioma_penalty=("axioma_penalty", "first"),
        shrinkage_strength=("shrinkage_strength", "first"),
    ).reset_index().sort_values("min_sharpe", ascending=False)
    for _, r in rob.head(10).iterrows():
        print(f"  {r['config']:40s} min_S {r.min_sharpe:+.3f}  "
              f"mean_S {r.mean_sharpe:+.3f}  mean_CAGR {r.mean_cagr*100:+6.2f}%  "
              f"worst_DD {r.worst_dd*100:+6.2f}%")

    print(f"\nRankBased baseline min_Sharpe: {RANKBASED_BASELINE_MIN_SHARPE:+.3f}")
    best_mvo = rob.iloc[0]
    gap = best_mvo.min_sharpe - RANKBASED_BASELINE_MIN_SHARPE
    print(f"Best MVO min_Sharpe:            {best_mvo.min_sharpe:+.3f}  "
          f"(delta {gap:+.3f})")
    if gap >= 0.10:
        print("  → passes the +0.10 threshold. Recommend Michaud confirmation.")
    else:
        print("  → does NOT pass +0.10 threshold. RankBased retained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
