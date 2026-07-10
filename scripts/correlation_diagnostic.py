"""Correlation-clustering experiment — T2.2 backtest.

Tests whether adding a correlation-cap constraint to the top-N portfolio
selection produces a materially better outcome than the current
correlation-blind top-N-by-score.

Reuses machinery from `factor_weight_diagnostic.py` (rolling scores,
integrated scores, metric computation) and adds a clustering hook on
the weight-construction step. Runs multiple (tau, num_positions)
combinations and produces the results table + plot.

Design note: docs/research/2026-07_correlation_clustering_design.md.

Expected shape of result:
  - Baseline (tau=1.0): matches current behaviour.
  - Aggressive tau (0.60): CAGR lower (constraint bites into top ranks),
    max drawdown expected lower.
  - Moderate tau (0.80): sweet spot if signal is real; small CAGR cost
    for a real diversification benefit.

Important caveat baked into the report: the 2021-2026 window is
predominantly a tech-led bull run. A single-regime backtest will likely
show clustering hurts. The design note explains this expectation and
recommends re-running once a 2010-2020 backfill lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.portfolio.clustering import (
    ClusteringConfig,
    QuarterlyCorrelationCache,
    pick_with_correlation_cap,
)
from src.portfolio.policy import load_policy


REPORT_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs" / "research" / "2026-07_correlation_diagnostic.md"
)
PLOT_PATH = REPORT_PATH.with_suffix(".png")


def _metrics(returns: pd.Series, rf_annual: float = 0.04) -> dict:
    """Same metric set as regime_diagnostic — kept independent to avoid
    circular imports between diagnostic scripts."""
    if len(returns) < 2:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan,
                "sortino": np.nan, "max_dd": np.nan, "turnover": np.nan}
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cum = (1.0 + returns).prod()
    cagr = cum ** (1.0 / max(years, 1e-6)) - 1.0
    vol = returns.std() * np.sqrt(252)
    daily_rf = rf_annual / 252
    excess = returns - daily_rf
    sharpe = excess.mean() / (returns.std() + 1e-12) * np.sqrt(252)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = (
        (excess.mean() * 252) / (downside_vol + 1e-12)
        if downside_vol == downside_vol else np.nan
    )
    cumret = (1.0 + returns).cumprod()
    peak = cumret.cummax()
    max_dd = (cumret / peak - 1.0).min()
    return {
        "cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe),
        "sortino": float(sortino), "max_dd": float(max_dd),
    }


def _load_scores_and_prices():
    """Load cached rolling factor scores + current prices from the diagnostic."""
    # Import lazily to avoid heavy pandas/mpl at module import time.
    from scripts.factor_weight_diagnostic import (
        DEFAULT_ROLLING_SCORES_PATH,
        _unpack_cache,
        _integrated_scores,
    )
    if not DEFAULT_ROLLING_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Rolling scores cache missing at {DEFAULT_ROLLING_SCORES_PATH}. "
            "Run scripts/factor_weight_diagnostic.py first."
        )
    scores = _unpack_cache(pd.read_parquet(DEFAULT_ROLLING_SCORES_PATH))

    # Load prices from the IB historical cache.
    cache_dir = Path.home() / "trade_data" / "ETFTrader" / "ib_historical"
    from src.data_collection.comprehensive_etf_list import COMPREHENSIVE_ETF_UNIVERSE
    universe = set()
    for tickers in COMPREHENSIVE_ETF_UNIVERSE.values():
        universe.update(tickers)
    price_frames = {}
    for tkr in sorted(universe):
        p = cache_dir / f"{tkr}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["close"])
            price_frames[tkr] = df["close"]
        except Exception:
            continue
    prices = pd.DataFrame(price_frames).sort_index()
    return scores, prices, _integrated_scores


def simulate_with_clustering(
    integrated_scores: pd.DataFrame,
    prices: pd.DataFrame,
    num_positions: int,
    tau: float,
    corr_window_days: int = 126,
    max_candidates_multiplier: int = 3,
    txn_cost_bps: float = 5.0,
    weighting_scheme: str = "exponential",
    rf_annual: float = 0.04,
) -> pd.Series:
    """Backtest a single (num_positions, tau) combination.

    Returns the daily portfolio return series over the window.
    """
    corr_cache = QuarterlyCorrelationCache()
    target_weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    n_candidates = num_positions * max_candidates_multiplier

    for date, row in integrated_scores.iterrows():
        eligible = row.dropna()
        if len(eligible) < num_positions:
            continue
        # Top N * multiplier ranked candidates.
        ranked = eligible.nlargest(n_candidates).index.tolist()

        if tau >= 1.0:
            picked = ranked[:num_positions]
        else:
            corr = corr_cache.get(prices, date, window_days=corr_window_days)
            picked = pick_with_correlation_cap(ranked, corr, num_positions, tau)

        if len(picked) == 0:
            continue

        # Exponential rank weighting on picked list.
        if weighting_scheme == "exponential":
            ranks = pd.Series(range(1, len(picked) + 1), index=picked)
            w = np.exp(-ranks / num_positions)
            w = w / w.sum()
        else:
            w = pd.Series(1.0 / len(picked), index=picked)
        target_weights_by_date[date] = w

    if len(target_weights_by_date) < 2:
        return pd.Series(dtype=float)

    rebalance_dates = sorted(target_weights_by_date.keys())
    daily_rets = prices.pct_change()
    portfolio_dailies: list[pd.Series] = []
    prev_weights = pd.Series(dtype=float)

    for i in range(len(rebalance_dates) - 1):
        d = rebalance_dates[i]
        d_next = rebalance_dates[i + 1]
        w = target_weights_by_date[d]
        mask = (daily_rets.index > d) & (daily_rets.index <= d_next)
        cols = w.index.intersection(daily_rets.columns)
        period = daily_rets.loc[mask, cols]
        if period.empty:
            continue
        aligned = w.reindex(period.columns).fillna(0.0)
        r = (period * aligned).sum(axis=1)

        # Turnover-based transaction cost on the first day of holding.
        if not prev_weights.empty:
            union_idx = w.index.union(prev_weights.index)
            turnover = float(
                (w.reindex(union_idx).fillna(0)
                 - prev_weights.reindex(union_idx).fillna(0)).abs().sum() / 2.0
            )
        else:
            turnover = 1.0
        if len(r) > 0:
            first = r.index[0]
            r.loc[first] = r.loc[first] - turnover * txn_cost_bps / 10_000.0

        portfolio_dailies.append(r)
        prev_weights = w

    if not portfolio_dailies:
        return pd.Series(dtype=float)

    return pd.concat(portfolio_dailies)


def _format_pct(x: float) -> str:
    if not (x == x):
        return "  n/a"
    return f"{x*100:+.2f}%"


def _plot(cum_by_label: dict[str, pd.Series], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, series in cum_by_label.items():
        if len(series) == 0:
            continue
        ax.plot(series, label=label, linewidth=1.1)
    ax.set_ylabel("Cumulative return")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("T2.2 Correlation clustering — cumulative returns")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    print("Loading cached rolling scores + prices...")
    scores, prices, integrated_scores_fn = _load_scores_and_prices()
    policy = load_policy()

    # Use policy-configured factor weights.
    weights = policy.factor_weights.as_dict()
    integrated = integrated_scores_fn(scores, weights)
    # Restrict integrated scores to dates within the price cache.
    integrated = integrated.reindex(integrated.index.intersection(prices.index))
    # Restrict to tickers we actually have price data for, so ranked
    # candidates always have a return series to trade. Without this,
    # the top-N by score often lists tickers absent from `prices` and
    # their contribution collapses to zero, degrading the baseline.
    common_cols = integrated.columns.intersection(prices.columns)
    integrated = integrated[common_cols]
    print(f"universe: {len(integrated.columns)} tickers priced × {len(integrated)} rebalance dates")
    if len(integrated) < 3 or len(integrated.columns) < 30:
        print("ERROR: too few rebalance dates or tickers in the "
              "intersection of rolling scores and prices. Check cache "
              "states.", file=sys.stderr)
        return 1

    # Grid.
    tau_values = [1.0, 0.90, 0.80, 0.70, 0.60]  # 1.0 = baseline
    npos_values = [20, 30]  # historical + new policy

    results: list[dict] = []
    cum_curves: dict[str, pd.Series] = {}

    for npos in npos_values:
        for tau in tau_values:
            label = f"N={npos}, tau={tau:.2f}"
            print(f"  simulating {label}...")
            daily = simulate_with_clustering(
                integrated_scores=integrated,
                prices=prices,
                num_positions=npos,
                tau=tau,
            )
            m = _metrics(daily)
            m.update({"num_positions": npos, "tau": tau})
            results.append(m)
            if len(daily) > 0:
                cum_curves[label] = (1.0 + daily).cumprod()

    df = pd.DataFrame(results)
    _plot(cum_curves, PLOT_PATH)

    # Write report.
    lines = [
        "# T2.2 Correlation-Clustering Diagnostic",
        "",
        "**Date generated:** 2026-07-10",
        "",
        f"**Data window:** {integrated.index.min().date()} to "
        f"{integrated.index.max().date()} "
        f"({(integrated.index[-1] - integrated.index[0]).days / 365.25:.1f} years)",
        f"**Rebalance dates:** {len(integrated)} (from cached rolling factor scores)",
        f"**Universe:** {len(prices.columns)} tickers with cached price data",
        f"**Factor weights (from policy):** momentum={weights.get('momentum', 0):.2f}, "
        f"quality={weights.get('quality', 0):.2f}, "
        f"volatility={weights.get('volatility', 0):.2f}, "
        f"value={weights.get('value', 0):.2f}",
        "",
        "## Results",
        "",
        "| N positions | tau | CAGR | Vol | Sharpe | Sortino | Max DD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['num_positions']} | {r['tau']:.2f} | "
            f"{_format_pct(r['cagr'])} | {_format_pct(r['vol'])} | "
            f"{r['sharpe']:+.2f} | {r['sortino']:+.2f} | "
            f"{_format_pct(r['max_dd'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "**Baseline** (`tau = 1.0`) = current top-N-by-score behaviour.",
        "**Clustered** (`tau < 1.0`) filters out candidates whose absolute",
        "correlation with any already-picked position exceeds tau.",
        "",
        "**Expected shape in a bull-heavy window:** clustering should ",
        "underperform on CAGR — the winning cohort is highly correlated ",
        "(tech, growth), and forcing decorrelation trades winners for ",
        "lower-scored uncorrelated names. Look instead at max drawdown ",
        "and Sortino for the diversification premium.",
        "",
        "This test uses the 2021--2026 window from the cached rolling ",
        "scores. Per the design note (`docs/research/2026-07_correlation_",
        "clustering_design.md`), a multi-regime backtest across 2010--2020 ",
        "is where clustering is expected to show its true value. This ",
        "backfill is on the roadmap.",
        "",
        "## Plot",
        "",
        f"![Correlation clustering]({PLOT_PATH.name})",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Console summary.
    print()
    print("=" * 76)
    print(f"{'N':>3}  {'tau':>5}  {'CAGR':>9}  {'Vol':>9}  {'Sharpe':>8}  "
          f"{'Sortino':>8}  {'Max DD':>9}")
    print("-" * 76)
    for r in results:
        print(f"{r['num_positions']:>3}  {r['tau']:>5.2f}  "
              f"{_format_pct(r['cagr']):>9}  {_format_pct(r['vol']):>9}  "
              f"{r['sharpe']:>+8.2f}  {r['sortino']:>+8.2f}  "
              f"{_format_pct(r['max_dd']):>9}")
    print()
    print(f"Report: {REPORT_PATH}")
    print(f"Plot:   {PLOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
