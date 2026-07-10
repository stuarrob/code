"""Regime overlay diagnostic — T2.1 backtest.

Tests whether the SPY 200d + VIX regime signal with hysteresis and dwell
would have materially improved a passive SPY-hold's drawdown-adjusted
returns across 2010-2026 (16 years covering EU crisis 2011, 2015-16 China,
2018 Q4 correction, 2020 COVID, 2022 rate hikes).

This is a signal-quality test, not a full-portfolio test. The full-portfolio
overlay lives in `pipeline.py` and is exercised by the top-30 strategy
backtest once T2.2 lands. The purpose here is to prove the regime signal
itself is producing sensible on/off timing before we spend hours running it
inside a 792-ETF portfolio simulation.

Reports:
  - Regime-on days, regime-off days, number of switches
  - Passive SPY: CAGR, vol, Sharpe, max drawdown
  - Regime-overlay SPY: same metrics
  - Delta on each metric; whether the overlay is worth the complexity

Writes: docs/research/2026-07_regime_diagnostic.md and a plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_collection.fmp_market_data import load_spy_cache, load_vix_cache
from src.portfolio.regime import RegimeConfig, compute_regime_signal


REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "research" / "2026-07_regime_diagnostic.md"
PLOT_PATH = Path(__file__).resolve().parent.parent / "docs" / "research" / "2026-07_regime_diagnostic.png"


def _metrics(returns: pd.Series, rf_annual: float = 0.04) -> dict:
    """Standard performance metrics on a daily-return series."""
    if len(returns) < 2:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan,
                "sortino": np.nan, "max_dd": np.nan}
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
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
    }


def _apply_overlay(spy_rets: pd.Series, regime: pd.Series,
                   risk_off_mult: float) -> pd.Series:
    """Apply the regime multiplier to daily SPY returns.

    Regime signal at t applies to the return realised over (t, t+1] — this
    is what a live implementation would do: signal decided from close t,
    trade at close t, exposure held over the next day.
    """
    aligned_regime = regime.reindex(spy_rets.index).ffill().fillna(1).astype(int)
    # Shift the regime forward by one bar so we don't act on today's close
    # within today's return (no look-ahead).
    signal_shifted = aligned_regime.shift(1).fillna(1).astype(int)
    multiplier = np.where(signal_shifted == 1, 1.0, risk_off_mult)
    return spy_rets * multiplier


def _switch_count(regime: pd.Series) -> int:
    """Number of on↔off transitions in the signal."""
    diffs = regime.diff().fillna(0).abs()
    return int(diffs.sum())


def _format_pct(x: float) -> str:
    return f"{x*100:+.2f}%"


def _plot(spy_cum: pd.Series, overlay_cum: pd.Series,
          regime: pd.Series, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(spy_cum, label="SPY passive", linewidth=1.2)
    ax1.plot(overlay_cum, label="SPY + regime overlay", linewidth=1.2)
    ax1.set_ylabel("Cumulative return")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_title("Regime overlay diagnostic — 2010–2026")

    # Shade the risk-off bands.
    off = regime.reindex(spy_cum.index).fillna(1) == 0
    ax2.fill_between(spy_cum.index, 0, off.astype(int), step="pre",
                     alpha=0.5, color="tab:orange", label="risk-off")
    ax2.set_ylabel("regime")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    spy = load_spy_cache()
    vix = load_vix_cache()
    if spy is None or vix is None:
        print("ERROR: SPY or VIX cache missing. Run fetch_spy_history / "
              "fetch_vix_history first.", file=sys.stderr)
        return 1

    spy_close = spy["close"]
    vix_close = vix["close"]

    # Config: canonical slowly-varying defaults per RegimeConfig.
    cfg = RegimeConfig(
        trend_sma_days=200,
        vix_threshold=25.0,
        hysteresis_days=10,
        min_dwell_days=30,
        risk_off_equity_multiplier=0.60,
    )
    regime = compute_regime_signal(spy_close, vix_close, cfg)

    # Daily SPY returns aligned to regime dates.
    common = spy_close.index.intersection(regime.index)
    spy_rets = spy_close.reindex(common).pct_change().fillna(0.0)
    regime_common = regime.reindex(common)

    overlay_rets = _apply_overlay(spy_rets, regime_common, cfg.risk_off_equity_multiplier)

    passive_m = _metrics(spy_rets.loc[spy_rets.index[1]:])
    overlay_m = _metrics(overlay_rets.loc[overlay_rets.index[1]:])

    # Regime statistics.
    n_days = len(regime_common)
    off_days = int((regime_common == 0).sum())
    on_days = n_days - off_days
    switches = _switch_count(regime_common)
    off_periods_avg_days = off_days / max(1, switches // 2) if switches > 0 else off_days

    # Report + plot.
    spy_cum = (1.0 + spy_rets).cumprod()
    overlay_cum = (1.0 + overlay_rets).cumprod()
    _plot(spy_cum, overlay_cum, regime_common, PLOT_PATH)

    lines = [
        "# T2.1 Regime Overlay Diagnostic",
        "",
        f"**Date generated:** 2026-07-10",
        f"**Data window:** {common[0].date()} to {common[-1].date()} "
        f"({(common[-1] - common[0]).days / 365.25:.1f} years)",
        f"**Source:** SPY + ^VIX daily from FMP `historical-price-eod/full`",
        "",
        "## Config (per `RegimeConfig` defaults)",
        "",
        f"- Trend SMA: {cfg.trend_sma_days} days",
        f"- VIX threshold: {cfg.vix_threshold}",
        f"- Hysteresis window: {cfg.hysteresis_days} days",
        f"- Minimum dwell: {cfg.min_dwell_days} days",
        f"- Risk-off equity multiplier: {cfg.risk_off_equity_multiplier}",
        "",
        "## Regime statistics",
        "",
        f"- Total trading days: **{n_days}**",
        f"- Risk-on days: **{on_days}** ({on_days/n_days*100:.1f}%)",
        f"- Risk-off days: **{off_days}** ({off_days/n_days*100:.1f}%)",
        f"- Total switches (on↔off): **{switches}**",
        f"- Average risk-off spell: **~{off_periods_avg_days:.0f} days** "
        f"(if switches > 0)",
        "",
        "## Performance",
        "",
        "| Metric | SPY passive | SPY + overlay | Delta |",
        "|---|---:|---:|---:|",
        f"| CAGR | {_format_pct(passive_m['cagr'])} | {_format_pct(overlay_m['cagr'])} | "
        f"{_format_pct(overlay_m['cagr'] - passive_m['cagr'])} |",
        f"| Volatility | {_format_pct(passive_m['vol'])} | {_format_pct(overlay_m['vol'])} | "
        f"{_format_pct(overlay_m['vol'] - passive_m['vol'])} |",
        f"| Sharpe | {passive_m['sharpe']:+.2f} | {overlay_m['sharpe']:+.2f} | "
        f"{overlay_m['sharpe'] - passive_m['sharpe']:+.2f} |",
        f"| Sortino | {passive_m['sortino']:+.2f} | {overlay_m['sortino']:+.2f} | "
        f"{overlay_m['sortino'] - passive_m['sortino']:+.2f} |",
        f"| Max drawdown | {_format_pct(passive_m['max_dd'])} | "
        f"{_format_pct(overlay_m['max_dd'])} | "
        f"{_format_pct(overlay_m['max_dd'] - passive_m['max_dd'])} |",
        "",
        "## Interpretation",
        "",
        "The regime overlay's job is to **improve drawdown-adjusted return**, not",
        "necessarily raw CAGR. In a bull-heavy sample the overlay may lag on CAGR by",
        "sitting out productive risk-on days that briefly triggered risk-off, but",
        "should show meaningful gains on max drawdown and Sortino.",
        "",
        "Read the CAGR delta not as \"does the overlay make more money\" but as \"what",
        "premium do we pay for the drawdown protection\". A modest CAGR drag paired",
        "with a large max-drawdown improvement is the intended outcome; a CAGR drag",
        "with no drawdown improvement is a sign the signal is too jumpy.",
        "",
        "## Plot",
        "",
        f"![Regime overlay]({PLOT_PATH.name})",
        "",
        "## Files",
        "",
        f"- Report: `{REPORT_PATH.relative_to(Path(__file__).resolve().parent.parent).as_posix()}`",
        f"- Plot: `{PLOT_PATH.relative_to(Path(__file__).resolve().parent.parent).as_posix()}`",
        f"- Module under test: `src/portfolio/regime.py`",
        f"- Unit tests: `tests/test_portfolio/test_regime.py`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 60)
    print("T2.1 Regime Overlay — Diagnostic Summary")
    print("=" * 60)
    print(f"Window: {common[0].date()} to {common[-1].date()} ({(common[-1]-common[0]).days/365.25:.1f}y)")
    print(f"On/off days: {on_days}/{off_days}  switches: {switches}")
    print()
    print(f"{'Metric':<16}{'Passive':>12}{'Overlay':>12}{'Delta':>12}")
    print(f"{'CAGR':<16}{_format_pct(passive_m['cagr']):>12}{_format_pct(overlay_m['cagr']):>12}"
          f"{_format_pct(overlay_m['cagr']-passive_m['cagr']):>12}")
    print(f"{'Sharpe':<16}{passive_m['sharpe']:>+12.2f}{overlay_m['sharpe']:>+12.2f}"
          f"{overlay_m['sharpe']-passive_m['sharpe']:>+12.2f}")
    print(f"{'Sortino':<16}{passive_m['sortino']:>+12.2f}{overlay_m['sortino']:>+12.2f}"
          f"{overlay_m['sortino']-passive_m['sortino']:>+12.2f}")
    print(f"{'Max DD':<16}{_format_pct(passive_m['max_dd']):>12}{_format_pct(overlay_m['max_dd']):>12}"
          f"{_format_pct(overlay_m['max_dd']-passive_m['max_dd']):>12}")
    print()
    print(f"Report: {REPORT_PATH}")
    print(f"Plot:   {PLOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
