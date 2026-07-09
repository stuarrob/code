"""ETF smart-beta pipeline — shared entry points for the applet + notebooks.

The three functions here are the *only* interface the applet and any
notebook should call to drive the ETF smart-beta flow. They wrap the
underlying `src.factors`, `src.data_collection.etf_filters`, and
`src.portfolio.optimizer` machinery so business logic never lives in a
notebook cell or a Streamlit page.

Per ADR-0001:
- The deterministic pipeline decides all numbers (universe, scores,
  weights). The applet and LLM narrator only display the results.
- Every tuneable parameter is read from :class:`SmartBetaPolicy` — no
  magic constants inside this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.data_collection.etf_filters import filter_leveraged_etfs
from src.factors import (
    FactorIntegrator,
    MomentumFactor,
    QualityFactor,
    SimplifiedValueFactor,
    VolatilityFactor,
)
from src.portfolio.optimizer import (
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
    RankBasedOptimizer,
    SimpleOptimizer,
)
from src.portfolio.policy import SmartBetaPolicy

logger = logging.getLogger(__name__)

DEFAULT_IB_CACHE_DIR = Path.home() / "trade_data" / "ETFTrader" / "ib_historical"
DEFAULT_PROCESSED_DIR = Path.home() / "trade_data" / "ETFTrader" / "processed"
REFRESH_IB_CLIENT_ID = 31  # Distinct from ib_state's read-only client (30).

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of :func:`refresh_prices_from_ib`.

    Attributes:
        n_current: Tickers already up-to-date (no IB request made).
        n_stale: Tickers that were incrementally updated.
        n_missing: Tickers that were freshly downloaded.
        n_universe: Total tickers in the ETF universe.
        parquet_path: Path where the combined price matrix was saved.
    """

    n_current: int
    n_stale: int
    n_missing: int
    n_universe: int
    parquet_path: Path

_MIN_HISTORY_DAYS = 252
_MAX_MISSING_PCT = 10.0
_PROCESSED_FILENAMES = (
    "etf_prices_db.parquet",
    "etf_prices_ib.parquet",
    "etf_prices_filtered.parquet",
)


# ────────────────────────────────────────────────────────────────
# Return types
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PriceLoad:
    """Outcome of :func:`collect_prices`.

    Attributes:
        prices: Wide DataFrame — dates × tickers, forward-filled.
        source: Human-readable label of the parquet file that was loaded.
        start_date: First price date in the frame.
        end_date: Last price date in the frame.
        n_tickers: Number of tickers surviving the quality filter.
    """

    prices: pd.DataFrame
    source: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_tickers: int


@dataclass(frozen=True)
class ScoringResult:
    """Outcome of :func:`score_factors`.

    Attributes:
        combined_scores: Integrated factor score per ticker (0-1 rank).
        factor_scores: Per-ticker per-factor DataFrame.
        active_weights: Weights actually used after any value-factor
            skip; sums to 1.0.
        universe: Tickers surviving the leveraged/quality filters.
    """

    combined_scores: pd.Series
    factor_scores: pd.DataFrame
    active_weights: dict[str, float]
    universe: tuple[str, ...]


# ────────────────────────────────────────────────────────────────
# 1. Collect
# ────────────────────────────────────────────────────────────────

def collect_prices(
    policy: SmartBetaPolicy,
    processed_dir: Path,
    max_history_years: float = 5.0,
) -> PriceLoad:
    """Load the most recent cached ETF price matrix and apply the quality filter.

    Priority order (matches s2_collect.py): Databento → IB → yfinance.

    Args:
        policy: The active policy (used for the min-history requirement
            via ``factor_lookbacks.momentum``).
        processed_dir: Directory containing the ``etf_prices_*.parquet``
            files (typically ``~/trade_data/ETFTrader/processed``).
        max_history_years: Trim history to this many trailing years so
            factor scoring never sees data older than we care about.

    Returns:
        :class:`PriceLoad` with a clean, forward-filled wide price frame.

    Raises:
        FileNotFoundError: If no cached parquet is present.
        ValueError: If the loaded frame has fewer than 20 tickers with
            enough history to score.
    """
    processed_dir = Path(processed_dir)
    parquet = _resolve_cached_prices(processed_dir)
    prices = pd.read_parquet(parquet)

    # Trim to requested history window (avoids scoring on 20-yr-old data).
    if len(prices) > 0:
        cutoff = prices.index.max() - pd.Timedelta(days=int(max_history_years * 366))
        prices = prices.loc[prices.index >= cutoff]

    prices = _apply_quality_filter(
        prices,
        min_days=max(policy.factor_lookbacks.momentum, _MIN_HISTORY_DAYS),
        max_missing_pct=_MAX_MISSING_PCT,
    )

    if prices.shape[1] < 20:
        raise ValueError(
            f"Only {prices.shape[1]} tickers passed the quality filter — "
            f"the universe is too small to score. Refresh the price cache."
        )

    return PriceLoad(
        prices=prices,
        source=parquet.name,
        start_date=prices.index.min(),
        end_date=prices.index.max(),
        n_tickers=prices.shape[1],
    )


def cache_status(
    cache_dir: Path = DEFAULT_IB_CACHE_DIR,
    stale_days: int = 1,
) -> dict[str, object]:
    """Classify the per-ticker parquet cache without touching IB.

    Returns counts for ``current`` / ``stale`` / ``missing`` against the
    full ETF universe, plus ``latest_bar`` (max last-date across all
    per-ticker parquets) and ``n_universe``. Cheap enough to call on
    every applet rerun.

    A ticker is:
      - ``current`` if its cache exists and the latest bar is within
        ``stale_days`` of today,
      - ``stale`` if its cache exists but the gap exceeds ``stale_days``,
      - ``missing`` if there is no readable parquet (absent, empty, or
        unreadable).
    """
    from src.data_collection.comprehensive_etf_list import load_full_universe

    tickers, _ = load_full_universe()
    cache_dir = Path(cache_dir)
    today = pd.Timestamp.now().normalize()

    current = stale = missing = 0
    latest_bar: Optional[pd.Timestamp] = None
    for t in tickers:
        cache_file = cache_dir / f"{t}.parquet"
        if not cache_file.exists():
            missing += 1
            continue
        try:
            df = pd.read_parquet(cache_file)
            if df.empty:
                missing += 1
                continue
            last_date = pd.Timestamp(df.index.max())
            gap_days = (today - last_date).days
            if gap_days <= stale_days:
                current += 1
            else:
                stale += 1
            if latest_bar is None or last_date > latest_bar:
                latest_bar = last_date
        except Exception:  # noqa: BLE001 — unreadable parquet ≈ missing
            missing += 1

    return {
        "n_current": current,
        "n_stale": stale,
        "n_missing": missing,
        "n_universe": len(tickers),
        "latest_bar": latest_bar,
    }


def refresh_prices_from_ib(
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = REFRESH_IB_CLIENT_ID,
    cache_dir: Path = DEFAULT_IB_CACHE_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    duration: str = "5 Y",
    rate_limit_interval: float = 12.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> RefreshResult:
    """Refresh the per-ticker parquet cache via IB Gateway, then rebuild
    the combined price matrix.

    Uses ``IBDataCollector``'s smart classification — CURRENT tickers
    (gap <= stale_days) are skipped entirely, STALE tickers get an
    incremental update, MISSING tickers get a full ``duration``
    download. So a run against an already-up-to-date cache costs
    seconds; a run against a stale cache costs minutes; the first-time
    full download costs hours (rate-limit-bound).

    Args:
        ib_host / ib_port / ib_client_id: Gateway connection details.
            ``ib_client_id`` defaults to :data:`REFRESH_IB_CLIENT_ID`
            so it will not collide with the read-only snapshot connection.
        cache_dir: Where per-ticker parquets live.
        processed_dir: Where the combined ``etf_prices_ib.parquet`` is written.
        duration / rate_limit_interval: Passed to ``IBDataCollector``.
        progress_callback: Optional ``(i, total, ticker) -> None`` — the
            applet uses this to drive a Streamlit progress bar.

    Returns:
        :class:`RefreshResult` summarising the run and giving the path
        to the rebuilt price matrix (which :func:`collect_prices` will
        then re-read).

    Raises:
        RuntimeError: If IB Gateway cannot be connected (bubbles up
            from :func:`src.portfolio.ib_state.connect_read_only`).
    """
    from src.data_collection.comprehensive_etf_list import load_full_universe
    from src.data_collection.ib_data_collector import IBDataCollector
    from src.portfolio.ib_state import connect_read_only

    cache_dir = Path(cache_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    tickers, _ = load_full_universe()
    logger.info("Refreshing %d tickers via IB (client_id=%d)", len(tickers), ib_client_id)

    ib = connect_read_only(host=ib_host, port=ib_port, client_id=ib_client_id)
    try:
        collector = IBDataCollector(
            ib=ib,
            cache_dir=str(cache_dir),
            rate_limit_interval=rate_limit_interval,
            duration=duration,
        )

        pre_status = cache_status(cache_dir=cache_dir)

        # Adapter — collector's callback signature is (i, total, ticker, ok);
        # the applet only cares about (i, total, ticker).
        def _adapter(i: int, total: int, ticker: str, ok: bool) -> None:  # noqa: FBT001
            if progress_callback is not None:
                progress_callback(i, total, ticker)

        prices, _ = collector.collect_universe(tickers, progress_callback=_adapter)
    finally:
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            logger.warning("ib.disconnect() raised — ignored during refresh cleanup")

    out_path = processed_dir / "etf_prices_ib.parquet"
    if prices is not None and not prices.empty:
        prices.to_parquet(out_path)
        logger.info("Saved refreshed price matrix: %s (%s)", out_path, prices.shape)

    return RefreshResult(
        n_current=pre_status["n_current"],
        n_stale=pre_status["n_stale"],
        n_missing=pre_status["n_missing"],
        n_universe=pre_status["n_universe"],
        parquet_path=out_path,
    )


def _resolve_cached_prices(processed_dir: Path) -> Path:
    for filename in _PROCESSED_FILENAMES:
        candidate = processed_dir / filename
        if candidate.exists():
            logger.info("Loading prices from %s", candidate.name)
            return candidate
    tried = ", ".join(_PROCESSED_FILENAMES)
    raise FileNotFoundError(
        f"No cached ETF price parquet under {processed_dir!s} "
        f"(tried: {tried}). Run scripts/daily_etf_data.py or the daily cron."
    )


def _apply_quality_filter(
    prices: pd.DataFrame,
    min_days: int,
    max_missing_pct: float,
) -> pd.DataFrame:
    """Keep tickers with enough history and low missing-value rate.

    Forward-fills then back-fills small gaps. Mirrors the historic
    behaviour of ``s2_collect.apply_quality_filter`` but reads its
    thresholds from arguments rather than magic constants.
    """
    if prices.empty:
        return prices
    missing_pct = prices.isnull().sum() / len(prices) * 100
    keep = (prices.count() >= min_days) & (missing_pct < max_missing_pct)
    filtered = prices.loc[:, keep].ffill().bfill()
    logger.info(
        "Quality filter: %d -> %d tickers (min_days=%d, max_missing=%.1f%%)",
        prices.shape[1], filtered.shape[1], min_days, max_missing_pct,
    )
    return filtered


# ────────────────────────────────────────────────────────────────
# 2. Score
# ────────────────────────────────────────────────────────────────

def score_factors(
    prices: pd.DataFrame,
    policy: SmartBetaPolicy,
    expense_ratios: Optional[pd.Series] = None,
) -> ScoringResult:
    """Compute per-factor scores + weighted-geometric-mean integration.

    Applies the standard smart-beta filter (drop leveraged / inverse
    ETFs) before scoring so a Direxion 3× fund never dominates the
    momentum quintile.

    If no expense-ratio series is provided, the value factor is
    silently dropped and its weight redistributed proportionally to
    the remaining factors (matches historic s3_factors.py behaviour).

    Args:
        prices: Wide DataFrame (dates × tickers).
        policy: Active policy (factor weights + per-factor lookbacks).
        expense_ratios: Optional Series (ticker → decimal expense
            ratio, e.g. 0.0045 for 45 bps). Missing tickers are
            imputed to the median before the value factor runs.

    Returns:
        :class:`ScoringResult` with combined scores, per-factor
        breakdown, the *actually used* weights, and the surviving
        universe.
    """
    all_tickers = prices.columns.tolist()
    basic_tickers = filter_leveraged_etfs(all_tickers)
    excluded = len(all_tickers) - len(basic_tickers)
    logger.info(
        "Universe: %d -> %d after leveraged/inverse exclusion (%d dropped)",
        len(all_tickers), len(basic_tickers), excluded,
    )
    prices_basic = prices[basic_tickers]

    lookbacks = policy.factor_lookbacks
    momentum = MomentumFactor(
        lookback=lookbacks.momentum,
        skip_recent=lookbacks.momentum_skip_recent,
    ).calculate(prices_basic)
    quality = QualityFactor(lookback=lookbacks.quality).calculate(prices_basic)
    volatility = VolatilityFactor(lookback=lookbacks.volatility).calculate(prices_basic)

    weights = dict(policy.factor_weights.as_dict())
    factor_dict: dict[str, pd.Series] = {
        "momentum": momentum,
        "quality": quality,
        "volatility": volatility,
    }

    if expense_ratios is not None and expense_ratios.notna().sum() > 0:
        median_er = expense_ratios.dropna().median()
        aligned_er = expense_ratios.reindex(prices_basic.columns).fillna(median_er)
        factor_dict["value"] = SimplifiedValueFactor().calculate(prices_basic, aligned_er)
    else:
        logger.info(
            "No expense ratios provided — value factor skipped, "
            "weight redistributed proportionally to remaining factors."
        )
        weights.pop("value", None)
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    factor_df = pd.DataFrame(factor_dict)
    combined = FactorIntegrator(factor_weights=weights).integrate(factor_df)

    return ScoringResult(
        combined_scores=combined,
        factor_scores=factor_df,
        active_weights=weights,
        universe=tuple(prices_basic.columns.tolist()),
    )


# ────────────────────────────────────────────────────────────────
# 3. Optimise
# ────────────────────────────────────────────────────────────────

_OPTIMIZERS = {"mvo", "rankbased", "minvar", "simple"}


def optimize_portfolio(
    scoring: ScoringResult,
    prices: pd.DataFrame,
    policy: SmartBetaPolicy,
    optimizer_type: str = "rankbased",
) -> pd.Series:
    """Run the configured optimizer over the integrated scores.

    Weight bounds and target position count come from the policy.
    ``optimizer_type`` selects between the four available implementations
    in :mod:`src.portfolio.optimizer` — default ``rankbased`` matches the
    historic script default, but ``mvo`` matches the current tech-doc
    canonical config. This selection is intentionally left as an argument
    (not a policy field) until the calibration diagnostic in
    ``docs/RESEARCH_BACKLOG.md`` #1 resolves the tech-doc/policy drift.

    Args:
        scoring: Output of :func:`score_factors`.
        prices: Same wide price frame — some optimizers need it for
            covariance / return estimation.
        policy: Active policy (num_positions, min/max weight).
        optimizer_type: One of ``mvo``, ``rankbased``, ``minvar``, ``simple``.

    Returns:
        Series of ticker → target weight. Sums to 1.0.
    """
    if optimizer_type not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer_type {optimizer_type!r} — expected one of {sorted(_OPTIMIZERS)}"
        )

    scores = scoring.combined_scores.reindex(list(scoring.universe)).dropna()
    n = policy.num_positions

    if optimizer_type == "mvo":
        return MeanVarianceOptimizer(
            num_positions=n,
            lookback=policy.factor_lookbacks.volatility,
            risk_aversion=policy.risk_aversion,
            use_factor_scores_as_alpha=True,
            min_weight=policy.min_weight,
            max_weight=policy.max_weight,
        ).optimize(scores, prices[list(scoring.universe)])

    if optimizer_type == "rankbased":
        return RankBasedOptimizer(
            num_positions=n,
            weighting_scheme="exponential",
        ).optimize(scores)

    if optimizer_type == "minvar":
        return MinVarianceOptimizer(
            num_positions=n,
            lookback=policy.factor_lookbacks.volatility,
        ).optimize(scores, prices[list(scoring.universe)])

    return SimpleOptimizer(num_positions=n).optimize(scores)


# ────────────────────────────────────────────────────────────────
# 4. Portfolio-level diagnostics — used by the applet + trade sizer
# ────────────────────────────────────────────────────────────────

def portfolio_volatility(
    target_weights: pd.Series,
    prices: pd.DataFrame,
    trading_days: int = 252,
) -> float:
    """Ex-ante annualised volatility of the target portfolio.

    Returns ``float('nan')`` when insufficient price history is available.
    """
    tickers = [t for t in target_weights.index if t in prices.columns]
    if len(tickers) < len(target_weights) or len(prices) < 60:
        return float("nan")
    rets = prices[tickers].pct_change().dropna()
    if rets.empty:
        return float("nan")
    cov = rets.cov() * trading_days
    w = target_weights[tickers].values
    return float(np.sqrt(w @ cov.values @ w))


def portfolio_hhi(target_weights: pd.Series) -> float:
    """Herfindahl-Hirschman Index — concentration measure in [1/n, 1]."""
    return float((target_weights ** 2).sum())
