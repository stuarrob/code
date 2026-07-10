"""Unit tests for src.portfolio.pipeline.

The pipeline is the deterministic core the applet drives — these tests
pin the shape and invariants of every stage on a small synthetic universe
so the applet cannot silently drift.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.portfolio.pipeline import (
    PriceLoad,
    RefreshResult,
    ScoringResult,
    cache_status,
    collect_prices,
    optimize_portfolio,
    portfolio_hhi,
    portfolio_volatility,
    refresh_prices_from_ib,
    score_factors,
)
from src.portfolio.policy import load_policy


@pytest.fixture
def policy():
    return load_policy()


@pytest.fixture
def synthetic_prices():
    """500 days x 60 tickers of correlated random-walk prices."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=500)
    n = 60
    tickers = [f"ETF{i:03d}" for i in range(n)]
    drift = np.random.uniform(0.0001, 0.0006, n)
    vol = np.random.uniform(0.008, 0.020, n)
    returns = np.random.randn(500, n) * vol + drift
    prices = pd.DataFrame(
        100 * (1 + returns).cumprod(axis=0),
        index=dates,
        columns=tickers,
    )
    return prices


# ────────────────────────────────────────────────────────────────
# collect_prices
# ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_collect_prices_reads_ib_parquet_when_present(tmp_path, policy, synthetic_prices):
    synthetic_prices.to_parquet(tmp_path / "etf_prices_ib.parquet")
    load = collect_prices(policy, processed_dir=tmp_path)
    assert isinstance(load, PriceLoad)
    assert load.source == "etf_prices_ib.parquet"
    assert load.n_tickers == synthetic_prices.shape[1]
    assert load.start_date <= load.end_date


@pytest.mark.unit
def test_collect_prices_priority_db_over_ib(tmp_path, policy, synthetic_prices):
    """Databento cache takes precedence over IB cache when both exist."""
    synthetic_prices.to_parquet(tmp_path / "etf_prices_db.parquet")
    synthetic_prices.to_parquet(tmp_path / "etf_prices_ib.parquet")
    load = collect_prices(policy, processed_dir=tmp_path)
    assert load.source == "etf_prices_db.parquet"


@pytest.mark.unit
def test_collect_prices_missing_directory_raises(tmp_path, policy):
    with pytest.raises(FileNotFoundError, match="cached ETF price parquet"):
        collect_prices(policy, processed_dir=tmp_path / "empty")


@pytest.mark.unit
def test_collect_prices_rejects_tiny_universe(tmp_path, policy):
    """Fewer than 20 usable tickers means we cannot score — raise clearly."""
    tiny = pd.DataFrame(
        np.random.rand(300, 10) + 100,
        index=pd.bdate_range("2024-01-01", periods=300),
        columns=[f"T{i}" for i in range(10)],
    )
    tiny.to_parquet(tmp_path / "etf_prices_ib.parquet")
    with pytest.raises(ValueError, match="too small to score"):
        collect_prices(policy, processed_dir=tmp_path)


# ────────────────────────────────────────────────────────────────
# score_factors
# ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_score_factors_returns_all_active_factors(policy, synthetic_prices):
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    # No expense ratios provided → value dropped, weights redistribute.
    assert set(scoring.active_weights.keys()) == {"momentum", "quality", "volatility"}
    assert abs(sum(scoring.active_weights.values()) - 1.0) < 1e-6
    assert set(scoring.factor_scores.columns) == {"momentum", "quality", "volatility"}


@pytest.mark.unit
def test_score_factors_includes_value_when_expense_ratios_provided(policy, synthetic_prices):
    expense = pd.Series(
        np.random.uniform(0.0005, 0.007, len(synthetic_prices.columns)),
        index=synthetic_prices.columns,
    )
    scoring = score_factors(synthetic_prices, policy, expense_ratios=expense, use_curated=False)
    assert "value" in scoring.active_weights
    assert "value" in scoring.factor_scores.columns
    # With value present, weights match the policy exactly.
    assert scoring.active_weights == policy.factor_weights.as_dict()


@pytest.mark.unit
def test_score_factors_universe_excludes_none(policy, synthetic_prices):
    """A synthetic universe with no leveraged tickers should survive intact."""
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    assert len(scoring.universe) == synthetic_prices.shape[1]


@pytest.mark.unit
def test_score_factors_missing_expense_ratios_imputed(policy, synthetic_prices):
    """Partial expense ratios (some NaN) should not drop the value factor —
    NaN entries fall back to the median."""
    expense = pd.Series(
        np.random.uniform(0.0005, 0.007, len(synthetic_prices.columns)),
        index=synthetic_prices.columns,
    )
    expense.iloc[:20] = np.nan  # 20 unknowns
    scoring = score_factors(synthetic_prices, policy, expense_ratios=expense, use_curated=False)
    assert "value" in scoring.active_weights


# ────────────────────────────────────────────────────────────────
# optimize_portfolio
# ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_optimize_rankbased_returns_target_positions(policy, synthetic_prices):
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    weights = optimize_portfolio(scoring, synthetic_prices, policy, optimizer_type="rankbased")
    assert len(weights) == policy.num_positions
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()


@pytest.mark.unit
def test_optimize_rejects_unknown_optimizer(policy, synthetic_prices):
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        optimize_portfolio(scoring, synthetic_prices, policy, optimizer_type="bogus")


@pytest.mark.unit
def test_optimize_simple_returns_target_positions(policy, synthetic_prices):
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    weights = optimize_portfolio(scoring, synthetic_prices, policy, optimizer_type="simple")
    assert len(weights) == policy.num_positions
    assert abs(weights.sum() - 1.0) < 1e-6


# ────────────────────────────────────────────────────────────────
# portfolio_volatility / portfolio_hhi
# ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_portfolio_volatility_positive_finite(policy, synthetic_prices):
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    weights = optimize_portfolio(scoring, synthetic_prices, policy)
    vol = portfolio_volatility(weights, synthetic_prices)
    assert vol == vol  # not NaN
    assert 0.0 < vol < 1.0


@pytest.mark.unit
def test_portfolio_hhi_bounds(policy, synthetic_prices):
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    weights = optimize_portfolio(scoring, synthetic_prices, policy)
    hhi = portfolio_hhi(weights)
    equal_weight_hhi = 1.0 / policy.num_positions
    assert equal_weight_hhi <= hhi <= 1.0


@pytest.mark.unit
def test_portfolio_volatility_nan_when_ticker_missing(policy, synthetic_prices):
    """If one target ticker has no price series, vol is NaN, not a lie."""
    scoring = score_factors(synthetic_prices, policy, use_curated=False)
    weights = optimize_portfolio(scoring, synthetic_prices, policy)
    # Rename one column in prices so the ticker is not found.
    prices_short = synthetic_prices.rename(columns={weights.index[0]: "ZZZ_MISSING"})
    vol = portfolio_volatility(weights, prices_short)
    assert vol != vol  # NaN


# ────────────────────────────────────────────────────────────────
# cache_status
# ────────────────────────────────────────────────────────────────

def _write_ticker_cache(cache_dir: Path, ticker: str, last_date: pd.Timestamp) -> None:
    """Create a per-ticker parquet whose latest bar is ``last_date``."""
    dates = pd.bdate_range(end=last_date, periods=30)
    df = pd.DataFrame({"close": np.linspace(100, 110, 30)}, index=dates)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{ticker}.parquet")


@pytest.mark.unit
def test_cache_status_classifies_current_stale_missing(tmp_path):
    """cache_status buckets tickers correctly against a stale_days=1 rule."""
    today = pd.Timestamp.now().normalize()
    _write_ticker_cache(tmp_path, "SPY", today)           # current
    _write_ticker_cache(tmp_path, "QQQ", today - pd.Timedelta(days=10))  # stale
    # ZZZ_MISS deliberately not written -> missing

    with patch("src.data_collection.comprehensive_etf_list.load_full_universe") as loader:
        loader.return_value = (["SPY", "QQQ", "ZZZ_MISS"], {})
        status = cache_status(cache_dir=tmp_path)

    assert status["n_universe"] == 3
    assert status["n_current"] == 1
    assert status["n_stale"] == 1
    assert status["n_missing"] == 1
    assert status["latest_bar"].normalize() == today


@pytest.mark.unit
def test_cache_status_empty_dir_all_missing(tmp_path):
    with patch("src.data_collection.comprehensive_etf_list.load_full_universe") as loader:
        loader.return_value = (["AAA", "BBB", "CCC"], {})
        status = cache_status(cache_dir=tmp_path)
    assert status["n_missing"] == 3
    assert status["n_current"] == 0
    assert status["latest_bar"] is None


# ────────────────────────────────────────────────────────────────
# refresh_prices_from_ib
# ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_refresh_prices_from_ib_writes_matrix_and_disconnects(tmp_path):
    """The refresh wraps IBDataCollector: connects read-only, runs the
    collector, writes the combined matrix, and always disconnects."""
    fake_prices = pd.DataFrame(
        {"SPY": [400.0, 401.0, 402.0], "QQQ": [350.0, 351.0, 352.0]},
        index=pd.bdate_range("2026-06-01", periods=3),
    )
    fake_ib = MagicMock()
    fake_collector = MagicMock()
    fake_collector.collect_universe.return_value = (fake_prices, pd.DataFrame())

    processed_dir = tmp_path / "processed"
    cache_dir = tmp_path / "ib_historical"

    with patch("src.portfolio.ib_state.connect_read_only", return_value=fake_ib) as connect, \
         patch("src.data_collection.ib_data_collector.IBDataCollector",
               return_value=fake_collector), \
         patch("src.data_collection.comprehensive_etf_list.load_full_universe",
               return_value=(["SPY", "QQQ"], {})):
        result = refresh_prices_from_ib(
            ib_host="127.0.0.1", ib_port=4001, ib_client_id=31,
            cache_dir=cache_dir, processed_dir=processed_dir,
        )

    connect.assert_called_once()
    fake_collector.collect_universe.assert_called_once()
    fake_ib.disconnect.assert_called_once()
    assert isinstance(result, RefreshResult)
    assert result.parquet_path == processed_dir / "etf_prices_ib.parquet"
    assert result.parquet_path.exists()
    saved = pd.read_parquet(result.parquet_path)
    assert list(saved.columns) == ["SPY", "QQQ"]


@pytest.mark.unit
def test_refresh_prices_from_ib_disconnects_on_collector_failure(tmp_path):
    """Even if collect_universe raises, we must not leak the IB connection."""
    fake_ib = MagicMock()
    fake_collector = MagicMock()
    fake_collector.collect_universe.side_effect = RuntimeError("IB pacing violation")

    with patch("src.portfolio.ib_state.connect_read_only", return_value=fake_ib), \
         patch("src.data_collection.ib_data_collector.IBDataCollector",
               return_value=fake_collector), \
         patch("src.data_collection.comprehensive_etf_list.load_full_universe",
               return_value=(["SPY"], {})):
        with pytest.raises(RuntimeError, match="pacing violation"):
            refresh_prices_from_ib(processed_dir=tmp_path)

    fake_ib.disconnect.assert_called_once()


@pytest.mark.unit
def test_refresh_prices_from_ib_forwards_progress_callback(tmp_path):
    """The applet passes a progress callback expecting (i, total, ticker) —
    the collector calls back with (i, total, ticker, ok); the adapter must
    strip the ok flag."""
    fake_ib = MagicMock()
    fake_collector = MagicMock()
    fake_collector.collect_universe.return_value = (pd.DataFrame(), pd.DataFrame())

    seen = []

    def _cb(i, total, ticker):
        seen.append((i, total, ticker))

    def _fake_collect(tickers, progress_callback=None, **kwargs):
        # Simulate the collector calling its 4-arg callback
        if progress_callback:
            progress_callback(1, 2, "SPY", True)
            progress_callback(2, 2, "QQQ", False)
        return pd.DataFrame(), pd.DataFrame()

    fake_collector.collect_universe.side_effect = _fake_collect

    with patch("src.portfolio.ib_state.connect_read_only", return_value=fake_ib), \
         patch("src.data_collection.ib_data_collector.IBDataCollector",
               return_value=fake_collector), \
         patch("src.data_collection.comprehensive_etf_list.load_full_universe",
               return_value=(["SPY", "QQQ"], {})):
        refresh_prices_from_ib(processed_dir=tmp_path, progress_callback=_cb)

    assert seen == [(1, 2, "SPY"), (2, 2, "QQQ")]
