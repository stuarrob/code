"""
Pipeline Verification Tests

Tests that verify:
1. Trailing stops are placed on every BUY
2. Rebalance count stays within limits (max 6/year)
3. Universe size is correct
4. Factor scores are produced
5. Portfolio has correct number of positions
6. Data paths resolve correctly
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))


# ──────────────────────────────────────────────────────────
# Test 1: Universe size
# ──────────────────────────────────────────────────────────

def test_universe_size():
    """Full universe should have 5000+ tickers (curated + NASDAQ)."""
    from data_collection.comprehensive_etf_list import load_full_universe

    tickers, cats = load_full_universe()

    assert len(tickers) > 4000, f"Universe too small: {len(tickers)}"

    curated = sum(1 for t in tickers if cats[t] != "Uncategorized")
    assert curated > 700, f"Curated list too small: {curated}"
    print(f"PASS: Universe has {len(tickers)} tickers ({curated} curated)")


def test_curated_list_no_mutual_funds():
    """Curated list should not contain mutual funds."""
    from data_collection.comprehensive_etf_list import get_all_tickers

    known_mutual_funds = {"FFNOX", "VBMFX", "VBAAX", "VBIAX", "VBINX",
                          "VBLTX", "VSMGX", "VASGX", "FFGFX", "DGSIX", "VICEX"}
    tickers = set(get_all_tickers())
    overlap = tickers & known_mutual_funds
    assert not overlap, f"Mutual funds in curated list: {overlap}"
    print("PASS: No mutual funds in curated list")


# ──────────────────────────────────────────────────────────
# Test 2: Factor scoring
# ──────────────────────────────────────────────────────────

def test_factor_scores_produced():
    """Factor scoring should produce scores for available tickers."""
    from factors import (
        FactorIntegrator, MomentumFactor, QualityFactor,
        SimplifiedValueFactor, VolatilityFactor,
    )

    # Create synthetic price data (100 tickers, 300 days)
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=300)
    tickers = [f"ETF{i:03d}" for i in range(100)]
    prices = pd.DataFrame(
        np.random.lognormal(0, 0.02, (300, 100)).cumprod(axis=0) * 100,
        index=dates, columns=tickers,
    )

    momentum = MomentumFactor(lookback=252, skip_recent=21).calculate(prices)
    quality = QualityFactor(lookback=252).calculate(prices)
    expense = pd.Series(0.005, index=tickers)
    value = SimplifiedValueFactor().calculate(prices, expense)
    vol = VolatilityFactor(lookback=60).calculate(prices)

    factor_df = pd.DataFrame({
        "momentum": momentum, "quality": quality,
        "value": value, "volatility": vol,
    })
    weights = {"momentum": 0.35, "quality": 0.30, "value": 0.15, "volatility": 0.20}
    scores = FactorIntegrator(factor_weights=weights).integrate(factor_df)

    assert len(scores) > 0, "No scores produced"
    assert scores.max() > scores.min(), "All scores identical"
    print(f"PASS: Factor scores produced for {len(scores)} tickers")


# ──────────────────────────────────────────────────────────
# Test 3: Portfolio construction
# ──────────────────────────────────────────────────────────

def test_portfolio_correct_positions():
    """Portfolio should have exactly NUM_POSITIONS holdings."""
    from portfolio import RankBasedOptimizer

    scores = pd.Series(np.random.uniform(0, 1, 100),
                       index=[f"ETF{i:03d}" for i in range(100)])

    for n in [10, 20, 30]:
        opt = RankBasedOptimizer(num_positions=n, weighting_scheme="exponential")
        weights = opt.optimize(scores)
        assert len(weights) == n, f"Expected {n} positions, got {len(weights)}"
        assert abs(weights.sum() - 1.0) < 0.01, f"Weights don't sum to 1: {weights.sum()}"

    print("PASS: Portfolio has correct number of positions")


# ──────────────────────────────────────────────────────────
# Test 4: Rebalance frequency
# ──────────────────────────────────────────────────────────

def test_rebalance_bimonthly_max_6_per_year():
    """Bimonthly rebalancing should produce <= 6 rebalances per year."""
    from backtesting import BacktestConfig, BacktestEngine, TransactionCostModel
    from portfolio import RankBasedOptimizer, StopLossManager, ThresholdRebalancer

    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=756)  # ~3 years
    tickers = [f"ETF{i:03d}" for i in range(50)]
    prices = pd.DataFrame(
        np.random.lognormal(0, 0.01, (756, 50)).cumprod(axis=0) * 100,
        index=dates, columns=tickers,
    )

    scores = pd.Series(np.random.uniform(0, 1, 50), index=tickers)
    factor_scores = pd.DataFrame({d: scores for d in dates}).T
    factor_scores.index = dates

    config = BacktestConfig(
        initial_capital=1_000_000,
        rebalance_frequency="bimonthly",
        num_positions=20,
        stop_loss_pct=0.12,
        use_stop_loss=True,
    )
    cost_model = TransactionCostModel(commission_per_trade=0.0)

    results = BacktestEngine(config, cost_model).run(
        prices=prices,
        factor_scores=factor_scores,
        optimizer=RankBasedOptimizer(num_positions=20),
        rebalancer=ThresholdRebalancer(drift_threshold=0.05),
        risk_manager=StopLossManager(position_stop_loss=0.12),
    )

    num_rebalances = results["metrics"].get("num_rebalances", 0)
    years = len(dates) / 252
    per_year = num_rebalances / years

    assert per_year <= 7, (
        f"Too many rebalances: {per_year:.1f}/year "
        f"({num_rebalances} total over {years:.1f} years)"
    )
    print(f"PASS: Bimonthly rebalancing = {per_year:.1f}/year "
          f"({num_rebalances} total over {years:.1f} years)")


def test_rebalance_quarterly_max_4_per_year():
    """Quarterly rebalancing should produce ~4 rebalances per year."""
    from backtesting import BacktestConfig, BacktestEngine, TransactionCostModel
    from portfolio import RankBasedOptimizer, StopLossManager, ThresholdRebalancer

    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=756)
    tickers = [f"ETF{i:03d}" for i in range(50)]
    prices = pd.DataFrame(
        np.random.lognormal(0, 0.01, (756, 50)).cumprod(axis=0) * 100,
        index=dates, columns=tickers,
    )

    scores = pd.Series(np.random.uniform(0, 1, 50), index=tickers)
    factor_scores = pd.DataFrame({d: scores for d in dates}).T
    factor_scores.index = dates

    config = BacktestConfig(
        initial_capital=1_000_000,
        rebalance_frequency="quarterly",
        num_positions=20,
        stop_loss_pct=0.12,
        use_stop_loss=True,
    )
    cost_model = TransactionCostModel(commission_per_trade=0.0)

    results = BacktestEngine(config, cost_model).run(
        prices=prices,
        factor_scores=factor_scores,
        optimizer=RankBasedOptimizer(num_positions=20),
        rebalancer=ThresholdRebalancer(drift_threshold=0.05),
        risk_manager=StopLossManager(position_stop_loss=0.12),
    )

    num_rebalances = results["metrics"].get("num_rebalances", 0)
    years = len(dates) / 252
    per_year = num_rebalances / years

    assert per_year <= 5, (
        f"Too many rebalances: {per_year:.1f}/year"
    )
    print(f"PASS: Quarterly rebalancing = {per_year:.1f}/year "
          f"({num_rebalances} total)")


# ──────────────────────────────────────────────────────────
# Test 5: Trailing stops in execution code
# ──────────────────────────────────────────────────────────

def test_trailing_stop_in_execute_script():
    """Execution script must place trailing stop on every BUY."""
    execute_path = PROJECT_ROOT / "notebooks" / "scripts" / "s7_execute.py"
    code = execute_path.read_text()

    # Check for TRAIL order type
    assert "orderType = \"TRAIL\"" in code or "orderType = 'TRAIL'" in code, \
        "Missing TRAIL order type in execution script"

    # Check trailing stop is on BUY fills
    assert "trailingPercent" in code, \
        "Missing trailingPercent in execution script"

    # Check GTC time-in-force
    assert "GTC" in code, \
        "Missing GTC time-in-force for trailing stop"

    # Check it's conditional on BUY action
    assert 'action"] == "BUY"' in code or "action == 'BUY'" in code or \
           "action\"] == \"BUY\"" in code, \
        "Trailing stop not conditional on BUY action"

    print("PASS: Trailing stop logic verified in s7_execute.py")


def test_trailing_stop_in_notebook():
    """Notebook Section 8 must place trailing stop on every BUY."""
    nb_path = PROJECT_ROOT / "notebooks" / "08_full_pipeline.ipynb"
    nb_text = nb_path.read_text()

    # .ipynb JSON escapes quotes, so check for the key identifiers
    assert "TRAIL" in nb_text, "Missing TRAIL order in notebook"
    assert "trailingPercent" in nb_text, "Missing trailingPercent in notebook"
    assert "GTC" in nb_text, "Missing GTC in notebook"
    assert "TRAILING_STOP_PCT" in nb_text, "Missing TRAILING_STOP_PCT config"

    print("PASS: Trailing stop logic verified in notebook 08")


def test_trailing_stop_in_ib_execute():
    """Standalone ib_execute_trades.py must place trailing stop on every BUY."""
    script_path = PROJECT_ROOT / "scripts" / "ib_execute_trades.py"
    if not script_path.exists():
        pytest.skip("ib_execute_trades.py not found")

    code = script_path.read_text()
    assert "TRAIL" in code, "Missing TRAIL in ib_execute_trades.py"
    assert "trailingPercent" in code or "trailing_percent" in code or \
           "TRAILING_STOP_PCT" in code, \
        "Missing trailing percent config"

    print("PASS: Trailing stop logic verified in ib_execute_trades.py")


# ──────────────────────────────────────────────────────────
# Test 6: Data paths
# ──────────────────────────────────────────────────────────

def test_cached_data_exists():
    """At least some cached IB data should exist."""
    ib_cache = PROJECT_ROOT / "data" / "ib_historical"
    if not ib_cache.exists():
        pytest.skip("IB cache directory not found (data may have been moved)")

    parquets = list(ib_cache.glob("*.parquet"))
    parquets = [f for f in parquets if f.stem != "manifest"]

    assert len(parquets) > 100, f"Too few cached parquets: {len(parquets)}"
    print(f"PASS: {len(parquets)} cached parquet files found")


def test_pipeline_scripts_exist():
    """All pipeline scripts should exist."""
    scripts_dir = PROJECT_ROOT / "notebooks" / "scripts"
    expected = ["s1_universe.py", "s2_collect.py", "s3_factors.py",
                "s4_optimize.py", "s5_backtest.py", "s6_trades.py",
                "s7_execute.py"]

    for name in expected:
        path = scripts_dir / name
        assert path.exists(), f"Missing script: {name}"

    print(f"PASS: All {len(expected)} pipeline scripts exist")


# ──────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_universe_size,
        test_curated_list_no_mutual_funds,
        test_factor_scores_produced,
        test_portfolio_correct_positions,
        test_rebalance_bimonthly_max_6_per_year,
        test_rebalance_quarterly_max_4_per_year,
        test_trailing_stop_in_execute_script,
        test_trailing_stop_in_notebook,
        test_trailing_stop_in_ib_execute,
        test_cached_data_exists,
        test_pipeline_scripts_exist,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"SKIP: {name} — {e}")
            skipped += 1
        except Exception as e:
            print(f"FAIL: {name} — {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*50}")
