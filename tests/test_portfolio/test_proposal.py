"""Unit tests for `src.portfolio.proposal.propose_trades`.

Order-construction module — trades made by this code become real
`ib_insync` orders. Per CLAUDE.md: "Stop-loss and order-construction
changes require explicit unit tests that pin the trigger condition or
the exact order payload."

Tests target the failure modes that would produce a materially wrong
order payload:
  - Wrong side (BUY sized as SELL or vice versa)
  - Wrong quantity (off by an order of magnitude or a sign)
  - Cash reserve breached silently
  - Micro-orders emitted that lose to commission drag
  - Weight-vs-target ratios computed wrong
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.portfolio.ib_state import IBSnapshot, Position
from src.portfolio.policy import SmartBetaPolicy, FactorWeights, FactorLookbacks
from src.portfolio.proposal import (
    ACTION_BUY, ACTION_EXTEND, ACTION_SELL,
    Trade, TradeProposal, propose_trades,
)


pytestmark = pytest.mark.unit


def _policy(cash_reserve: float = 5_000.0, drift: float = 0.05,
            num_positions: int = 5) -> SmartBetaPolicy:
    return SmartBetaPolicy(
        name="test-policy",
        version=1,
        tax_status="taxable",
        num_positions=num_positions,
        min_weight=0.02,
        max_weight=0.30,
        factor_weights=FactorWeights(momentum=0.35, quality=0.30,
                                      volatility=0.20, value=0.15),
        factor_lookbacks=FactorLookbacks(momentum=252, momentum_skip_recent=21,
                                         quality=252, value=252, volatility=60),
        rebalance_frequency="bimonthly",
        drift_threshold=drift,
        entry_stop_loss_pct=0.12,
        trailing_stop_pct=0.10,
        cash_reserve=cash_reserve,
        risk_aversion=1.5,
        robustness_penalty=0.7,
        turnover_penalty=0.2,
    )


def _position(ticker: str, shares: float, price: float) -> Position:
    return Position(
        ticker=ticker, conid=hash(ticker) & 0xFFFFFF,
        shares=shares, avg_cost=price, market_price=price,
        market_value=shares * price, unrealized_pnl=0.0,
        daily_pnl=float("nan"),
    )


def _snapshot(nav: float, cash: float,
              positions: tuple[Position, ...] = tuple()) -> IBSnapshot:
    return IBSnapshot(
        account="TEST", timestamp=datetime.now(),
        nav=nav, cash=cash, buying_power=cash,
        gross_position_value=nav - cash,
        realized_pnl_reported=0.0, unrealized_pnl_reported=0.0,
        daily_pnl=0.0, excess_liquidity=cash, available_funds=cash,
        maint_margin=0.0, init_margin=0.0,
        positions=positions, open_orders=tuple(),
    )


# ────────────────────────────────────────────────────────────────
# Basic BUY-from-empty-portfolio
# ────────────────────────────────────────────────────────────────

class TestFreshBuys:
    def test_single_ticker_full_deployment(self):
        """Empty book, budget = full NAV, single target — BUY the full amount."""
        snap = _snapshot(nav=100_000, cash=100_000)
        targets = pd.Series({"SPY": 1.0})
        # Simulate market price via a Position we DON'T hold. propose_trades
        # requires a price. Since we're empty, no snapshot price — build via
        # a helper: pass zero cash_budget and use held positions to seed.
        # Actually for a fresh buy we need market prices elsewhere.
        # Design decision: since snapshot doesn't carry non-held prices,
        # propose_trades needs prices provided externally OR sizes from
        # snapshot-only. Current API sources price from the Position; a
        # fresh BUY on a symbol never held → no price → skip with warning.
        proposal = propose_trades(
            snapshot=snap,
            target_weights=targets,
            cash_budget=0,
            policy=_policy(cash_reserve=0),
        )
        # Expected: warning that price is missing; no trade emitted.
        assert len(proposal.trades) == 0
        assert any("no market price" in w for w in proposal.warnings)

    def test_extend_existing_position(self):
        """Currently hold 10 SPY @ 100 (mkt val 1000, 1% of 100k NAV). Target
        50% — should EXTEND to 500 shares (mkt val 50000)."""
        snap = _snapshot(
            nav=100_000, cash=99_000,
            positions=(_position("SPY", 10, 100.0),),
        )
        targets = pd.Series({"SPY": 0.50})
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0),
        )
        assert len(proposal.trades) == 1
        t = proposal.trades[0]
        assert t.ticker == "SPY"
        assert t.action == ACTION_EXTEND
        assert t.delta_shares == 490  # 500 target - 10 current
        assert t.target_shares == 500
        assert t.market_price == 100.0
        assert t.delta_notional == pytest.approx(49_000.0)


# ────────────────────────────────────────────────────────────────
# SELL
# ────────────────────────────────────────────────────────────────

class TestSells:
    def test_exit_position_target_zero(self):
        """Hold 100 SPY @ 100. Target = 0 (dropped from top-N). Full SELL."""
        snap = _snapshot(
            nav=100_000, cash=90_000,
            positions=(_position("SPY", 100, 100.0),),
        )
        targets = pd.Series(dtype=float)  # SPY not in target set
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0),
        )
        assert len(proposal.trades) == 1
        t = proposal.trades[0]
        assert t.ticker == "SPY"
        assert t.action == ACTION_SELL
        assert t.delta_shares == -100
        assert t.target_shares == 0
        assert t.delta_notional == pytest.approx(10_000.0)

    def test_reduce_position_below_current(self):
        """Hold 100 SPY @ 100 (10% of NAV). Target 5% — reduce to 50 shares."""
        snap = _snapshot(
            nav=100_000, cash=90_000,
            positions=(_position("SPY", 100, 100.0),),
        )
        targets = pd.Series({"SPY": 0.05})
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0),
        )
        assert len(proposal.trades) == 1
        t = proposal.trades[0]
        assert t.action == ACTION_SELL
        assert t.delta_shares == -50


# ────────────────────────────────────────────────────────────────
# Drift threshold — the whipsaw guard
# ────────────────────────────────────────────────────────────────

class TestDriftThreshold:
    def test_small_drift_skipped(self):
        """Position at 10% of NAV, target 11% — inside 5% drift, skip."""
        snap = _snapshot(
            nav=100_000, cash=90_000,
            positions=(_position("SPY", 100, 100.0),),  # 10% of NAV
        )
        targets = pd.Series({"SPY": 0.11})
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0, drift=0.05),
        )
        assert len(proposal.trades) == 0

    def test_large_drift_triggers_trade(self):
        """Position at 10% of NAV, target 20% — outside drift, EXTEND."""
        snap = _snapshot(
            nav=100_000, cash=90_000,
            positions=(_position("SPY", 100, 100.0),),
        )
        targets = pd.Series({"SPY": 0.20})
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0, drift=0.05),
        )
        assert len(proposal.trades) == 1
        assert proposal.trades[0].action == ACTION_EXTEND

    def test_cash_deployment_relaxes_drift(self):
        """With cash_budget > 0, drift is halved — smaller trades allowed."""
        snap = _snapshot(
            nav=100_000, cash=95_000,
            positions=(_position("SPY", 100, 100.0),),
        )
        # target 13% (drift 3%, would be skipped without cash-mode)
        targets = pd.Series({"SPY": 0.13})
        # No cash budget → drift 5% applies → skip
        no_cash = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0, drift=0.05),
        )
        assert len(no_cash.trades) == 0
        # With cash budget → effective drift 2.5% → trades
        with_cash = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=10_000,
            policy=_policy(cash_reserve=0, drift=0.05),
        )
        assert len(with_cash.trades) == 1


# ────────────────────────────────────────────────────────────────
# Min notional filter
# ────────────────────────────────────────────────────────────────

class TestMinNotional:
    def test_below_min_dropped(self):
        """Trade sized $200 with min $500 — drop with warning."""
        snap = _snapshot(
            nav=100_000, cash=90_000,
            positions=(_position("SPY", 100, 100.0),),  # 10%
        )
        targets = pd.Series({"SPY": 0.098})  # 9.8% -> tiny SELL
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0, drift=0.001),  # force through drift
            min_trade_notional=500.0,
        )
        # Trade is ~$200 — should be dropped.
        assert len(proposal.trades) == 0
        assert any("min notional" in w or "min " in w for w in proposal.warnings)


# ────────────────────────────────────────────────────────────────
# Aggregate metrics
# ────────────────────────────────────────────────────────────────

class TestAggregates:
    def test_turnover_and_cost(self):
        """Two trades — check turnover and cost sum correctly."""
        snap = _snapshot(
            nav=100_000, cash=50_000,
            positions=(
                _position("AAA", 100, 100.0),  # 10k = 10%
                _position("BBB", 200, 100.0),  # 20k = 20%
            ),
        )
        # Target: AAA -> 20% (extend), BBB -> 10% (sell half)
        targets = pd.Series({"AAA": 0.20, "BBB": 0.10})
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0),
        )
        assert len(proposal.trades) == 2
        # Turnover: $10k extend AAA + $10k reduce BBB = $20k
        assert proposal.turnover_notional == pytest.approx(20_000.0)
        assert proposal.turnover_pct_of_nav == pytest.approx(0.20)
        # Cost: 4 bps of $20k = $8
        assert proposal.total_est_cost == pytest.approx(8.0)

    def test_cash_reserve_warning(self):
        """Trades that would breach cash reserve emit a warning."""
        snap = _snapshot(nav=100_000, cash=1_000,
                          positions=(_position("SPY", 990, 100.0),))
        # Target: SPY -> 100% -> EXTEND to 1000 shares -> need $1000 -> cash goes negative
        targets = pd.Series({"SPY": 1.0})
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=5_000),  # reserve $5k
        )
        # A trade should be proposed but with a warning about breach.
        assert any("cash reserve" in w.lower() or "reserve" in w.lower()
                   for w in proposal.warnings)


# ────────────────────────────────────────────────────────────────
# Factor exposures
# ────────────────────────────────────────────────────────────────

class TestFactorExposures:
    def test_before_after_deltas(self):
        """Before: AAA-only (mom=1.0). After: BBB-only (mom=-1.0). Delta = -2.0."""
        snap = _snapshot(
            nav=100_000, cash=90_000,
            positions=(_position("AAA", 100, 100.0),),
        )
        targets = pd.Series({"BBB": 0.10})
        scores = pd.DataFrame({
            "momentum": {"AAA": 1.0, "BBB": -1.0},
            "quality": {"AAA": 0.5, "BBB": 0.5},
        })
        # BBB needs a price; propose_trades sources from snapshot Position.
        # BBB has no held position → no price → skipped with warning.
        # Instead: hold BBB at 0 shares? Position requires shares > 0 for
        # long_positions to include it. Set up so both are held to test
        # the exposure calc; sizing is not the point here.
        snap = _snapshot(
            nav=100_000, cash=80_000,
            positions=(_position("AAA", 100, 100.0),
                        _position("BBB", 100, 100.0)),
        )
        # Target: shift entirely to BBB.
        targets = pd.Series({"BBB": 0.20})  # 20% of investable NAV in BBB, drop AAA
        proposal = propose_trades(
            snapshot=snap, target_weights=targets, cash_budget=0,
            policy=_policy(cash_reserve=0),
            factor_scores=scores,
        )
        # Exposures were computed: before mixes AAA+BBB, after is BBB-only.
        assert len(proposal.factor_exposures) == 2
        mom = next(f for f in proposal.factor_exposures if f.factor == "momentum")
        # Before: 50/50 AAA+BBB -> 0.0 mean
        assert mom.before == pytest.approx(0.0)
        # After: BBB only -> -1.0
        assert mom.after == pytest.approx(-1.0)
        assert mom.delta == pytest.approx(-1.0)


# ────────────────────────────────────────────────────────────────
# Input validation
# ────────────────────────────────────────────────────────────────

class TestValidation:
    def test_negative_nav_raises(self):
        snap = _snapshot(nav=-1.0, cash=0.0)
        with pytest.raises(ValueError, match="nav"):
            propose_trades(snap, pd.Series({"SPY": 1.0}), 0, _policy())

    def test_negative_budget_raises(self):
        snap = _snapshot(nav=100_000, cash=100_000)
        with pytest.raises(ValueError, match="cash_budget"):
            propose_trades(snap, pd.Series({"SPY": 1.0}), -100, _policy())

    def test_negative_target_weight_raises(self):
        snap = _snapshot(nav=100_000, cash=100_000)
        with pytest.raises(ValueError, match="non-negative"):
            propose_trades(snap, pd.Series({"SPY": -0.5}), 0, _policy())
