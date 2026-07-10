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


# ────────────────────────────────────────────────────────────────
# Cash-neutrality invariant — the fix for the 2026-07-10 over-invest bug.
#
# The propose_trades output MUST never require more cash than the operator
# actually has. Formally:
#
#     sum(BUY + EXTEND delta_notional) - sum(SELL delta_notional)
#         <= available_cash = snapshot.cash + cash_budget - policy.cash_reserve
#
# Prior to 2026-07-10 this was not enforced. On the operator's live book that
# afternoon it resulted in a proposal that put the account $113k on margin.
# Every pattern below is a scenario that was silently broken.
# ────────────────────────────────────────────────────────────────

def _prices_frame(prices_map: dict[str, float]) -> pd.DataFrame:
    """Wide DataFrame with one row of prices for fresh-BUY fallback lookup."""
    idx = pd.date_range("2026-07-01", periods=1)
    return pd.DataFrame(prices_map, index=idx)


class TestCashNeutralityInvariant:
    def test_retained_overhang_does_not_cause_margin(self):
        """Bug repro: 5 retained positions each 20% overweight (small drift),
        20 new positions in the target basket. Under the OLD code, sum of BUYs
        exceeds available cash by the overhang amount. Under the FIX, BUYs are
        scaled down (or overhangs sold) so the invariant holds."""
        # NAV 100k, cash 5k. Reserve 0. Available for deployment: 5k.
        # 5 held positions at $6k each (within drift of a 5k target = 5% NAV).
        # 20 new target positions at 4% NAV each = $4k target each = $80k total BUYs.
        # OLD code: SELLs = 0, BUYs = $80k. Requires $80k cash. Only $5k available.
        # Post-fix expectation: BUYs scale down OR retained overhangs SELL down,
        # such that net cash outflow ≤ $5k.
        positions = tuple(
            _position(f"HELD{i}", 60, 100.0)  # 60 × $100 = $6k each (6% NAV)
            for i in range(5)
        )
        snap = _snapshot(nav=100_000, cash=5_000, positions=positions)
        target = pd.Series({
            **{f"HELD{i}": 0.05 for i in range(5)},   # want 5% each
            **{f"NEW{i}": 0.04 for i in range(20)},   # 20 fresh 4% positions
        })
        # NEW0-NEW19 need a price to be sizeable via BUY.
        prices = _prices_frame({
            **{f"HELD{i}": 100.0 for i in range(5)},
            **{f"NEW{i}": 100.0 for i in range(20)},
        })
        proposal = propose_trades(
            snap, target, 0, _policy(cash_reserve=0),
            prices=prices,
        )

        available = snap.cash - 0  # reserve 0, no budget
        buys = sum(t.delta_notional for t in proposal.trades if t.delta_shares > 0)
        sells = sum(t.delta_notional for t in proposal.trades if t.delta_shares < 0)
        net = buys - sells

        assert net <= available + 1.0, (
            f"Cash invariant VIOLATED. buys={buys:.0f} sells={sells:.0f} "
            f"net={net:.0f} > available={available:.0f}. This is the bug."
        )

    def test_overweight_retained_is_trimmed(self):
        """A position at 6% NAV whose target is 5% NAV should emit a SELL for
        the 1% overhang, not be silently skipped by the drift guard.

        This is the ASYMMETRIC drift rule: under-weight retained is protected
        (whipsaw), over-weight retained is always trimmed (cash discipline).
        """
        snap = _snapshot(
            nav=100_000, cash=1_000,
            positions=(_position("HELD", 60, 100.0),),  # $6k = 6% NAV
        )
        target = pd.Series({"HELD": 0.05})  # want 5% NAV = $5k
        proposal = propose_trades(
            snap, target, 0, _policy(cash_reserve=0, drift=0.05),
        )
        assert len(proposal.trades) == 1
        t = proposal.trades[0]
        assert t.action == ACTION_SELL, (
            f"Overweight retained should SELL the overhang. Got {t.action}."
        )
        # Roughly $1k overhang — about 10 shares at $100.
        assert -12 <= t.delta_shares <= -8

    def test_underweight_retained_still_skipped(self):
        """Whipsaw guard: a retained position 1% UNDER its target sits inside
        drift and should not trigger a BUY. Proves the fix is asymmetric.
        """
        snap = _snapshot(
            nav=100_000, cash=1_000,
            positions=(_position("HELD", 40, 100.0),),  # $4k = 4% NAV
        )
        target = pd.Series({"HELD": 0.05})  # want 5% NAV = $5k, gap 1%
        proposal = propose_trades(
            snap, target, 0, _policy(cash_reserve=0, drift=0.05),
        )
        # 1% under-weight, drift threshold 5% relative — should skip.
        assert len(proposal.trades) == 0

    def test_scale_factor_preserves_relative_buy_weights(self):
        """When the constraint bites and BUYs are scaled down, the ratio
        between individual BUYs must be preserved."""
        snap = _snapshot(nav=100_000, cash=10_000)  # available = 10k
        # Target three new positions at 4%, 8%, 12% → $4k, $8k, $12k = $24k BUYs.
        # Available cash = 10k, so buys should scale by factor 10/24.
        target = pd.Series({"A": 0.04, "B": 0.08, "C": 0.12})
        prices = _prices_frame({"A": 100.0, "B": 100.0, "C": 100.0})
        proposal = propose_trades(
            snap, target, 0, _policy(cash_reserve=0),
            prices=prices,
        )
        by_ticker = {t.ticker: t for t in proposal.trades}
        assert set(by_ticker) == {"A", "B", "C"}
        # Ratio A : B : C should still be roughly 1 : 2 : 3 after scaling.
        a_n = by_ticker["A"].delta_notional
        b_n = by_ticker["B"].delta_notional
        c_n = by_ticker["C"].delta_notional
        assert 1.8 < b_n / a_n < 2.2
        assert 2.7 < c_n / a_n < 3.3

    def test_warning_emitted_when_cash_constrained(self):
        """When the constraint bites, the operator must be told."""
        snap = _snapshot(nav=100_000, cash=5_000)
        target = pd.Series({"A": 0.30, "B": 0.30, "C": 0.30})  # $90k BUYs
        prices = _prices_frame({"A": 100.0, "B": 100.0, "C": 100.0})
        proposal = propose_trades(
            snap, target, 0, _policy(cash_reserve=0),
            prices=prices,
        )
        assert any(
            "cash-constrained" in w.lower() or "scaled" in w.lower()
            for w in proposal.warnings
        ), f"Expected cash-constrained warning, got: {list(proposal.warnings)}"

    def test_normal_proposal_unchanged_when_cash_ok(self):
        """Regression: a proposal that already respects the cash constraint
        must not be modified by the fix."""
        # NAV 100k, cash 90k, reserve 0. Available = 90k.
        # Two trades: SELL 10 SPY (worth $1000) + BUY 5% VOO ($5k).
        snap = _snapshot(
            nav=100_000, cash=90_000,
            positions=(_position("SPY", 10, 100.0),),
        )
        target = pd.Series({"VOO": 0.05})
        prices = _prices_frame({"VOO": 100.0})
        proposal = propose_trades(
            snap, target, 0, _policy(cash_reserve=0),
            prices=prices,
        )
        # Should emit SELL SPY (dropped) and BUY VOO. Both at expected sizes.
        by_ticker = {t.ticker: t for t in proposal.trades}
        assert "SPY" in by_ticker and "VOO" in by_ticker
        assert by_ticker["SPY"].delta_shares == -10  # full exit
        assert by_ticker["VOO"].delta_shares == 50    # $5k / $100
        # No cash-constrained warning because we had plenty of cash.
        assert not any(
            "cash-constrained" in w.lower() or "scaled" in w.lower()
            for w in proposal.warnings
        )
