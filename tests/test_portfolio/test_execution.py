"""Unit tests for `src.portfolio.execution.execute_proposal`.

Order-construction module — the only path that turns a TradeProposal
into real ib_insync order payloads. Per CLAUDE.md:
    "Stop-loss and order-construction changes require explicit unit
     tests that pin the trigger condition or the exact order payload.
     'Looks right' is not acceptable."

We mock ib_insync at the module boundary using a `StubIB` that records
every payload passed to `placeOrder`. Tests then assert on:
  - The exact side (BUY/SELL) and quantity emitted per Trade row
  - Trailing stop attachment ONLY on BUY / EXTEND, never on SELL
  - Trail quantity covers new shares only (EXTEND) or full position (BUY)
  - Dry-run mode never touches the stub
  - Audit log is written with correct structure
  - Errors on individual orders do not abort the whole walk
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest

from src.portfolio.execution import (
    ExecutionReceipt, OrderResult,
    execute_proposal,
)
from src.portfolio.policy import (
    FactorLookbacks, FactorWeights, SmartBetaPolicy,
)
from src.portfolio.proposal import (
    ACTION_BUY, ACTION_EXTEND, ACTION_SELL,
    Trade, TradeProposal,
)


pytestmark = pytest.mark.unit


# ────────────────────────────────────────────────────────────────
# Fixtures + stubs
# ────────────────────────────────────────────────────────────────

def _policy(trail_pct: float = 0.10) -> SmartBetaPolicy:
    return SmartBetaPolicy(
        name="exec-test", version=1, tax_status="taxable",
        num_positions=5, min_weight=0.02, max_weight=0.30,
        factor_weights=FactorWeights(0.35, 0.30, 0.20, 0.15),
        factor_lookbacks=FactorLookbacks(252, 21, 252, 252, 60),
        rebalance_frequency="bimonthly", drift_threshold=0.05,
        entry_stop_loss_pct=0.12, trailing_stop_pct=trail_pct,
        cash_reserve=0.0,
        risk_aversion=1.0, robustness_penalty=0.5, turnover_penalty=0.1,
    )


def _trade(ticker: str, action: str, delta: int, price: float = 100.0,
           current: int = 0) -> Trade:
    if action == ACTION_SELL:
        target = max(0, current + delta)
    else:
        target = current + delta
    return Trade(
        ticker=ticker, action=action,
        current_shares=current, target_shares=target,
        delta_shares=delta, market_price=price,
        delta_notional=abs(delta) * price,
        est_cost=abs(delta) * price * 0.0004,
        current_weight_pct=0.05, target_weight_pct=0.05,
        weight_gap_pct=0.0,
    )


def _proposal(trades) -> TradeProposal:
    total = sum(t.delta_notional for t in trades)
    return TradeProposal(
        trades=tuple(trades),
        turnover_notional=total, turnover_pct_of_nav=0.1,
        total_est_cost=total * 0.0004,
        factor_exposures=tuple(),
        investable_nav=100_000, cash_after=50_000,
        n_positions_after=len(trades),
        warnings=tuple(),
    )


class _StubOrderStatus:
    def __init__(self, status: str = "Submitted"):
        self.status = status


class _StubOrder:
    def __init__(self, order_id: int = 0):
        self.orderId = order_id
        # Fields set by MarketOrder / Order construction:
        self.action: Optional[str] = None
        self.totalQuantity: Optional[float] = None
        self.orderType: Optional[str] = None
        self.trailingPercent: Optional[float] = None
        self.tif: Optional[str] = None


class _StubTrade:
    """What ib.placeOrder returns."""
    def __init__(self, order: _StubOrder, status: str = "Submitted"):
        self.order = order
        self.orderStatus = _StubOrderStatus(status)


class StubIB:
    """Minimal ib_insync-shaped stub. Records every call for assertions."""

    def __init__(self, fail_on: Optional[set[str]] = None):
        self.qualify_calls: list[Any] = []
        self.place_calls: list[tuple[Any, Any]] = []
        self.sleep_calls: list[float] = []
        self.next_id = 1000
        self.fail_on: set[str] = fail_on or set()

    def qualifyContracts(self, *contracts) -> None:
        self.qualify_calls.extend(contracts)

    def placeOrder(self, contract, order) -> _StubTrade:
        ticker = getattr(contract, "symbol", "?")
        if ticker in self.fail_on:
            raise RuntimeError(f"stub failure on {ticker}")
        # Give the order an ID so downstream code can read it.
        order.orderId = self.next_id
        self.next_id += 1
        self.place_calls.append((contract, order))
        return _StubTrade(order)

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)


# ────────────────────────────────────────────────────────────────
# Dry run — no orders, but payloads simulated
# ────────────────────────────────────────────────────────────────

class TestDryRun:
    def test_no_orders_placed_in_dry_run(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([
            _trade("SPY", ACTION_BUY, 10),
            _trade("VTI", ACTION_SELL, -5, current=5),
        ])
        receipt = execute_proposal(
            proposal, _policy(), ib=stub, dry_run=True,
            audit_dir=tmp_path,
        )
        assert len(stub.place_calls) == 0
        assert receipt.dry_run is True
        # One result per trade + one trail per BUY/EXTEND.
        assert len(receipt.results) == 2
        # SPY is a BUY → TRAIL attached. VTI is a SELL → no TRAIL.
        assert len(receipt.trail_results) == 1
        assert receipt.trail_results[0].ticker == "SPY"

    def test_dry_run_status_marker(self, tmp_path):
        proposal = _proposal([_trade("SPY", ACTION_BUY, 10)])
        receipt = execute_proposal(
            proposal, _policy(), ib=None, dry_run=True,
            audit_dir=tmp_path,
        )
        for r in list(receipt.results) + list(receipt.trail_results):
            assert r.status == "DRY_RUN"


# ────────────────────────────────────────────────────────────────
# Live path — exact payload assertions on the stub
# ────────────────────────────────────────────────────────────────

class TestOrderPayloads:
    def test_buy_places_market_buy(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([_trade("SPY", ACTION_BUY, 25, price=400.0)])
        execute_proposal(proposal, _policy(), ib=stub, dry_run=False,
                          audit_dir=tmp_path)
        # First placeOrder call = the market BUY.
        contract, order = stub.place_calls[0]
        assert contract.symbol == "SPY"
        assert order.action == "BUY"
        assert order.totalQuantity == 25
        assert order.orderType == "MKT"

    def test_sell_places_market_sell(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([
            _trade("XLK", ACTION_SELL, -30, price=200.0, current=30),
        ])
        execute_proposal(proposal, _policy(), ib=stub, dry_run=False,
                          audit_dir=tmp_path)
        contract, order = stub.place_calls[0]
        assert contract.symbol == "XLK"
        assert order.action == "SELL"
        assert order.totalQuantity == 30

    def test_extend_places_market_buy_for_delta_only(self, tmp_path):
        stub = StubIB()
        # Currently hold 100, extending by 25 more.
        proposal = _proposal([
            _trade("VTI", ACTION_EXTEND, 25, price=200.0, current=100),
        ])
        execute_proposal(proposal, _policy(), ib=stub, dry_run=False,
                          audit_dir=tmp_path)
        contract, order = stub.place_calls[0]
        assert contract.symbol == "VTI"
        assert order.action == "BUY"
        # ONLY the new shares, not the total position.
        assert order.totalQuantity == 25


class TestTrailAttachment:
    def test_trail_attached_after_buy(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([_trade("SPY", ACTION_BUY, 50)])
        receipt = execute_proposal(
            proposal, _policy(trail_pct=0.10), ib=stub,
            dry_run=False, audit_dir=tmp_path,
        )
        assert len(stub.place_calls) == 2  # MARKET + TRAIL
        trail_contract, trail_order = stub.place_calls[1]
        assert trail_contract.symbol == "SPY"
        assert trail_order.action == "SELL"
        assert trail_order.orderType == "TRAIL"
        assert trail_order.tif == "GTC"
        # trailingPercent is IB's whole-percent form: 0.10 → 10.0
        assert trail_order.trailingPercent == 10.0
        assert trail_order.totalQuantity == 50  # full new position

    def test_trail_covers_new_shares_only_on_extend(self, tmp_path):
        stub = StubIB()
        # Held 100, adding 25. Trail should cover 25, not 125.
        proposal = _proposal([
            _trade("VTI", ACTION_EXTEND, 25, price=200.0, current=100),
        ])
        execute_proposal(proposal, _policy(), ib=stub, dry_run=False,
                          audit_dir=tmp_path)
        trail_contract, trail_order = stub.place_calls[1]
        assert trail_contract.symbol == "VTI"
        assert trail_order.orderType == "TRAIL"
        assert trail_order.totalQuantity == 25

    def test_no_trail_on_sell(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([
            _trade("XLK", ACTION_SELL, -30, current=30),
        ])
        execute_proposal(proposal, _policy(), ib=stub, dry_run=False,
                          audit_dir=tmp_path)
        # Only the MARKET SELL — no TRAIL.
        assert len(stub.place_calls) == 1
        assert stub.place_calls[0][1].orderType == "MKT"

    def test_trail_percent_reflects_policy(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([_trade("SPY", ACTION_BUY, 10)])
        execute_proposal(
            proposal, _policy(trail_pct=0.15), ib=stub,
            dry_run=False, audit_dir=tmp_path,
        )
        _, trail_order = stub.place_calls[1]
        assert trail_order.trailingPercent == pytest.approx(15.0)

    def test_disable_trail_attachment(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([_trade("SPY", ACTION_BUY, 10)])
        receipt = execute_proposal(
            proposal, _policy(), ib=stub, dry_run=False,
            audit_dir=tmp_path, attach_trailing_stops=False,
        )
        assert len(stub.place_calls) == 1  # MARKET only
        assert len(receipt.trail_results) == 0


# ────────────────────────────────────────────────────────────────
# Order ordering — sells before extends before buys
# ────────────────────────────────────────────────────────────────

class TestExecutionOrder:
    def test_sell_extend_buy_ordering(self, tmp_path):
        stub = StubIB()
        # Deliberately shuffled input order.
        proposal = _proposal([
            _trade("QQQ", ACTION_BUY, 10),
            _trade("XLK", ACTION_SELL, -5, current=5),
            _trade("VTI", ACTION_EXTEND, 3, current=100),
        ])
        execute_proposal(proposal, _policy(), ib=stub, dry_run=False,
                          audit_dir=tmp_path)
        # place_calls is [SELL_XLK, EXTEND_VTI_market, EXTEND_VTI_trail, BUY_QQQ_market, BUY_QQQ_trail]
        market_calls = [c for c, o in stub.place_calls if o.orderType == "MKT"]
        assert [c.symbol for c in market_calls] == ["XLK", "VTI", "QQQ"]


# ────────────────────────────────────────────────────────────────
# Error handling — one bad order does not abort the walk
# ────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_error_on_one_ticker_does_not_abort_others(self, tmp_path):
        stub = StubIB(fail_on={"BADTICKER"})
        proposal = _proposal([
            _trade("SPY", ACTION_BUY, 10),
            _trade("BADTICKER", ACTION_BUY, 5),
            _trade("VTI", ACTION_BUY, 20),
        ])
        receipt = execute_proposal(
            proposal, _policy(), ib=stub, dry_run=False,
            audit_dir=tmp_path,
        )
        # Good tickers ran; bad ticker recorded ERROR.
        statuses = {r.ticker: r.status for r in receipt.results}
        assert statuses["SPY"] == "Submitted"
        assert statuses["VTI"] == "Submitted"
        assert statuses["BADTICKER"] == "ERROR"

    def test_dry_run_without_ib_ok(self, tmp_path):
        proposal = _proposal([_trade("SPY", ACTION_BUY, 10)])
        receipt = execute_proposal(proposal, _policy(), ib=None,
                                    dry_run=True, audit_dir=tmp_path)
        assert receipt.dry_run
        assert len(receipt.results) == 1

    def test_live_without_ib_raises(self, tmp_path):
        proposal = _proposal([_trade("SPY", ACTION_BUY, 10)])
        with pytest.raises(ValueError, match="ib client is required"):
            execute_proposal(proposal, _policy(), ib=None,
                              dry_run=False, audit_dir=tmp_path)


# ────────────────────────────────────────────────────────────────
# Audit log — every trade written as one JSON line
# ────────────────────────────────────────────────────────────────

class TestAuditLog:
    def test_audit_log_written(self, tmp_path):
        stub = StubIB()
        proposal = _proposal([
            _trade("SPY", ACTION_BUY, 10),
            _trade("VTI", ACTION_SELL, -5, current=5),
        ])
        receipt = execute_proposal(
            proposal, _policy(), ib=stub, dry_run=False,
            audit_dir=tmp_path,
        )
        audit_path = Path(receipt.audit_log_path)
        assert audit_path.exists()
        lines = audit_path.read_text().strip().split("\n")
        # 2 market trades + 1 trail (SPY only, VTI is a sell) = 3 lines.
        assert len(lines) == 3
        # Each line is valid JSON.
        payloads = [json.loads(line) for line in lines]
        actions = [p["action"] for p in payloads]
        assert ACTION_BUY in actions
        assert ACTION_SELL in actions
        assert "TRAIL" in actions

    def test_dry_run_still_writes_audit(self, tmp_path):
        proposal = _proposal([_trade("SPY", ACTION_BUY, 10)])
        receipt = execute_proposal(
            proposal, _policy(), ib=None, dry_run=True,
            audit_dir=tmp_path,
        )
        audit_path = Path(receipt.audit_log_path)
        assert audit_path.exists()
        lines = audit_path.read_text().strip().split("\n")
        assert all("DRY_RUN" in line for line in lines)
