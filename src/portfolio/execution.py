"""Execute a TradeProposal against a live IB account.

**Safety-critical.** This module is the only path that places real
orders. Every code change here requires explicit unit tests
(CLAUDE.md rule).

Design:
- **Dry-run by default.** `execute_proposal(..., dry_run=True)` walks
  the proposal, constructs the exact order payloads, writes the audit
  log — but does NOT call `ib.placeOrder`. Use this to smoke-test the
  Apply flow without risking capital.
- **Two-key safety in the UI.** The Streamlit tab requires an explicit
  "IB Gateway Read-Only API is OFF" checkbox AND an explicit
  "I understand this places real orders" checkbox before the Apply
  button is enabled. This module trusts its `dry_run` argument — the
  UI is responsible for setting it to False only under the two-key
  condition.
- **Audit log always.** Every attempted trade (dry or live) is written
  to JSONL under `~/trade_data/ETFTrader/execution_log/`. One line per
  trade, one line per trailing-stop attachment.
- **Trailing stops on BUY / EXTEND only.** For a BUY, TRAIL covers the
  full new position. For an EXTEND, TRAIL covers only the new shares —
  existing TRAILs on the pre-existing position are not touched
  (`_compute_trail_qty` design from `scripts/ib_execute_trades.py`).
- **Market orders.** For ETFs the spread is typically tight; the simple
  choice keeps the code auditable. Limit-order flavour is left as a
  future enhancement.

Not in scope for this module:
- Sizing decisions (that's `proposal.py`).
- LLM commentary (that's `explain.py`).
- Position reconciliation post-fill (that's `ib_state.py` on next snapshot).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

try:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

from src.portfolio.policy import SmartBetaPolicy
from src.portfolio.proposal import (
    ACTION_BUY, ACTION_EXTEND, ACTION_SELL,
    Trade, TradeProposal,
)


DEFAULT_AUDIT_DIR = (
    Path.home() / "trade_data" / "ETFTrader" / "execution_log"
)

# Number of seconds to wait after placing an order for IB to report
# an acknowledgement. Kept short so a fail-fast is possible on a bad
# connection.
_ACK_WAIT_SEC = 2.0


# ────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrderResult:
    """The outcome of one placeOrder call.

    For dry-run mode, `status = "DRY_RUN"` and `order_id` is None.
    For a real fill, `status` mirrors IB's orderStatus.status.
    """
    ticker: str
    action: str          # BUY / SELL / EXTEND
    order_kind: str      # MKT / TRAIL
    shares: int
    limit_or_trail_pct: Optional[float]  # None for MKT, decimal for TRAIL
    order_id: Optional[int]
    status: str
    message: str
    timestamp: str


@dataclass(frozen=True)
class ExecutionReceipt:
    """Full record of an execution attempt.

    Attributes:
        started_at: ISO timestamp when execute_proposal began.
        finished_at: ISO timestamp when the last order returned.
        dry_run: True if no orders actually went to IB.
        n_trades: number of Trade rows in the input proposal.
        results: per-trade result rows (main orders).
        trail_results: per-BUY/EXTEND trailing-stop results.
        errors: strings for any exceptions caught during placement.
        audit_log_path: where the JSONL log was written.
    """
    started_at: str
    finished_at: str
    dry_run: bool
    n_trades: int
    results: tuple[OrderResult, ...]
    trail_results: tuple[OrderResult, ...]
    errors: tuple[str, ...] = field(default_factory=tuple)
    audit_log_path: Optional[str] = None

    @property
    def n_ok(self) -> int:
        return sum(
            1 for r in list(self.results) + list(self.trail_results)
            if r.status in ("Filled", "Submitted", "PreSubmitted", "DRY_RUN")
        )

    @property
    def n_failed(self) -> int:
        total = len(self.results) + len(self.trail_results)
        return total - self.n_ok


# ────────────────────────────────────────────────────────────────
# IB client interface (dependency injection point for testing)
# ────────────────────────────────────────────────────────────────

class IBExecutor(Protocol):
    """Minimal surface for placing orders. `ib_insync.IB` satisfies this.

    Kept as a Protocol so tests can pass a mock/stub without depending
    on the real IB Gateway being up.
    """
    def qualifyContracts(self, *contracts) -> None: ...
    def placeOrder(self, contract, order): ...
    def sleep(self, seconds: float) -> None: ...


# ────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────

def execute_proposal(
    proposal: TradeProposal,
    policy: SmartBetaPolicy,
    ib: Optional[IBExecutor] = None,
    dry_run: bool = True,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    attach_trailing_stops: bool = True,
    ack_wait_sec: float = _ACK_WAIT_SEC,
) -> ExecutionReceipt:
    """Walk the proposal, place orders, attach trailing stops, log everything.

    Args:
        proposal: the ground-truth trade blotter from `propose_trades`.
        policy: for `trailing_stop_pct`.
        ib: an ib_insync `IB` client (or test stub). Required if
            `dry_run=False`.
        dry_run: if True, construct payloads and log them but do NOT
            call `ib.placeOrder`. Default True for safety.
        audit_dir: where to write the JSONL log.
        attach_trailing_stops: if True, follow every BUY and EXTEND with
            a TRAIL order covering the new shares.
        ack_wait_sec: seconds to sleep after each order to allow IB to
            report an ack.

    Returns:
        ExecutionReceipt with per-trade results.

    Never raises on individual order failure — errors are captured and
    the walk continues. Only raises for setup issues (missing IB when
    not dry-run).
    """
    if not dry_run and ib is None:
        raise ValueError("ib client is required when dry_run=False")

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"execution_{datetime.utcnow():%Y%m%d_%H%M%S}.jsonl"
    audit_log_fh = audit_path.open("w", encoding="utf-8")

    started_at = _now_iso()
    results: list[OrderResult] = []
    trail_results: list[OrderResult] = []
    errors: list[str] = []

    try:
        # Order: SELLs first (raise cash), then EXTENDs (redeploy), then
        # BUYs (fresh positions). This is the natural ordering for a
        # rebalance and matches the operator's usual manual walk.
        actioned = sorted(
            proposal.trades,
            key=lambda t: {ACTION_SELL: 0, ACTION_EXTEND: 1, ACTION_BUY: 2}[t.action],
        )

        for trade in actioned:
            result = _place_market_trade(trade, ib, dry_run, ack_wait_sec)
            results.append(result)
            _write_audit(audit_log_fh, result, trade)

            if attach_trailing_stops and trade.delta_shares > 0:
                trail_qty = _compute_trail_qty(trade)
                if trail_qty <= 0:
                    continue
                trail = _place_trailing_stop(
                    ticker=trade.ticker,
                    shares=trail_qty,
                    trail_pct=policy.trailing_stop_pct,
                    ib=ib, dry_run=dry_run, ack_wait_sec=ack_wait_sec,
                )
                trail_results.append(trail)
                _write_audit(audit_log_fh, trail, None)

    except Exception as exc:  # noqa: BLE001
        errors.append(f"execute_proposal: unexpected error: {exc}")
        logger.exception("execute_proposal caught unexpected exception")
    finally:
        audit_log_fh.close()

    finished_at = _now_iso()
    return ExecutionReceipt(
        started_at=started_at,
        finished_at=finished_at,
        dry_run=dry_run,
        n_trades=len(proposal.trades),
        results=tuple(results),
        trail_results=tuple(trail_results),
        errors=tuple(errors),
        audit_log_path=str(audit_path),
    )


# ────────────────────────────────────────────────────────────────
# Order placement helpers
# ────────────────────────────────────────────────────────────────

def _place_market_trade(trade: Trade, ib: Optional[IBExecutor],
                        dry_run: bool, ack_wait_sec: float) -> OrderResult:
    """Place a market order. Returns the result row."""
    ib_side = "BUY" if trade.delta_shares > 0 else "SELL"
    shares = abs(int(trade.delta_shares))

    if dry_run:
        return OrderResult(
            ticker=trade.ticker, action=trade.action, order_kind="MKT",
            shares=shares, limit_or_trail_pct=None,
            order_id=None, status="DRY_RUN",
            message=f"DRY_RUN {ib_side} {shares} {trade.ticker} @ MKT",
            timestamp=_now_iso(),
        )

    # Real placement — import ib_insync lazily so unit tests don't need it.
    try:
        from ib_insync import Stock, MarketOrder
    except ImportError as exc:
        return _error_result(trade.ticker, trade.action, "MKT", shares,
                              f"ib_insync not installed: {exc}")

    contract = Stock(trade.ticker, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception as exc:  # noqa: BLE001
        return _error_result(trade.ticker, trade.action, "MKT", shares,
                              f"qualifyContracts failed: {exc}")

    order = MarketOrder(ib_side, shares)
    try:
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(ack_wait_sec)
        status = trade_obj.orderStatus.status
        order_id = getattr(trade_obj.order, "orderId", None)
        return OrderResult(
            ticker=trade.ticker, action=trade.action, order_kind="MKT",
            shares=shares, limit_or_trail_pct=None,
            order_id=order_id, status=status,
            message=f"{ib_side} {shares} {trade.ticker} @ MKT",
            timestamp=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(trade.ticker, trade.action, "MKT", shares,
                              f"placeOrder failed: {exc}")


def _place_trailing_stop(ticker: str, shares: int, trail_pct: float,
                          ib: Optional[IBExecutor], dry_run: bool,
                          ack_wait_sec: float) -> OrderResult:
    """Attach a TRAIL SELL covering `shares` at `trail_pct` (decimal, e.g. 0.10)."""
    if dry_run:
        return OrderResult(
            ticker=ticker, action="TRAIL", order_kind="TRAIL",
            shares=shares, limit_or_trail_pct=trail_pct,
            order_id=None, status="DRY_RUN",
            message=f"DRY_RUN TRAIL SELL {shares} {ticker} @ {trail_pct:.1%}",
            timestamp=_now_iso(),
        )

    try:
        from ib_insync import Stock, Order
    except ImportError as exc:
        return _error_result(ticker, "TRAIL", "TRAIL", shares,
                              f"ib_insync not installed: {exc}")

    contract = Stock(ticker, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception as exc:  # noqa: BLE001
        return _error_result(ticker, "TRAIL", "TRAIL", shares,
                              f"qualifyContracts failed: {exc}")

    order = Order()
    order.action = "SELL"
    order.totalQuantity = shares
    order.orderType = "TRAIL"
    # IB expects trailing_percent as a whole percent (10 not 0.10).
    order.trailingPercent = float(trail_pct * 100)
    order.tif = "GTC"

    try:
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(ack_wait_sec)
        status = trade_obj.orderStatus.status
        order_id = getattr(trade_obj.order, "orderId", None)
        return OrderResult(
            ticker=ticker, action="TRAIL", order_kind="TRAIL",
            shares=shares, limit_or_trail_pct=trail_pct,
            order_id=order_id, status=status,
            message=f"TRAIL SELL {shares} {ticker} @ {trail_pct:.1%}",
            timestamp=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(ticker, "TRAIL", "TRAIL", shares,
                              f"placeOrder failed: {exc}")


def _compute_trail_qty(trade: Trade) -> int:
    """Number of shares the new TRAIL should cover.

    Design decision from `scripts/ib_execute_trades.py`: for an EXTEND,
    the new TRAIL covers only the new shares — existing TRAILs on the
    prior position are NOT cancelled or rebased.

    - BUY (fresh): cover the full new position (delta = target).
    - EXTEND: cover only the delta shares.
    - SELL: never — no TRAIL is placed on a reducing trade.
    """
    if trade.delta_shares <= 0:
        return 0
    return int(trade.delta_shares)


# ────────────────────────────────────────────────────────────────
# Audit log
# ────────────────────────────────────────────────────────────────

def _write_audit(fh, result: OrderResult, trade: Optional[Trade]) -> None:
    payload = asdict(result)
    if trade is not None:
        payload["source_trade"] = {
            "delta_shares": trade.delta_shares,
            "market_price": trade.market_price,
            "delta_notional": trade.delta_notional,
            "target_weight_pct": trade.target_weight_pct,
        }
    fh.write(json.dumps(payload, default=str) + "\n")
    fh.flush()


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _error_result(ticker: str, action: str, kind: str, shares: int,
                   message: str) -> OrderResult:
    return OrderResult(
        ticker=ticker, action=action, order_kind=kind,
        shares=shares, limit_or_trail_pct=None,
        order_id=None, status="ERROR", message=message,
        timestamp=_now_iso(),
    )
