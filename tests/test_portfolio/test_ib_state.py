"""Unit tests for src.portfolio.ib_state.

Per CLAUDE.md, tests must never touch a live IB Gateway — ``ib_insync``
is mocked at the module boundary so the deterministic behaviour of the
snapshot extraction, TRAIL detection, and NAV history append can be
pinned end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.portfolio.ib_state import (
    IBSnapshot,
    OpenOrder,
    Position,
    _safe_float,
    append_nav_snapshot,
    fetch_snapshot,
    load_nav_history,
)


# ────────────────────────────────────────────────────────────────
# Test helpers — build a fake ib_insync IB well enough to drive fetch_snapshot
# ────────────────────────────────────────────────────────────────

def _fake_contract(symbol: str) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, secType="STK", exchange="SMART", currency="USD")


def _fake_account_value(tag: str, value: str, currency: str = "USD") -> SimpleNamespace:
    return SimpleNamespace(tag=tag, value=value, currency=currency)


def _fake_portfolio_item(
    symbol: str, position: float, avg_cost: float,
    market_price: float, market_value: float, unrealized_pnl: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        contract=_fake_contract(symbol),
        position=position,
        averageCost=avg_cost,
        marketPrice=market_price,
        marketValue=market_value,
        unrealizedPNL=unrealized_pnl,
    )


def _fake_open_trade(
    symbol: str, action: str, order_type: str, total_qty: int,
    filled: int = 0, remaining: int | None = None,
    lmt_price: float | None = None, aux_price: float | None = None,
    trailing_percent: float | None = None,
    tif: str = "GTC", status: str = "Submitted", order_id: int = 1,
) -> SimpleNamespace:
    if remaining is None:
        remaining = total_qty - filled
    return SimpleNamespace(
        contract=_fake_contract(symbol),
        order=SimpleNamespace(
            action=action,
            orderType=order_type,
            totalQuantity=total_qty,
            lmtPrice=lmt_price,
            auxPrice=aux_price,
            trailingPercent=trailing_percent,
            tif=tif,
            orderId=order_id,
        ),
        orderStatus=SimpleNamespace(
            status=status,
            filled=filled,
            remaining=remaining,
        ),
    )


def _make_fake_ib(
    account: str = "U1234567",
    summary: dict[str, str] | None = None,
    portfolio: list[SimpleNamespace] | None = None,
    positions: list[SimpleNamespace] | None = None,
    open_trades: list[SimpleNamespace] | None = None,
) -> MagicMock:
    summary = summary or {
        "NetLiquidation": "500000",
        "TotalCashValue": "120000",
        "BuyingPower": "1000000",
        "GrossPositionValue": "380000",
        "RealizedPnL": "1200.50",
        "UnrealizedPnL": "3400.00",
    }
    fake_ib = MagicMock()
    fake_ib.managedAccounts.return_value = [account]
    fake_ib.accountSummary.return_value = [
        _fake_account_value(tag, value) for tag, value in summary.items()
    ]
    fake_ib.portfolio.return_value = portfolio or []
    fake_ib.positions.return_value = positions or []
    fake_ib.openTrades.return_value = open_trades or []
    return fake_ib


# ────────────────────────────────────────────────────────────────
# _safe_float
# ────────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [("1234.56", 1234.56), ("", 0.0), (None, 0.0), ("abc", 0.0), (42, 42.0)],
)
def test_safe_float(raw, expected):
    assert _safe_float(raw) == expected


# ────────────────────────────────────────────────────────────────
# fetch_snapshot
# ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_fetch_snapshot_maps_account_summary_tags():
    ib = _make_fake_ib()
    snap = fetch_snapshot(ib)
    assert snap.account == "U1234567"
    assert snap.nav == 500000.0
    assert snap.cash == 120000.0
    assert snap.buying_power == 1000000.0
    assert snap.gross_position_value == 380000.0
    assert snap.realized_pnl_reported == 1200.50
    assert snap.unrealized_pnl_reported == 3400.00
    assert isinstance(snap.timestamp, datetime)


@pytest.mark.unit
def test_fetch_snapshot_ignores_non_usd_currency():
    """A CAD-currency NetLiquidation entry must not overwrite the USD value."""
    ib = _make_fake_ib()
    ib.accountSummary.return_value = [
        _fake_account_value("NetLiquidation", "500000", currency="USD"),
        _fake_account_value("NetLiquidation", "999999", currency="CAD"),
    ]
    snap = fetch_snapshot(ib)
    assert snap.nav == 500000.0


@pytest.mark.unit
def test_fetch_snapshot_uses_portfolio_items_when_present():
    ib = _make_fake_ib(
        portfolio=[
            _fake_portfolio_item("SPY", 100, 400.0, 450.0, 45000.0, 5000.0),
            _fake_portfolio_item("QQQ", 50, 350.0, 340.0, 17000.0, -500.0),
            _fake_portfolio_item("XLK", 0, 200.0, 200.0, 0.0, 0.0),  # skipped
        ]
    )
    snap = fetch_snapshot(ib)
    assert len(snap.positions) == 2
    symbols = {p.ticker for p in snap.positions}
    assert symbols == {"SPY", "QQQ"}
    spy = next(p for p in snap.positions if p.ticker == "SPY")
    assert spy.shares == 100
    assert spy.avg_cost == 400.0
    assert spy.market_price == 450.0
    assert spy.market_value == 45000.0
    assert spy.unrealized_pnl == 5000.0


@pytest.mark.unit
def test_fetch_snapshot_falls_back_to_positions_when_no_portfolio():
    """When portfolio() returns nothing (e.g. before mkt-data subscribes),
    positions() is used and market_price defaults to avg_cost."""
    ib = _make_fake_ib(
        portfolio=[],
        positions=[SimpleNamespace(
            contract=_fake_contract("SPY"), position=100, avgCost=400.0,
        )],
    )
    snap = fetch_snapshot(ib)
    assert len(snap.positions) == 1
    p = snap.positions[0]
    assert p.ticker == "SPY"
    assert p.market_price == 400.0
    assert p.unrealized_pnl == 0.0


@pytest.mark.unit
def test_fetch_snapshot_extracts_open_orders_with_trailing_percent():
    """TRAIL orders must carry their trailingPercent through so the
    Propose panel can identify which tickers are already protected."""
    ib = _make_fake_ib(open_trades=[
        _fake_open_trade("SPY", "SELL", "TRAIL", 100, trailing_percent=10.0, order_id=42),
        _fake_open_trade("QQQ", "BUY", "LMT", 50, lmt_price=340.0, order_id=43),
    ])
    snap = fetch_snapshot(ib)
    assert len(snap.open_orders) == 2

    trail = next(o for o in snap.open_orders if o.order_type == "TRAIL")
    assert trail.ticker == "SPY"
    assert trail.action == "SELL"
    assert trail.trailing_percent == 10.0
    assert trail.tif == "GTC"
    assert trail.order_id == 42

    lmt = next(o for o in snap.open_orders if o.order_type == "LMT")
    assert lmt.limit_price == 340.0
    assert lmt.trailing_percent is None


@pytest.mark.unit
def test_snapshot_open_trails_and_tickers_with_open_trail():
    ib = _make_fake_ib(open_trades=[
        _fake_open_trade("SPY", "SELL", "TRAIL", 100, trailing_percent=10.0),
        _fake_open_trade("QQQ", "SELL", "TRAIL", 50, trailing_percent=8.0),
        # BUY TRAIL — should not count as protective (would be a "buy the dip" order)
        _fake_open_trade("VTI", "BUY", "TRAIL", 30, trailing_percent=5.0),
        _fake_open_trade("QQQ", "BUY", "LMT", 25, lmt_price=340.0),
    ])
    snap = fetch_snapshot(ib)
    trails = snap.open_trails
    assert {o.ticker for o in trails} == {"SPY", "QQQ"}
    assert snap.tickers_with_open_trail == frozenset({"SPY", "QQQ"})


@pytest.mark.unit
def test_snapshot_no_managed_accounts_raises():
    ib = MagicMock()
    ib.managedAccounts.return_value = []
    with pytest.raises(RuntimeError, match="no managed accounts"):
        fetch_snapshot(ib)


@pytest.mark.unit
def test_snapshot_positions_df_shape_and_pct():
    ib = _make_fake_ib(
        portfolio=[
            _fake_portfolio_item("SPY", 100, 400.0, 450.0, 45000.0, 5000.0),
            _fake_portfolio_item("QQQ", 50, 350.0, 340.0, 17000.0, -500.0),
        ]
    )
    df = fetch_snapshot(ib).positions_df()
    assert list(df.columns) == [
        "ticker", "shares", "avg_cost", "market_price",
        "market_value", "unrealized_pnl", "unrealized_pct",
    ]
    spy_row = df[df["ticker"] == "SPY"].iloc[0]
    assert spy_row["unrealized_pct"] == pytest.approx(450.0 / 400.0 - 1.0)


@pytest.mark.unit
def test_snapshot_positions_df_empty_returns_typed_frame():
    ib = _make_fake_ib(portfolio=[], positions=[])
    df = fetch_snapshot(ib).positions_df()
    assert df.empty
    assert "ticker" in df.columns


# ────────────────────────────────────────────────────────────────
# NAV history append / load
# ────────────────────────────────────────────────────────────────

def _snap_with_nav(nav: float, cash: float, account: str = "U123",
                   when: datetime | None = None) -> IBSnapshot:
    return IBSnapshot(
        account=account,
        timestamp=when or datetime.now(timezone.utc),
        nav=nav, cash=cash, buying_power=0.0, gross_position_value=nav - cash,
        realized_pnl_reported=0.0, unrealized_pnl_reported=0.0,
        positions=(), open_orders=(),
    )


@pytest.mark.unit
def test_append_nav_snapshot_creates_file(tmp_path):
    path = tmp_path / "nav.parquet"
    snap = _snap_with_nav(500_000, 100_000)
    hist = append_nav_snapshot(snap, path=path)
    assert path.exists()
    assert len(hist) == 1
    assert hist["nav"].iloc[0] == 500_000
    assert hist["cash"].iloc[0] == 100_000


@pytest.mark.unit
def test_append_nav_snapshot_idempotent_same_day(tmp_path):
    """Two calls on the same day update the row, they do not duplicate."""
    path = tmp_path / "nav.parquet"
    append_nav_snapshot(_snap_with_nav(500_000, 100_000), path=path)
    hist = append_nav_snapshot(_snap_with_nav(510_000, 110_000), path=path)
    assert len(hist) == 1
    assert hist["nav"].iloc[0] == 510_000
    assert hist["cash"].iloc[0] == 110_000


@pytest.mark.unit
def test_append_nav_snapshot_different_days_accumulate(tmp_path):
    path = tmp_path / "nav.parquet"
    day1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    day2 = datetime(2026, 6, 2, tzinfo=timezone.utc)
    append_nav_snapshot(_snap_with_nav(500_000, 100_000, when=day1), path=path)
    hist = append_nav_snapshot(_snap_with_nav(510_000, 105_000, when=day2), path=path)
    assert len(hist) == 2
    assert hist["nav"].tolist() == [500_000, 510_000]


@pytest.mark.unit
def test_append_nav_snapshot_multi_account_independent_rows(tmp_path):
    path = tmp_path / "nav.parquet"
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    append_nav_snapshot(_snap_with_nav(500_000, 100_000, account="U1", when=day), path=path)
    hist = append_nav_snapshot(_snap_with_nav(200_000, 50_000, account="U2", when=day), path=path)
    assert len(hist) == 2
    assert set(hist["account"]) == {"U1", "U2"}


@pytest.mark.unit
def test_load_nav_history_missing_file_returns_empty(tmp_path):
    path = tmp_path / "does_not_exist.parquet"
    hist = load_nav_history(path=path)
    assert hist.empty
    assert list(hist.columns) == ["account", "nav", "cash", "gross_position_value"]


@pytest.mark.unit
def test_load_nav_history_round_trip(tmp_path):
    path = tmp_path / "nav.parquet"
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    append_nav_snapshot(_snap_with_nav(500_000, 100_000, when=day), path=path)
    hist = load_nav_history(path=path)
    assert len(hist) == 1
    assert hist["nav"].iloc[0] == 500_000
