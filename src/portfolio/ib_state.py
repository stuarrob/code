"""Read-only IB Gateway state snapshot for the portfolio applet.

Pulls account value, positions, open orders, and per-position market
value / unrealized PnL from a live IB connection using ``ib_insync``.
The returned :class:`IBSnapshot` is a frozen dataclass — the applet
must not mutate it, and the deterministic pipeline reads from it.

Also appends today's NAV to a local parquet so that repeated applet
runs build up an equity curve without needing an IB Flex Query.

Per ADR-0001 and CLAUDE.md:
- Read-only path only. Any write path (order transmission) belongs in
  a separate module invoked from the guarded "BIG switch".
- No trading maths in this module. Only structured data extraction.
- ``ib_insync`` is mocked at the module boundary in tests.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from ib_insync import IB

logger = logging.getLogger(__name__)


DEFAULT_IB_HOST = "127.0.0.1"
DEFAULT_IB_PORT = 4001
DEFAULT_IB_CLIENT_ID = 30  # Reserved for the applet — avoids collision with
                            # existing scripts using 1/2/5/6/7/15/16/20/22.
DEFAULT_IB_TIMEOUT = 10

DEFAULT_NAV_HISTORY_PATH = (
    Path.home() / "trade_data" / "ETFTrader" / "nav_history.parquet"
)


# ────────────────────────────────────────────────────────────────
# Data classes — the deterministic pipeline's read-only inputs
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Position:
    """A single held position at snapshot time."""

    ticker: str
    shares: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float


@dataclass(frozen=True)
class OpenOrder:
    """A single working order at snapshot time.

    ``trailing_percent`` is populated for TRAIL orders — the applet
    needs this to reason about existing stop protection when generating
    Extend trades (see `_compute_trail_qty` in s7_execute.py).
    """

    ticker: str
    action: str  # "BUY" | "SELL"
    order_type: str  # "MKT" | "LMT" | "STP" | "TRAIL" | ...
    total_qty: int
    filled_qty: int
    remaining_qty: int
    limit_price: Optional[float]
    stop_price: Optional[float]
    trailing_percent: Optional[float]
    tif: str  # "DAY" | "GTC" | ...
    status: str  # "Submitted" | "PreSubmitted" | "Filled" | ...
    order_id: int


@dataclass(frozen=True)
class IBSnapshot:
    """Complete read-only view of the IB account at a moment in time."""

    account: str
    timestamp: datetime
    nav: float
    cash: float
    buying_power: float
    gross_position_value: float
    realized_pnl_reported: float
    unrealized_pnl_reported: float
    positions: tuple[Position, ...]
    open_orders: tuple[OpenOrder, ...]

    @property
    def long_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions if p.shares > 0)

    @property
    def short_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions if p.shares < 0)

    @property
    def open_trails(self) -> tuple[OpenOrder, ...]:
        """Existing TRAIL sells — protecting current holdings.

        The Propose panel uses this to decide whether a top-up on a
        held ticker should place a TRAIL for the full position (no
        existing TRAIL) or only the new shares (TRAIL already present).
        """
        return tuple(
            o for o in self.open_orders
            if o.order_type == "TRAIL" and o.action == "SELL"
        )

    @property
    def tickers_with_open_trail(self) -> frozenset[str]:
        return frozenset(o.ticker for o in self.open_trails)

    def positions_df(self) -> pd.DataFrame:
        """Positions as a display-ready DataFrame."""
        if not self.positions:
            return pd.DataFrame(
                columns=[
                    "ticker", "shares", "avg_cost", "market_price",
                    "market_value", "unrealized_pnl", "unrealized_pct",
                ]
            )
        rows = []
        for p in self.positions:
            rows.append(
                {
                    "ticker": p.ticker,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pct": (
                        (p.market_price / p.avg_cost - 1.0)
                        if p.avg_cost > 0
                        else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    def orders_df(self) -> pd.DataFrame:
        if not self.open_orders:
            return pd.DataFrame(
                columns=[
                    "ticker", "action", "order_type", "total_qty",
                    "remaining_qty", "limit_price", "stop_price",
                    "trailing_percent", "tif", "status",
                ]
            )
        return pd.DataFrame([asdict(o) for o in self.open_orders]).sort_values(
            ["ticker", "order_type"]
        ).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────
# Connection helper
# ────────────────────────────────────────────────────────────────

def connect_read_only(
    host: str = DEFAULT_IB_HOST,
    port: int = DEFAULT_IB_PORT,
    client_id: int = DEFAULT_IB_CLIENT_ID,
    timeout: int = DEFAULT_IB_TIMEOUT,
) -> "IB":
    """Connect to IB Gateway in read-only mode.

    Args:
        host: IB Gateway host.
        port: IB Gateway port (4001 live, 4002 paper).
        client_id: Reserved for the applet — do not reuse from scripts.
        timeout: Seconds to wait for the initial handshake.

    Returns:
        A connected ``ib_insync.IB`` instance.

    Raises:
        RuntimeError: If the connection fails or reports not connected.
    """
    from ib_insync import IB

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — surface any handshake failure
        raise RuntimeError(f"IB connect failed at {host}:{port}: {exc}") from exc
    if not ib.isConnected():
        raise RuntimeError(f"IB reported disconnected after connect() to {host}:{port}")
    return ib


# ────────────────────────────────────────────────────────────────
# Snapshot fetch
# ────────────────────────────────────────────────────────────────

_USD_ACCOUNT_SUMMARY_TAGS = (
    "NetLiquidation",
    "TotalCashValue",
    "BuyingPower",
    "GrossPositionValue",
    "RealizedPnL",
    "UnrealizedPnL",
)


def _safe_float(value: object) -> float:
    """Coerce an IB account-value string to float, tolerating missing/blank."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def fetch_snapshot(ib: "IB") -> IBSnapshot:
    """Pull a full read-only snapshot from a connected IB session.

    Args:
        ib: Connected IB instance (usually from :func:`connect_read_only`).

    Returns:
        A frozen :class:`IBSnapshot`.

    Raises:
        RuntimeError: If ``ib.managedAccounts()`` returns no accounts.
    """
    accounts = ib.managedAccounts()
    if not accounts:
        raise RuntimeError("IB session has no managed accounts")
    account = accounts[0]

    summary_by_tag: dict[str, float] = {tag: 0.0 for tag in _USD_ACCOUNT_SUMMARY_TAGS}
    for av in ib.accountSummary(account):
        if av.currency == "USD" and av.tag in summary_by_tag:
            summary_by_tag[av.tag] = _safe_float(av.value)

    positions = tuple(_extract_positions(ib))
    open_orders = tuple(_extract_open_orders(ib))

    return IBSnapshot(
        account=account,
        timestamp=datetime.now(timezone.utc),
        nav=summary_by_tag["NetLiquidation"],
        cash=summary_by_tag["TotalCashValue"],
        buying_power=summary_by_tag["BuyingPower"],
        gross_position_value=summary_by_tag["GrossPositionValue"],
        realized_pnl_reported=summary_by_tag["RealizedPnL"],
        unrealized_pnl_reported=summary_by_tag["UnrealizedPnL"],
        positions=positions,
        open_orders=open_orders,
    )


def _extract_positions(ib: "IB") -> list[Position]:
    """Use ib.portfolio() so we get marketPrice + unrealized PnL for free.

    Falls back to ib.positions() (no market data) if portfolio() is empty.
    """
    items = ib.portfolio()
    if items:
        out: list[Position] = []
        for it in items:
            shares = float(it.position)
            if shares == 0:
                continue
            out.append(
                Position(
                    ticker=it.contract.symbol,
                    shares=shares,
                    avg_cost=float(it.averageCost),
                    market_price=float(it.marketPrice),
                    market_value=float(it.marketValue),
                    unrealized_pnl=float(it.unrealizedPNL),
                )
            )
        return out

    out: list[Position] = []
    for p in ib.positions():
        shares = float(p.position)
        if shares == 0:
            continue
        avg_cost = float(p.avgCost)
        out.append(
            Position(
                ticker=p.contract.symbol,
                shares=shares,
                avg_cost=avg_cost,
                market_price=avg_cost,  # No market data source available
                market_value=shares * avg_cost,
                unrealized_pnl=0.0,
            )
        )
    return out


def _extract_open_orders(ib: "IB") -> list[OpenOrder]:
    out: list[OpenOrder] = []
    for t in ib.openTrades():
        order = t.order
        status = t.orderStatus
        out.append(
            OpenOrder(
                ticker=t.contract.symbol,
                action=order.action,
                order_type=order.orderType,
                total_qty=int(order.totalQuantity),
                filled_qty=int(status.filled),
                remaining_qty=int(status.remaining),
                limit_price=getattr(order, "lmtPrice", None) or None,
                stop_price=getattr(order, "auxPrice", None) or None,
                trailing_percent=getattr(order, "trailingPercent", None) or None,
                tif=order.tif or "DAY",
                status=status.status or "Unknown",
                order_id=int(order.orderId),
            )
        )
    return out


# ────────────────────────────────────────────────────────────────
# NAV history — local equity curve
# ────────────────────────────────────────────────────────────────

_NAV_HISTORY_COLUMNS = ["date", "account", "nav", "cash", "gross_position_value"]


def append_nav_snapshot(
    snapshot: IBSnapshot,
    path: Path = DEFAULT_NAV_HISTORY_PATH,
) -> pd.DataFrame:
    """Persist today's NAV/cash to the local NAV history parquet.

    Idempotent by (date, account): running the applet twice on the same
    day updates the row with the newer snapshot rather than appending
    a duplicate.

    Args:
        snapshot: A fresh :class:`IBSnapshot`.
        path: Parquet path (default lives under ``~/trade_data/ETFTrader``).

    Returns:
        The full NAV history DataFrame after the update, indexed by date.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_date = snapshot.timestamp.astimezone(timezone.utc).date()
    new_row = {
        "date": pd.Timestamp(snapshot_date),
        "account": snapshot.account,
        "nav": snapshot.nav,
        "cash": snapshot.cash,
        "gross_position_value": snapshot.gross_position_value,
    }

    if path.exists():
        existing = pd.read_parquet(path)
    else:
        existing = pd.DataFrame(columns=_NAV_HISTORY_COLUMNS)

    mask = (existing["date"] == new_row["date"]) & (existing["account"] == new_row["account"])
    if mask.any():
        # Replace today's row (same account, same day).
        for key, value in new_row.items():
            existing.loc[mask, key] = value
        combined = existing
    else:
        combined = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined.set_index("date")


def load_nav_history(path: Path = DEFAULT_NAV_HISTORY_PATH) -> pd.DataFrame:
    """Read the NAV history parquet. Returns an empty typed frame if absent."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=_NAV_HISTORY_COLUMNS).set_index("date")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")
    return df
