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

import asyncio
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
    """A single held position at snapshot time.

    ``daily_pnl`` comes from ``ib.reqPnLSingle`` (a streaming subscription
    the snapshot briefly opens then cancels). It is ``float('nan')`` if
    IB has not yet reported a value for the position within the polling
    window.
    """

    ticker: str
    conid: int
    shares: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    daily_pnl: float


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
    """Complete read-only view of the IB account at a moment in time.

    ``daily_pnl`` is from ``ib.reqPnL`` — the same P&L figure the
    Client Portal shows in its top strip. All margin / liquidity
    fields come from ``accountSummary``. Missing values default to
    ``0.0``; presence is signalled by inspecting ``timestamp``.
    """

    account: str
    timestamp: datetime
    nav: float
    cash: float
    buying_power: float
    gross_position_value: float
    realized_pnl_reported: float
    unrealized_pnl_reported: float
    daily_pnl: float
    excess_liquidity: float
    available_funds: float
    maint_margin: float
    init_margin: float
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
        columns = [
            "ticker", "shares", "avg_cost", "market_price",
            "market_value", "daily_pnl", "unrealized_pnl", "unrealized_pct",
        ]
        if not self.positions:
            return pd.DataFrame(columns=columns)
        rows = []
        for p in self.positions:
            rows.append(
                {
                    "ticker": p.ticker,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price,
                    "market_value": p.market_value,
                    "daily_pnl": p.daily_pnl,
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

def _ensure_event_loop() -> None:
    """Give the current thread an asyncio event loop.

    ``ib_insync`` is built on asyncio; Streamlit runs each user session in
    a worker thread that has no default event loop, which raises
    ``RuntimeError: There is no current event loop in thread ...`` on the
    very first ``IB()`` call. Same story for Jupyter, cron under nohup,
    and any non-main-thread caller. Creating one and installing it as the
    thread's default is safe and idempotent.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


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
    _ensure_event_loop()
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
    "ExcessLiquidity",
    "AvailableFunds",
    "MaintMarginReq",
    "InitMarginReq",
)

_PNL_SETTLE_SECONDS = 2  # How long to wait for streaming reqPnL updates.
_IB_UNSET_SENTINEL = -1.7976931348623157e+308  # IB's "no value" magic number.


def _safe_float(value: object) -> float:
    """Coerce an IB account-value string to float, tolerating missing/blank."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _safe_pnl(value: object) -> float:
    """Coerce a streaming-PnL value, treating IB's sentinel + NaN as NaN.

    ib_insync surfaces "no value yet" as either NaN, None, or IB's magic
    -1.797e308. The applet treats all three as NaN so the UI can render
    a blank rather than a spurious 0.
    """
    if value is None:
        return float("nan")
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if f != f:  # NaN
        return float("nan")
    if f == _IB_UNSET_SENTINEL:
        return float("nan")
    return f


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

    positions_raw = _extract_positions(ib)
    daily_pnl_by_conid, account_daily_pnl = _fetch_daily_pnl(
        ib, account, positions_raw
    )
    positions = tuple(
        Position(**{**p, "daily_pnl": daily_pnl_by_conid.get(p["conid"], float("nan"))})
        for p in positions_raw
    )
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
        daily_pnl=account_daily_pnl,
        excess_liquidity=summary_by_tag["ExcessLiquidity"],
        available_funds=summary_by_tag["AvailableFunds"],
        maint_margin=summary_by_tag["MaintMarginReq"],
        init_margin=summary_by_tag["InitMarginReq"],
        positions=positions,
        open_orders=open_orders,
    )


def _extract_positions(ib: "IB") -> list[dict]:
    """Return raw dicts (missing ``daily_pnl``) so the caller can attach
    per-position PnL from the streaming ``reqPnLSingle`` call.

    Uses ``ib.portfolio()`` so we get marketPrice + unrealized PnL for
    free; falls back to ``ib.positions()`` (no market data) if empty.
    """
    items = ib.portfolio()
    if items:
        out: list[dict] = []
        for it in items:
            shares = float(it.position)
            if shares == 0:
                continue
            out.append(
                {
                    "ticker": it.contract.symbol,
                    "conid": int(getattr(it.contract, "conId", 0) or 0),
                    "shares": shares,
                    "avg_cost": float(it.averageCost),
                    "market_price": float(it.marketPrice),
                    "market_value": float(it.marketValue),
                    "unrealized_pnl": float(it.unrealizedPNL),
                }
            )
        return out

    out = []
    for p in ib.positions():
        shares = float(p.position)
        if shares == 0:
            continue
        avg_cost = float(p.avgCost)
        out.append(
            {
                "ticker": p.contract.symbol,
                "conid": int(getattr(p.contract, "conId", 0) or 0),
                "shares": shares,
                "avg_cost": avg_cost,
                "market_price": avg_cost,  # No market data source available
                "market_value": shares * avg_cost,
                "unrealized_pnl": 0.0,
            }
        )
    return out


def _fetch_daily_pnl(
    ib: "IB",
    account: str,
    positions_raw: list[dict],
) -> tuple[dict[int, float], float]:
    """Batch-subscribe to reqPnL + reqPnLSingle, wait once, then cancel.

    Returns ``(daily_pnl_by_conid, account_daily_pnl)``.

    Values that IB hasn't populated within ``_PNL_SETTLE_SECONDS`` are
    returned as ``float('nan')`` so the UI can render a blank cell.

    Failures at any layer (rate limits, network, IB not exposing PnL for
    this account type) degrade gracefully to all-NaN. Snapshot is still
    returned; the applet doesn't crash because PnL isn't available.
    """
    account_pnl = None
    single_subs: list[tuple[int, object]] = []

    try:
        account_pnl = ib.reqPnL(account)
    except Exception:  # noqa: BLE001
        logger.warning("reqPnL failed for %s — daily P&L will be NaN", account)

    for p in positions_raw:
        conid = p.get("conid") or 0
        if conid <= 0:
            continue
        try:
            sub = ib.reqPnLSingle(account, "", conid)
            single_subs.append((conid, sub))
        except Exception:  # noqa: BLE001
            logger.warning(
                "reqPnLSingle failed for %s (conid=%s)", p.get("ticker"), conid
            )

    if account_pnl is not None or single_subs:
        try:
            ib.sleep(_PNL_SETTLE_SECONDS)
        except Exception:  # noqa: BLE001
            logger.warning("ib.sleep interrupted while waiting for PnL")

    account_daily = float("nan")
    if account_pnl is not None:
        account_daily = _safe_pnl(getattr(account_pnl, "dailyPnL", None))
        try:
            ib.cancelPnL(account)
        except Exception:  # noqa: BLE001
            pass

    daily_by_conid: dict[int, float] = {}
    for conid, sub in single_subs:
        daily_by_conid[conid] = _safe_pnl(getattr(sub, "dailyPnL", None))
        try:
            ib.cancelPnLSingle(account, "", conid)
        except Exception:  # noqa: BLE001
            pass

    return daily_by_conid, account_daily


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
