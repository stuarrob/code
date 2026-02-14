"""Pydantic schemas for Interactive Brokers API responses."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class IBConnectionStatus(BaseModel):
    """IB Gateway connection status."""

    enabled: bool
    connected: bool
    host: str
    port: int
    client_id: int
    readonly: bool
    accounts: List[str]


class IBQuote(BaseModel):
    """Real-time quote from IB."""

    ticker: str
    source: str  # 'ib_gateway' or 'parquet_fallback'
    timestamp: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None


class IBHistoricalBar(BaseModel):
    """Historical price bar from IB."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    average: Optional[float] = None
    bar_count: Optional[int] = None


class IBAccountSummary(BaseModel):
    """IB account summary."""

    source: str
    timestamp: str
    net_liquidation: Optional[float] = None
    total_cash: Optional[float] = None
    gross_position_value: Optional[float] = None
    available_funds: Optional[float] = None
    buying_power: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    values: Dict[str, dict] = {}


class IBPosition(BaseModel):
    """IB account position."""

    account: str
    ticker: str
    security_type: str
    exchange: str
    currency: str
    shares: float
    avg_cost: float
    market_value: float


class IBPnL(BaseModel):
    """Portfolio-level P&L from IB."""

    source: str
    timestamp: str
    account: str
    daily_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None


class IBPositionPnL(BaseModel):
    """Per-position P&L from IB."""

    ticker: str
    shares: float
    avg_cost: float
    daily_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    market_value: Optional[float] = None
