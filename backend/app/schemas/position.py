"""Position schemas."""

from pydantic import BaseModel, ConfigDict
from datetime import datetime
from decimal import Decimal
from typing import Optional


class PositionBase(BaseModel):
    """Base position schema."""
    ticker: str
    target_weight: Optional[Decimal] = None
    current_weight: Optional[Decimal] = None
    shares: Optional[Decimal] = None


class PositionResponse(PositionBase):
    """Schema for position response."""
    id: int
    portfolio_id: int
    entry_price: Optional[Decimal]
    current_price: Optional[Decimal]
    unrealized_pnl: Optional[Decimal]
    stop_loss_price: Optional[Decimal]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DriftResponse(BaseModel):
    """Schema for drift calculation response."""
    max_drift: float
    total_drift: float
    needs_rebalancing: bool
    threshold: float
    positions: list[dict]


class RebalanceRecommendation(BaseModel):
    """Schema for rebalancing recommendations."""
    ticker: str
    side: str  # 'buy' or 'sell'
    current_weight: float
    target_weight: float
    drift: float
    quantity: float
    estimated_cost: float
