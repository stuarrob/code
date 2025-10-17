"""Trade schemas."""

from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from decimal import Decimal
from typing import Optional


class TradeCreate(BaseModel):
    """Schema for creating a trade."""
    portfolio_id: int
    ticker: str = Field(..., min_length=1, max_length=10)
    side: str = Field(..., pattern="^(buy|sell)$")
    quantity: Optional[Decimal] = Field(None, gt=0)  # Shares to buy/sell
    dollar_amount: Optional[Decimal] = Field(None, gt=0)  # Alternative: specify dollar amount
    price: Optional[Decimal] = Field(None, gt=0)  # Optional: will fetch real price if not provided


class TradeResponse(BaseModel):
    """Schema for trade response."""
    id: int
    portfolio_id: int
    ticker: str
    side: str
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    commission: Decimal
    slippage: Decimal
    status: str
    executed_at: Optional[datetime]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
