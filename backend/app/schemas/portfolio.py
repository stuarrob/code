"""Portfolio schemas."""

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from decimal import Decimal
from typing import Optional


class PortfolioBase(BaseModel):
    """Base portfolio schema."""
    name: str = Field(..., min_length=1, max_length=100, description="Portfolio name")
    optimizer_type: str = Field(default="mvo", description="Optimizer type (mvo, rank_based, minvar, simple)")
    num_positions: int = Field(default=20, ge=1, le=100, description="Target number of positions")
    is_paper_trading: bool = Field(default=True, description="Paper trading mode")


class PortfolioCreate(PortfolioBase):
    """Schema for creating a portfolio."""
    initial_capital: Decimal = Field(..., gt=0, description="Initial capital")


class PortfolioUpdate(BaseModel):
    """Schema for updating a portfolio."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    status: Optional[str] = None
    current_value: Optional[Decimal] = None

    model_config = ConfigDict(from_attributes=True)


class PortfolioResponse(PortfolioBase):
    """Schema for portfolio response."""
    id: int
    initial_capital: Decimal
    current_value: Optional[Decimal]
    cash: Decimal
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
