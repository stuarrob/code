"""Trade model."""

from sqlalchemy import Column, Integer, String, Numeric, DateTime, ForeignKey, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class TradeSide(str, enum.Enum):
    """Trade side enum."""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, enum.Enum):
    """Trade status enum."""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Trade(Base):
    """
    Trade model - represents a buy or sell transaction.

    Attributes:
        id: Primary key
        portfolio_id: Foreign key to portfolios table
        ticker: ETF ticker symbol
        side: 'buy' or 'sell'
        quantity: Number of shares
        price: Execution price per share
        total_value: Total trade value (quantity * price)
        commission: Commission paid
        slippage: Slippage cost
        status: pending, executed, failed, cancelled
        executed_at: Execution timestamp
        created_at: Trade creation timestamp

    Relationships:
        portfolio: Parent portfolio
    """

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    side = Column(Enum(TradeSide), nullable=False)
    quantity = Column(Numeric(12, 4), nullable=False)
    price = Column(Numeric(10, 2), nullable=False)
    total_value = Column(Numeric(15, 2), nullable=False)
    commission = Column(Numeric(10, 2), default=0)
    slippage = Column(Numeric(10, 2), default=0)
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    executed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    portfolio = relationship("Portfolio", back_populates="trades")

    def __repr__(self):
        return f"<Trade({self.side} {self.quantity} {self.ticker} @ {self.price})>"
