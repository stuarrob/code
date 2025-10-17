"""Rebalance event model."""

from sqlalchemy import Column, Integer, String, Numeric, Date, DateTime, ForeignKey, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class RebalanceStatus(str, enum.Enum):
    """Rebalance event status."""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class RebalanceEvent(Base):
    """
    Rebalance event model - tracks portfolio rebalancing events.

    Attributes:
        id: Primary key
        portfolio_id: Foreign key to portfolios table
        date: Rebalance date
        reason: Reason for rebalancing (drift_threshold, stop_loss, scheduled)
        max_drift: Maximum position drift that triggered rebalance
        num_trades: Number of trades executed
        total_cost: Total transaction costs
        status: pending, executed, cancelled
        created_at: Event creation timestamp

    Relationships:
        portfolio: Parent portfolio
    """

    __tablename__ = "rebalance_events"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    reason = Column(String(100))
    max_drift = Column(Numeric(6, 4))
    num_trades = Column(Integer)
    total_cost = Column(Numeric(10, 2))
    status = Column(Enum(RebalanceStatus), default=RebalanceStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    portfolio = relationship("Portfolio", back_populates="rebalance_events")

    def __repr__(self):
        return f"<RebalanceEvent(date={self.date}, reason='{self.reason}', trades={self.num_trades})>"
