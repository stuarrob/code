"""Portfolio model."""

from sqlalchemy import Column, Integer, String, Numeric, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class PortfolioStatus(str, enum.Enum):
    """Portfolio status enum."""
    ACTIVE = "active"
    CLOSED = "closed"
    PAUSED = "paused"


class Portfolio(Base):
    """
    Portfolio model - represents a trading portfolio.

    Attributes:
        id: Primary key
        name: Portfolio name
        optimizer_type: Optimizer used (mvo, rank_based, minvar, simple)
        num_positions: Target number of positions
        initial_capital: Starting capital
        current_value: Current portfolio value (updated daily)
        is_paper_trading: True if paper trading, False if real
        status: active, closed, or paused
        created_at: Creation timestamp
        updated_at: Last update timestamp

    Relationships:
        positions: Current positions in portfolio
        trades: All trades executed
        values: Daily portfolio values
        rebalance_events: Rebalancing history
    """

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    optimizer_type = Column(String(50), nullable=False, default="mvo")
    num_positions = Column(Integer, nullable=False, default=20)
    initial_capital = Column(Numeric(15, 2), nullable=False)
    current_value = Column(Numeric(15, 2))
    cash = Column(Numeric(15, 2), nullable=False)  # Available cash for trading
    is_paper_trading = Column(Boolean, default=True, nullable=False)
    status = Column(
        Enum(PortfolioStatus),
        default=PortfolioStatus.ACTIVE,
        nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="portfolio", cascade="all, delete-orphan")
    values = relationship("PortfolioValue", back_populates="portfolio", cascade="all, delete-orphan")
    rebalance_events = relationship("RebalanceEvent", back_populates="portfolio", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.name}', value={self.current_value})>"
