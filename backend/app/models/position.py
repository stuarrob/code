"""Position model."""

from sqlalchemy import Column, Integer, String, Numeric, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Position(Base):
    """
    Position model - represents a position in a portfolio.

    Attributes:
        id: Primary key
        portfolio_id: Foreign key to portfolios table
        ticker: ETF ticker symbol
        target_weight: Target allocation weight (0.05 = 5%)
        current_weight: Current allocation weight
        shares: Number of shares held
        entry_price: Average entry price per share
        current_price: Latest market price
        unrealized_pnl: Unrealized profit/loss
        stop_loss_price: Stop-loss trigger price
        created_at: Position creation timestamp
        updated_at: Last update timestamp

    Relationships:
        portfolio: Parent portfolio
    """

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    target_weight = Column(Numeric(6, 4))  # 0.0500 = 5%
    current_weight = Column(Numeric(6, 4))
    shares = Column(Numeric(12, 4))
    entry_price = Column(Numeric(10, 2))
    current_price = Column(Numeric(10, 2))
    unrealized_pnl = Column(Numeric(12, 2))
    stop_loss_price = Column(Numeric(10, 2))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Ensure unique ticker per portfolio
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'ticker', name='_portfolio_ticker_uc'),
    )

    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")

    def __repr__(self):
        return f"<Position(ticker='{self.ticker}', shares={self.shares}, pnl={self.unrealized_pnl})>"
