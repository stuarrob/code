"""Portfolio value model."""

from sqlalchemy import Column, Integer, Numeric, Date, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from app.core.database import Base


class PortfolioValue(Base):
    """
    Portfolio value model - daily portfolio valuation.

    Attributes:
        id: Primary key
        portfolio_id: Foreign key to portfolios table
        date: Valuation date
        total_value: Total portfolio value
        cash: Cash balance
        invested_value: Value of invested positions
        daily_return: Daily return (as decimal)
        cumulative_return: Cumulative return since inception

    Relationships:
        portfolio: Parent portfolio
    """

    __tablename__ = "portfolio_values"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    total_value = Column(Numeric(15, 2), nullable=False)
    cash = Column(Numeric(15, 2), nullable=False)
    invested_value = Column(Numeric(15, 2), nullable=False)
    daily_return = Column(Numeric(8, 6))
    cumulative_return = Column(Numeric(10, 6))

    # Ensure unique date per portfolio
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'date', name='_portfolio_date_uc'),
    )

    # Relationships
    portfolio = relationship("Portfolio", back_populates="values")

    def __repr__(self):
        return f"<PortfolioValue(date={self.date}, value={self.total_value})>"
