"""Factor score model."""

from sqlalchemy import Column, Integer, String, Numeric, Date, UniqueConstraint

from app.core.database import Base


class FactorScore(Base):
    """
    Factor score model - daily factor scores for ETFs.

    Attributes:
        id: Primary key
        ticker: ETF ticker symbol
        date: Score calculation date
        momentum: Momentum factor score
        quality: Quality factor score
        value: Value factor score
        volatility: Volatility factor score
        composite: Composite (integrated) score
    """

    __tablename__ = "factor_scores"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    momentum = Column(Numeric(10, 6))
    quality = Column(Numeric(10, 6))
    value = Column(Numeric(10, 6))
    volatility = Column(Numeric(10, 6))
    composite = Column(Numeric(10, 6))

    # Ensure unique ticker/date combination
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='_ticker_date_uc'),
    )

    def __repr__(self):
        return f"<FactorScore(ticker='{self.ticker}', date={self.date}, composite={self.composite})>"
