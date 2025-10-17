"""ETF price model."""

from sqlalchemy import Column, Integer, String, Numeric, Date, BigInteger, UniqueConstraint

from app.core.database import Base


class ETFPrice(Base):
    """
    ETF price model - daily OHLCV data.

    Attributes:
        id: Primary key
        ticker: ETF ticker symbol
        date: Price date
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        adj_close: Adjusted closing price (for dividends/splits)
        volume: Trading volume
    """

    __tablename__ = "etf_prices"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Numeric(10, 2))
    high = Column(Numeric(10, 2))
    low = Column(Numeric(10, 2))
    close = Column(Numeric(10, 2))
    adj_close = Column(Numeric(10, 2))
    volume = Column(BigInteger)

    # Ensure unique ticker/date combination
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='_ticker_date_price_uc'),
    )

    def __repr__(self):
        return f"<ETFPrice(ticker='{self.ticker}', date={self.date}, close={self.close})>"
