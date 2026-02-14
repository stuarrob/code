"""Interactive Brokers market data service."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ib_insync import Stock

from app.services.ib_connection import ib_manager
from app.services.price_service import price_service

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    """Convert IB value to float, returning None for NaN."""
    if val is None or val != val:  # NaN != NaN
        return None
    return float(val)


def _safe_int(val) -> Optional[int]:
    """Convert IB value to int, returning None for NaN."""
    if val is None or val != val:
        return None
    return int(val)


class IBMarketDataService:
    """
    Service for fetching market data from Interactive Brokers.

    Falls back to PriceService (parquet files) when IB is not connected.
    """

    async def get_realtime_quote(self, ticker: str) -> Optional[dict]:
        """
        Get real-time snapshot quote for a single ticker.

        Returns dict with bid, ask, last, volume, etc. or None if unavailable.
        """
        ib = ib_manager.ib
        if not ib:
            return self._fallback_quote(ticker)

        try:
            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)

            ticker_data = ib.reqMktData(contract, snapshot=True)
            await ib.sleep(2)

            result = {
                "ticker": ticker,
                "source": "ib_gateway",
                "timestamp": datetime.now().isoformat(),
                "bid": _safe_float(ticker_data.bid),
                "ask": _safe_float(ticker_data.ask),
                "last": _safe_float(ticker_data.last),
                "volume": _safe_int(ticker_data.volume),
                "high": _safe_float(ticker_data.high),
                "low": _safe_float(ticker_data.low),
                "close": _safe_float(ticker_data.close),
            }

            ib.cancelMktData(contract)
            return result

        except Exception as e:
            logger.error(f"Error fetching IB quote for {ticker}: {e}")
            return self._fallback_quote(ticker)

    async def get_realtime_quotes(self, tickers: List[str]) -> Dict[str, dict]:
        """
        Get real-time snapshot quotes for multiple tickers.

        Returns dict mapping ticker -> quote data.
        """
        ib = ib_manager.ib
        if not ib:
            return self._fallback_quotes(tickers)

        try:
            contracts = [Stock(t, "SMART", "USD") for t in tickers]
            ib.qualifyContracts(*contracts)

            ticker_data_list = []
            for contract in contracts:
                td = ib.reqMktData(contract, snapshot=True)
                ticker_data_list.append((contract, td))

            await ib.sleep(3)

            results = {}
            for contract, td in ticker_data_list:
                symbol = contract.symbol
                results[symbol] = {
                    "ticker": symbol,
                    "source": "ib_gateway",
                    "timestamp": datetime.now().isoformat(),
                    "bid": _safe_float(td.bid),
                    "ask": _safe_float(td.ask),
                    "last": _safe_float(td.last),
                    "volume": _safe_int(td.volume),
                    "high": _safe_float(td.high),
                    "low": _safe_float(td.low),
                    "close": _safe_float(td.close),
                }
                ib.cancelMktData(contract)

            return results

        except Exception as e:
            logger.error(f"Error fetching IB quotes: {e}")
            return self._fallback_quotes(tickers)

    async def get_historical_bars(
        self,
        ticker: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST",
    ) -> List[dict]:
        """
        Get historical price bars from IB.

        Args:
            ticker: ETF ticker symbol.
            duration: How far back (e.g., '1 Y', '6 M', '30 D').
            bar_size: Bar granularity (e.g., '1 day', '1 hour', '5 mins').
            what_to_show: Price type ('ADJUSTED_LAST', 'TRADES', 'MIDPOINT').

        Returns:
            List of bar dicts with date, open, high, low, close, volume.
        """
        ib = ib_manager.ib
        if not ib:
            logger.warning(f"IB not connected, cannot fetch historical bars for {ticker}")
            return []

        try:
            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)

            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1,
            )

            return [
                {
                    "date": (
                        bar.date.isoformat()
                        if hasattr(bar.date, "isoformat")
                        else str(bar.date)
                    ),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "average": _safe_float(getattr(bar, "average", None)),
                    "bar_count": _safe_int(getattr(bar, "barCount", None)),
                }
                for bar in bars
            ]

        except Exception as e:
            logger.error(f"Error fetching IB historical bars for {ticker}: {e}")
            return []

    def _fallback_quote(self, ticker: str) -> Optional[dict]:
        """Fallback to PriceService when IB is not available."""
        price = price_service.get_price(ticker)
        if price is None:
            return None
        return {
            "ticker": ticker,
            "source": "parquet_fallback",
            "timestamp": datetime.now().isoformat(),
            "last": float(price),
            "bid": None,
            "ask": None,
            "volume": None,
            "high": None,
            "low": None,
            "close": float(price),
        }

    def _fallback_quotes(self, tickers: List[str]) -> Dict[str, dict]:
        """Fallback to PriceService for multiple tickers."""
        results = {}
        for ticker in tickers:
            quote = self._fallback_quote(ticker)
            if quote:
                results[ticker] = quote
        return results


# Global singleton instance
ib_market_data = IBMarketDataService()
