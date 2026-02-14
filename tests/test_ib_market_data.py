"""Tests for IB market data service."""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock


class TestIBMarketDataService:
    """Tests for IBMarketDataService."""

    @pytest.mark.asyncio
    @patch("backend.app.services.ib_market_data.ib_manager")
    @patch("backend.app.services.ib_market_data.price_service")
    async def test_fallback_when_disconnected(self, mock_price_svc, mock_ib_mgr):
        """Falls back to price_service when IB is not connected."""
        mock_ib_mgr.ib = None
        mock_price_svc.get_price.return_value = Decimal("450.50")

        from backend.app.services.ib_market_data import IBMarketDataService

        service = IBMarketDataService()
        quote = await service.get_realtime_quote("SPY")

        assert quote is not None
        assert quote["source"] == "parquet_fallback"
        assert quote["last"] == 450.50
        assert quote["ticker"] == "SPY"

    @pytest.mark.asyncio
    @patch("backend.app.services.ib_market_data.ib_manager")
    @patch("backend.app.services.ib_market_data.price_service")
    async def test_returns_none_when_no_data(self, mock_price_svc, mock_ib_mgr):
        """Returns None when no data is available from any source."""
        mock_ib_mgr.ib = None
        mock_price_svc.get_price.return_value = None

        from backend.app.services.ib_market_data import IBMarketDataService

        service = IBMarketDataService()
        quote = await service.get_realtime_quote("INVALID")
        assert quote is None

    @pytest.mark.asyncio
    @patch("backend.app.services.ib_market_data.ib_manager")
    @patch("backend.app.services.ib_market_data.price_service")
    async def test_fallback_quotes_multiple(self, mock_price_svc, mock_ib_mgr):
        """Falls back to price_service for multiple tickers."""
        mock_ib_mgr.ib = None
        mock_price_svc.get_price.side_effect = lambda t: {
            "SPY": Decimal("450.50"),
            "QQQ": Decimal("380.25"),
        }.get(t)

        from backend.app.services.ib_market_data import IBMarketDataService

        service = IBMarketDataService()
        quotes = await service.get_realtime_quotes(["SPY", "QQQ", "INVALID"])

        assert "SPY" in quotes
        assert "QQQ" in quotes
        assert "INVALID" not in quotes
        assert quotes["SPY"]["source"] == "parquet_fallback"

    @pytest.mark.asyncio
    @patch("backend.app.services.ib_market_data.ib_manager")
    async def test_historical_bars_returns_empty_when_disconnected(self, mock_ib_mgr):
        """Returns empty list when IB is not connected."""
        mock_ib_mgr.ib = None

        from backend.app.services.ib_market_data import IBMarketDataService

        service = IBMarketDataService()
        bars = await service.get_historical_bars("SPY")
        assert bars == []


class TestSafeFloat:
    """Tests for the _safe_float helper."""

    def test_normal_value(self):
        from backend.app.services.ib_market_data import _safe_float

        assert _safe_float(42.5) == 42.5

    def test_nan_value(self):
        from backend.app.services.ib_market_data import _safe_float

        assert _safe_float(float("nan")) is None

    def test_none_value(self):
        from backend.app.services.ib_market_data import _safe_float

        assert _safe_float(None) is None
