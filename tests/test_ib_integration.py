"""
Integration tests for IB Gateway.

These tests require a running IB Gateway on port 4002 (paper trading).
They are marked with pytest.mark.integration and will be skipped
when IB Gateway is not available.

Run with: pytest tests/test_ib_integration.py -m integration
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
async def ib_connection():
    """Connect to IB Gateway for each test, skip if unavailable."""
    from backend.app.services.ib_connection import IBConnectionManager

    manager = IBConnectionManager()

    # Temporarily enable IB for integration tests
    import backend.app.services.ib_connection as mod

    original_enabled = mod.settings.IB_ENABLED

    try:
        # Patch IB_ENABLED to True for integration tests
        object.__setattr__(mod.settings, "IB_ENABLED", True)
        connected = await manager.connect()
        if not connected:
            pytest.skip("IB Gateway not available on port 4002")
        yield manager
    finally:
        await manager.disconnect()
        object.__setattr__(mod.settings, "IB_ENABLED", original_enabled)


@pytest.mark.asyncio
async def test_real_quote(ib_connection):
    """Fetch a real quote from IB Gateway."""
    from backend.app.services.ib_market_data import IBMarketDataService

    service = IBMarketDataService()
    quote = await service.get_realtime_quote("SPY")
    assert quote is not None
    assert quote["source"] == "ib_gateway"
    assert quote["last"] is not None


@pytest.mark.asyncio
async def test_real_historical_bars(ib_connection):
    """Fetch real historical bars from IB Gateway."""
    from backend.app.services.ib_market_data import IBMarketDataService

    service = IBMarketDataService()
    bars = await service.get_historical_bars("SPY", duration="5 D", bar_size="1 day")
    assert len(bars) > 0
    assert "close" in bars[0]


@pytest.mark.asyncio
async def test_real_account_summary(ib_connection):
    """Fetch real account summary from IB Gateway."""
    from backend.app.services.ib_account import IBAccountService

    service = IBAccountService()
    summary = await service.get_account_summary()
    assert summary is not None
    assert summary["source"] == "ib_gateway"


@pytest.mark.asyncio
async def test_real_positions(ib_connection):
    """Fetch real positions from IB Gateway."""
    from backend.app.services.ib_account import IBAccountService

    service = IBAccountService()
    positions = await service.get_positions()
    # positions may be empty if no positions held, but should not error
    assert isinstance(positions, list)
