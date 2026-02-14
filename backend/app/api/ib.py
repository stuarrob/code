"""Interactive Brokers API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List

from app.services.ib_connection import ib_manager
from app.services.ib_market_data import ib_market_data
from app.services.ib_account import ib_account
from app.schemas.ib import (
    IBConnectionStatus,
    IBQuote,
    IBHistoricalBar,
    IBAccountSummary,
    IBPosition,
    IBPnL,
    IBPositionPnL,
)

router = APIRouter()


# --- Connection Status ---


@router.get("/status", response_model=IBConnectionStatus)
async def get_ib_status():
    """Get IB Gateway connection status."""
    return ib_manager.get_status()


@router.post("/connect")
async def connect_ib():
    """Manually trigger IB Gateway connection."""
    success = await ib_manager.connect()
    if success:
        return {"status": "connected", "detail": "Connected to IB Gateway"}
    return {"status": "failed", "detail": "Could not connect to IB Gateway"}


@router.post("/disconnect")
async def disconnect_ib():
    """Manually disconnect from IB Gateway."""
    await ib_manager.disconnect()
    return {"status": "disconnected", "detail": "Disconnected from IB Gateway"}


# --- Market Data ---


@router.get("/quote/{ticker}", response_model=IBQuote)
async def get_quote(ticker: str):
    """
    Get real-time quote for a single ticker.

    Falls back to stored price data when IB is not connected.
    """
    quote = await ib_market_data.get_realtime_quote(ticker.upper())
    if not quote:
        raise HTTPException(
            status_code=404, detail=f"No price data available for {ticker}"
        )
    return quote


@router.get("/quotes", response_model=Dict[str, IBQuote])
async def get_quotes(
    tickers: str = Query(
        ..., description="Comma-separated list of tickers (e.g., SPY,QQQ,IWM)"
    ),
):
    """
    Get real-time quotes for multiple tickers.

    Falls back to stored price data when IB is not connected.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(ticker_list) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 tickers per request")

    return await ib_market_data.get_realtime_quotes(ticker_list)


@router.get("/history/{ticker}", response_model=List[IBHistoricalBar])
async def get_historical_bars(
    ticker: str,
    duration: str = Query(
        "1 Y", description="Duration string (e.g., '1 Y', '6 M', '30 D')"
    ),
    bar_size: str = Query(
        "1 day", description="Bar size (e.g., '1 day', '1 hour', '5 mins')"
    ),
    what_to_show: str = Query("ADJUSTED_LAST", description="Price type"),
):
    """
    Get historical price bars from IB.

    Requires active IB connection (no fallback for historical data).
    """
    if not ib_manager.is_connected:
        raise HTTPException(
            status_code=503,
            detail="IB Gateway not connected. Historical data requires live connection.",
        )

    bars = await ib_market_data.get_historical_bars(
        ticker.upper(), duration, bar_size, what_to_show
    )

    if not bars:
        raise HTTPException(
            status_code=404, detail=f"No historical data for {ticker}"
        )

    return bars


# --- Account Data ---


@router.get("/account/summary", response_model=IBAccountSummary)
async def get_account_summary():
    """Get IB account summary (balances, net liquidation value, etc.)."""
    if not ib_manager.is_connected:
        raise HTTPException(status_code=503, detail="IB Gateway not connected")

    summary = await ib_account.get_account_summary()
    if not summary:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve account summary"
        )

    return summary


@router.get("/account/positions", response_model=List[IBPosition])
async def get_account_positions():
    """Get current positions from IB account."""
    if not ib_manager.is_connected:
        raise HTTPException(status_code=503, detail="IB Gateway not connected")

    return await ib_account.get_positions()


@router.get("/account/pnl", response_model=IBPnL)
async def get_account_pnl():
    """Get portfolio-level P&L from IB."""
    if not ib_manager.is_connected:
        raise HTTPException(status_code=503, detail="IB Gateway not connected")

    pnl = await ib_account.get_pnl()
    if not pnl:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve P&L data"
        )

    return pnl


@router.get("/account/positions/pnl", response_model=List[IBPositionPnL])
async def get_positions_pnl():
    """Get per-position P&L from IB."""
    if not ib_manager.is_connected:
        raise HTTPException(status_code=503, detail="IB Gateway not connected")

    return await ib_account.get_position_pnl()
