"""Risk management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, List
from decimal import Decimal
from pathlib import Path

from app.core.database import get_db
from app.models.portfolio import Portfolio
from app.models.position import Position
from app.core.config import settings


router = APIRouter()


@router.get("/{portfolio_id}")
async def get_risk_metrics(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db)
) -> Dict:
    """Get risk metrics and stop-loss status for portfolio."""
    # Get portfolio
    result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Get positions
    result = await db.execute(
        select(Position).where(Position.portfolio_id == portfolio_id)
    )
    positions = result.scalars().all()

    # Calculate risk metrics
    positions_at_risk = []
    for pos in positions:
        if pos.current_price and pos.entry_price:
            drawdown = (float(pos.current_price) - float(pos.entry_price)) / float(pos.entry_price)
            threshold = settings.DEFAULT_STOP_LOSS_PCT
            distance = threshold + drawdown

            if distance < 0.05:  # Within 5% of stop-loss
                positions_at_risk.append({
                    "ticker": pos.ticker,
                    "current_drawdown": drawdown,
                    "stop_loss_threshold": -threshold,
                    "distance_to_stop": distance
                })

    return {
        "stop_loss_threshold": settings.DEFAULT_STOP_LOSS_PCT,
        "vix_adjustment_enabled": settings.USE_VIX_ADJUSTMENT,
        "positions_at_risk": positions_at_risk,
        "num_positions": len(positions)
    }


@router.get("/vix")
async def get_vix_data() -> Dict:
    """Get VIX data and current regime."""
    # Try to load VIX data
    vix_file = Path(settings.DATA_DIR) / 'raw' / 'prices' / '^VIX.csv'
    
    if not vix_file.exists():
        return {
            "available": False,
            "message": "VIX data not found"
        }

    try:
        import pandas as pd
        vix_df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
        vix_series = vix_df['Adj Close'].dropna()
        current_vix = float(vix_series.iloc[-1])

        # Determine regime
        if current_vix < 15:
            regime = "Low Volatility"
            stop_loss = 0.15
        elif current_vix <= 25:
            regime = "Normal Volatility"
            stop_loss = 0.12
        else:
            regime = "High Volatility"
            stop_loss = 0.10

        # Get recent history
        recent = vix_series.tail(30)
        history = [
            {"date": date.strftime('%Y-%m-%d'), "value": float(val)}
            for date, val in recent.items()
        ]

        return {
            "available": True,
            "current": current_vix,
            "regime": regime,
            "stop_loss_threshold": stop_loss,
            "history": history
        }

    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }
