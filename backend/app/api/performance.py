"""Performance API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict
from decimal import Decimal

from app.core.database import get_db
from app.models.portfolio import Portfolio
from app.models.portfolio_value import PortfolioValue
from app.models.position import Position


router = APIRouter()


@router.get("/{portfolio_id}")
async def get_performance(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db)
) -> Dict:
    """Get performance metrics for a portfolio."""
    # Get portfolio
    result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Get historical values
    result = await db.execute(
        select(PortfolioValue)
        .where(PortfolioValue.portfolio_id == portfolio_id)
        .order_by(PortfolioValue.date)
    )
    values = result.scalars().all()

    if not values:
        return {
            "total_return": 0.0,
            "current_value": float(portfolio.current_value or portfolio.initial_capital),
            "initial_capital": float(portfolio.initial_capital),
            "num_data_points": 0
        }

    # Calculate metrics
    initial = float(portfolio.initial_capital)
    current = float(portfolio.current_value or initial)
    total_return = (current - initial) / initial if initial > 0 else 0.0

    # Prepare chart data
    chart_data = [
        {
            "date": v.date.isoformat(),
            "value": float(v.total_value),
            "return": float(v.cumulative_return or 0)
        }
        for v in values
    ]

    return {
        "total_return": total_return,
        "current_value": current,
        "initial_capital": initial,
        "num_data_points": len(values),
        "chart_data": chart_data
    }


@router.get("/{portfolio_id}/attribution")
async def get_attribution(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db)
) -> Dict:
    """Get performance attribution by position."""
    # Get positions
    result = await db.execute(
        select(Position).where(Position.portfolio_id == portfolio_id)
    )
    positions = result.scalars().all()

    attribution = []
    for pos in positions:
        pnl = float(pos.unrealized_pnl or 0)
        attribution.append({
            "ticker": pos.ticker,
            "unrealized_pnl": pnl,
            "shares": float(pos.shares or 0),
            "entry_price": float(pos.entry_price or 0),
            "current_price": float(pos.current_price or 0)
        })

    return {"positions": attribution}
