"""Position API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from decimal import Decimal

from app.core.database import get_db
from app.models.portfolio import Portfolio
from app.models.position import Position
from app.schemas.position import PositionResponse, DriftResponse, RebalanceRecommendation
from app.services.factor_service import factor_service
from app.core.config import settings


router = APIRouter()


@router.get("/{portfolio_id}/positions", response_model=List[PositionResponse])
async def get_positions(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get all positions for a portfolio."""
    result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    result = await db.execute(
        select(Position).where(Position.portfolio_id == portfolio_id)
    )
    positions = result.scalars().all()
    return positions


@router.get("/{portfolio_id}/drift", response_model=DriftResponse)
async def calculate_drift(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Calculate position drift from target weights."""
    result = await db.execute(
        select(Position).where(Position.portfolio_id == portfolio_id)
    )
    positions = result.scalars().all()

    if not positions:
        return DriftResponse(
            max_drift=0.0,
            total_drift=0.0,
            needs_rebalancing=False,
            threshold=float(settings.DEFAULT_DRIFT_THRESHOLD),
            positions=[]
        )

    position_drifts = []
    total_drift = Decimal('0')
    max_drift = Decimal('0')

    for position in positions:
        current_weight = position.current_weight or Decimal('0')
        target_weight = position.target_weight or Decimal('0')
        drift = abs(current_weight - target_weight)
        total_drift += drift
        max_drift = max(max_drift, drift)

        position_drifts.append({
            'ticker': position.ticker,
            'current_weight': float(current_weight),
            'target_weight': float(target_weight),
            'drift': float(drift),
            'shares': float(position.shares or 0)
        })

    threshold = Decimal(str(settings.DEFAULT_DRIFT_THRESHOLD))
    return DriftResponse(
        max_drift=float(max_drift),
        total_drift=float(total_drift),
        needs_rebalancing=max_drift > threshold,
        threshold=float(threshold),
        positions=position_drifts
    )
