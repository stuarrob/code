"""Trade API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from decimal import Decimal

from app.core.database import get_db
from app.models.trade import Trade, TradeSide
from app.schemas.trade import TradeCreate, TradeResponse
from app.services.paper_trading import PaperTradingEngine
from app.services.price_service import price_service


router = APIRouter()


@router.get("/{portfolio_id}/trades", response_model=List[TradeResponse])
async def get_trades(
    portfolio_id: int,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get trade history for a portfolio."""
    result = await db.execute(
        select(Trade)
        .where(Trade.portfolio_id == portfolio_id)
        .order_by(Trade.executed_at.desc())
        .offset(skip)
        .limit(limit)
    )
    trades = result.scalars().all()
    return trades


@router.post("/", response_model=TradeResponse, status_code=status.HTTP_201_CREATED)
async def execute_trade(
    trade_in: TradeCreate,
    db: AsyncSession = Depends(get_db)
):
    """Execute a paper trade."""
    engine = PaperTradingEngine(db)

    side = TradeSide.BUY if trade_in.side == 'buy' else TradeSide.SELL

    # Get real price
    price = trade_in.price
    if price is None or price == Decimal('100'):  # Default placeholder price
        real_price = price_service.get_price(trade_in.ticker)
        if real_price:
            price = real_price
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Price not available for {trade_in.ticker}. Please provide a price."
            )

    # Calculate quantity if dollar_amount provided
    quantity = trade_in.quantity
    if trade_in.dollar_amount and not quantity:
        # Calculate shares from dollar amount
        quantity = trade_in.dollar_amount / price
    elif not quantity:
        raise HTTPException(
            status_code=400,
            detail="Either quantity or dollar_amount must be provided"
        )

    trade = await engine.execute_trade(
        portfolio_id=trade_in.portfolio_id,
        ticker=trade_in.ticker,
        side=side,
        quantity=quantity,
        price=price
    )

    return trade
