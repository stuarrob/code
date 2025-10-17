"""Portfolio API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.models.portfolio import Portfolio, PortfolioStatus
from app.schemas.portfolio import PortfolioCreate, PortfolioResponse, PortfolioUpdate


router = APIRouter()


@router.get("/", response_model=List[PortfolioResponse])
async def list_portfolios(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List all portfolios.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of portfolios
    """
    result = await db.execute(
        select(Portfolio)
        .offset(skip)
        .limit(limit)
        .order_by(Portfolio.created_at.desc())
    )
    portfolios = result.scalars().all()
    return portfolios


@router.post("/", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_in: PortfolioCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new portfolio.

    Args:
        portfolio_in: Portfolio creation data
        db: Database session

    Returns:
        Created portfolio
    """
    portfolio = Portfolio(
        name=portfolio_in.name,
        optimizer_type=portfolio_in.optimizer_type,
        num_positions=portfolio_in.num_positions,
        initial_capital=portfolio_in.initial_capital,
        current_value=portfolio_in.initial_capital,  # Start with initial capital
        cash=portfolio_in.initial_capital,  # Start with all cash
        is_paper_trading=portfolio_in.is_paper_trading,
        status=PortfolioStatus.ACTIVE
    )

    db.add(portfolio)
    await db.flush()
    await db.refresh(portfolio)

    return portfolio


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get portfolio by ID.

    Args:
        portfolio_id: Portfolio ID
        db: Database session

    Returns:
        Portfolio details
    """
    result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id)
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio {portfolio_id} not found"
        )

    return portfolio


@router.patch("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_id: int,
    portfolio_update: PortfolioUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update portfolio.

    Args:
        portfolio_id: Portfolio ID
        portfolio_update: Fields to update
        db: Database session

    Returns:
        Updated portfolio
    """
    result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id)
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio {portfolio_id} not found"
        )

    # Update fields
    update_data = portfolio_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(portfolio, field, value)

    await db.flush()
    await db.refresh(portfolio)

    return portfolio


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete (close) portfolio.

    Args:
        portfolio_id: Portfolio ID
        db: Database session
    """
    result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id)
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio {portfolio_id} not found"
        )

    # Don't actually delete - just mark as closed
    portfolio.status = PortfolioStatus.CLOSED
    await db.flush()
