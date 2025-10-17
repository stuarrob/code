"""Paper trading engine - simulates trade execution."""

from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.portfolio import Portfolio
from app.models.position import Position
from app.models.trade import Trade, TradeSide, TradeStatus
from app.core.config import settings


class PaperTradingEngine:
    """
    Simulated trading engine for paper trading.

    Executes trades with realistic slippage and commissions.
    Updates portfolio positions and cash balance.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.commission = Decimal(str(settings.DEFAULT_COMMISSION))
        self.slippage_pct = Decimal(str(settings.DEFAULT_SLIPPAGE_PCT))

    async def execute_trade(
        self,
        portfolio_id: int,
        ticker: str,
        side: TradeSide,
        quantity: Decimal,
        price: Optional[Decimal] = None
    ) -> Trade:
        """
        Execute a simulated trade.

        Args:
            portfolio_id: Portfolio ID
            ticker: ETF ticker
            side: BUY or SELL
            quantity: Number of shares
            price: Execution price (if None, must be provided by caller)

        Returns:
            Executed trade record
        """
        # Get portfolio
        portfolio = await self._get_portfolio(portfolio_id)

        if price is None:
            raise ValueError("Price must be provided for paper trading")

        # Apply slippage
        if side == TradeSide.BUY:
            execution_price = price * (Decimal('1') + self.slippage_pct)
        else:
            execution_price = price * (Decimal('1') - self.slippage_pct)

        # Calculate costs
        slippage_cost = abs(quantity * (execution_price - price))
        total_value = abs(quantity * execution_price)

        # Create trade record
        trade = Trade(
            portfolio_id=portfolio_id,
            ticker=ticker,
            side=side,
            quantity=abs(quantity),
            price=execution_price,
            total_value=total_value,
            commission=self.commission,
            slippage=slippage_cost,
            status=TradeStatus.EXECUTED,
            executed_at=datetime.now()
        )

        # Update cash
        if side == TradeSide.BUY:
            portfolio.cash -= (total_value + self.commission)
        else:
            portfolio.cash += (total_value - self.commission)

        # Update or create position
        await self._update_position(portfolio_id, ticker, trade)

        # Save
        self.db.add(trade)
        await self.db.flush()
        await self.db.refresh(trade)

        return trade

    async def _get_portfolio(self, portfolio_id: int) -> Portfolio:
        """Get portfolio by ID."""
        result = await self.db.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        return portfolio

    async def _update_position(
        self,
        portfolio_id: int,
        ticker: str,
        trade: Trade
    ) -> Position:
        """Update or create position after trade execution."""
        # Get existing position
        result = await self.db.execute(
            select(Position).where(
                Position.portfolio_id == portfolio_id,
                Position.ticker == ticker
            )
        )
        position = result.scalar_one_or_none()

        if trade.side == TradeSide.BUY:
            if position:
                # Update existing position
                total_cost = (position.shares * position.entry_price) + (trade.quantity * trade.price)
                position.shares += trade.quantity
                position.entry_price = total_cost / position.shares
                position.current_price = trade.price
            else:
                # Create new position
                position = Position(
                    portfolio_id=portfolio_id,
                    ticker=ticker,
                    shares=trade.quantity,
                    entry_price=trade.price,
                    current_price=trade.price,
                    target_weight=Decimal('0'),
                    current_weight=Decimal('0')
                )
                self.db.add(position)

        else:  # SELL
            if position:
                position.shares -= trade.quantity
                if position.shares <= Decimal('0.0001'):
                    # Close position
                    await self.db.delete(position)
                    return None
                else:
                    position.current_price = trade.price
            else:
                raise ValueError(f"Cannot sell {ticker} - no position exists")

        if position:
            await self.db.flush()
            await self.db.refresh(position)

        return position

    async def check_stop_losses(
        self,
        portfolio_id: int,
        current_prices: Dict[str, Decimal],
        vix: Optional[Decimal] = None
    ) -> List[Dict]:
        """
        Check all positions for stop-loss breaches.

        Args:
            portfolio_id: Portfolio ID
            current_prices: Dict of {ticker: current_price}
            vix: Current VIX value for dynamic threshold

        Returns:
            List of positions that triggered stop-loss
        """
        # Get stop-loss threshold
        threshold = self._get_stop_loss_threshold(vix)

        # Get all positions
        result = await self.db.execute(
            select(Position).where(Position.portfolio_id == portfolio_id)
        )
        positions = result.scalars().all()

        triggered = []
        for position in positions:
            if position.ticker not in current_prices:
                continue

            current_price = current_prices[position.ticker]
            drawdown = (current_price - position.entry_price) / position.entry_price

            if drawdown < -threshold:
                triggered.append({
                    'ticker': position.ticker,
                    'entry_price': float(position.entry_price),
                    'current_price': float(current_price),
                    'drawdown': float(drawdown),
                    'threshold': float(-threshold),
                    'shares': float(position.shares)
                })

        return triggered

    def _get_stop_loss_threshold(self, vix: Optional[Decimal]) -> Decimal:
        """Get stop-loss threshold based on VIX regime."""
        if not settings.USE_VIX_ADJUSTMENT or vix is None:
            return Decimal(str(settings.DEFAULT_STOP_LOSS_PCT))

        # VIX-based dynamic threshold
        if vix < Decimal('15'):
            return Decimal('0.15')  # Low volatility - wider stop
        elif vix <= Decimal('25'):
            return Decimal('0.12')  # Normal volatility
        else:
            return Decimal('0.10')  # High volatility - tighter stop
