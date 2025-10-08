"""
Stop-Loss Strategy Implementation

Implements stop-loss strategies to protect against unforeseen losses:
- Fixed stop-loss: Sell when price drops below X% of purchase price
- Trailing stop-loss: Sell when price drops X% from peak since purchase
- Position-level stop-loss: Apply to individual holdings

Research shows 10% fixed stop-loss is optimal for ETF portfolios
(90% of purchase price as user suggested)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StopLossEvent:
    """Record of a stop-loss trigger event."""
    date: pd.Timestamp
    ticker: str
    purchase_price: float
    trigger_price: float
    loss_pct: float
    shares: float
    loss_amount: float


class StopLossManager:
    """
    Manage stop-loss orders for portfolio positions.

    Implements both fixed and trailing stop-loss strategies.
    Research shows 10% fixed stop-loss optimal for ETF portfolios.
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.10,  # 10% stop-loss (90% of purchase)
        use_trailing: bool = False,
        trailing_pct: float = 0.15     # 15% trailing stop if enabled
    ):
        """
        Initialize stop-loss manager.

        Parameters
        ----------
        stop_loss_pct : float
            Stop-loss percentage (default 10% = 90% of purchase price)
        use_trailing : bool
            Use trailing stop-loss instead of fixed (default False)
        trailing_pct : float
            Trailing stop percentage if use_trailing=True (default 15%)
        """
        self.stop_loss_pct = stop_loss_pct
        self.use_trailing = use_trailing
        self.trailing_pct = trailing_pct

        # Track positions: {ticker: {purchase_price, peak_price, shares}}
        self.positions: Dict[str, Dict] = {}

        # Track stop-loss events
        self.events: List[StopLossEvent] = []

        logger.info(
            f"Stop-Loss Manager initialized:\n"
            f"  Type: {'Trailing' if use_trailing else 'Fixed'}\n"
            f"  Stop-loss: {stop_loss_pct*100:.1f}%\n"
            f"  Trigger price: {(1-stop_loss_pct)*100:.1f}% of purchase"
        )

    def add_position(
        self,
        ticker: str,
        purchase_price: float,
        shares: float
    ) -> None:
        """
        Add new position to track for stop-loss.

        Parameters
        ----------
        ticker : str
            ETF ticker
        purchase_price : float
            Purchase price per share
        shares : float
            Number of shares purchased
        """
        self.positions[ticker] = {
            'purchase_price': purchase_price,
            'peak_price': purchase_price,
            'shares': shares,
            'initial_value': purchase_price * shares
        }

        logger.debug(
            f"Added position: {ticker} @ ${purchase_price:.2f}, "
            f"{shares:.0f} shares, stop @ ${purchase_price * (1-self.stop_loss_pct):.2f}"
        )

    def update_position(
        self,
        ticker: str,
        new_shares: float,
        purchase_price: float = None
    ) -> None:
        """
        Update existing position (after rebalancing).

        Parameters
        ----------
        ticker : str
            ETF ticker
        new_shares : float
            New number of shares
        purchase_price : float, optional
            New purchase price (if adding shares)
        """
        if ticker not in self.positions:
            if purchase_price is not None:
                self.add_position(ticker, purchase_price, new_shares)
            return

        pos = self.positions[ticker]

        if purchase_price is not None:
            # Adding shares - update average purchase price
            old_value = pos['shares'] * pos['purchase_price']
            new_value = (new_shares - pos['shares']) * purchase_price
            total_shares = new_shares
            avg_price = (old_value + new_value) / total_shares if total_shares > 0 else 0

            pos['purchase_price'] = avg_price
            pos['peak_price'] = max(pos['peak_price'], avg_price)

        pos['shares'] = new_shares

    def remove_position(self, ticker: str) -> None:
        """Remove position from tracking."""
        if ticker in self.positions:
            del self.positions[ticker]
            logger.debug(f"Removed position: {ticker}")

    def check_stops(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float]
    ) -> List[str]:
        """
        Check all positions for stop-loss triggers.

        Parameters
        ----------
        date : pd.Timestamp
            Current date
        prices : dict
            Current prices {ticker: price}

        Returns
        -------
        list
            List of tickers to sell (stop triggered)
        """
        triggered_tickers = []

        for ticker, pos in list(self.positions.items()):
            current_price = prices.get(ticker, 0)

            if current_price == 0:
                continue

            # Update peak price for trailing stop
            if self.use_trailing:
                pos['peak_price'] = max(pos['peak_price'], current_price)
                reference_price = pos['peak_price']
                stop_pct = self.trailing_pct
            else:
                reference_price = pos['purchase_price']
                stop_pct = self.stop_loss_pct

            # Check if stop triggered
            stop_price = reference_price * (1 - stop_pct)

            if current_price <= stop_price:
                # Stop triggered!
                loss_pct = (current_price - pos['purchase_price']) / pos['purchase_price']
                loss_amount = (current_price - pos['purchase_price']) * pos['shares']

                event = StopLossEvent(
                    date=date,
                    ticker=ticker,
                    purchase_price=pos['purchase_price'],
                    trigger_price=current_price,
                    loss_pct=loss_pct,
                    shares=pos['shares'],
                    loss_amount=loss_amount
                )

                self.events.append(event)
                triggered_tickers.append(ticker)

                logger.info(
                    f"🛑 STOP-LOSS TRIGGERED: {ticker}\n"
                    f"   Purchase: ${pos['purchase_price']:.2f}\n"
                    f"   Current:  ${current_price:.2f}\n"
                    f"   Stop:     ${stop_price:.2f}\n"
                    f"   Loss:     {loss_pct*100:.2f}% (${loss_amount:,.2f})\n"
                    f"   Shares:   {pos['shares']:.0f}"
                )

        return triggered_tickers

    def get_stop_summary(self) -> Dict:
        """
        Get summary of stop-loss activity.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self.events:
            return {
                'num_stops': 0,
                'total_loss': 0,
                'avg_loss_pct': 0,
                'max_loss_pct': 0,
                'tickers': []
            }

        losses_pct = [e.loss_pct for e in self.events]
        losses_amt = [e.loss_amount for e in self.events]

        return {
            'num_stops': len(self.events),
            'total_loss': sum(losses_amt),
            'avg_loss_pct': np.mean(losses_pct),
            'max_loss_pct': min(losses_pct),  # min because negative
            'tickers': [e.ticker for e in self.events]
        }

    def get_events_df(self) -> pd.DataFrame:
        """
        Get DataFrame of all stop-loss events.

        Returns
        -------
        pd.DataFrame
            Stop-loss events
        """
        if not self.events:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'date': e.date,
                'ticker': e.ticker,
                'purchase_price': e.purchase_price,
                'trigger_price': e.trigger_price,
                'loss_pct': e.loss_pct,
                'shares': e.shares,
                'loss_amount': e.loss_amount
            }
            for e in self.events
        ])

    def reset(self) -> None:
        """Reset all positions and events."""
        self.positions.clear()
        self.events.clear()
        logger.info("Stop-loss manager reset")

    def print_summary(self) -> None:
        """Print summary of stop-loss activity."""
        summary = self.get_stop_summary()

        print("\n" + "="*60)
        print("STOP-LOSS SUMMARY")
        print("="*60)
        print(f"\n  Strategy:           {'Trailing' if self.use_trailing else 'Fixed'}")
        print(f"  Stop-loss level:    {self.stop_loss_pct*100:.1f}%")
        print(f"  Trigger price:      {(1-self.stop_loss_pct)*100:.1f}% of purchase\n")

        if summary['num_stops'] == 0:
            print("  ✅ No stop-losses triggered")
        else:
            print(f"  🛑 Stops triggered:   {summary['num_stops']}")
            print(f"  💰 Total loss:        ${summary['total_loss']:,.2f}")
            print(f"  📊 Avg loss:          {summary['avg_loss_pct']*100:.2f}%")
            print(f"  📉 Max loss:          {summary['max_loss_pct']*100:.2f}%")
            print(f"\n  Tickers: {', '.join(summary['tickers'])}")

        print("="*60 + "\n")
