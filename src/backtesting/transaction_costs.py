"""
Transaction Cost Model

Implements conservative transaction cost model (2x actual costs):
- Commissions: $0.01/share (2x IBKR Pro typical)
- Bid-ask spread: 0.05% (conservative estimate)
- Slippage: 0.05% (market impact)
- Expense ratios: 0.6% annual (2x typical low-cost ETF)
"""

import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """Conservative transaction cost model for ETF trading."""

    def __init__(
        self,
        commission_per_share: float = 0.01,  # $0.01 per share (2x IBKR Pro)
        bid_ask_spread_pct: float = 0.0005,  # 0.05% spread
        slippage_pct: float = 0.0005,        # 0.05% slippage
        expense_ratio: float = 0.006,        # 0.6% annual (2x typical)
        min_commission: float = 1.00         # Minimum commission per trade
    ):
        """
        Initialize transaction cost model.

        Parameters
        ----------
        commission_per_share : float
            Commission per share in dollars (default $0.01, 2x IBKR Pro)
        bid_ask_spread_pct : float
            Bid-ask spread as percentage (default 0.05%)
        slippage_pct : float
            Slippage/market impact as percentage (default 0.05%)
        expense_ratio : float
            Annual expense ratio (default 0.6%, conservative 2x typical)
        min_commission : float
            Minimum commission per trade (default $1.00)
        """
        self.commission_per_share = commission_per_share
        self.bid_ask_spread_pct = bid_ask_spread_pct
        self.slippage_pct = slippage_pct
        self.expense_ratio = expense_ratio
        self.min_commission = min_commission

        logger.info(
            f"Transaction Cost Model initialized:\n"
            f"  Commission: ${commission_per_share}/share (min ${min_commission})\n"
            f"  Bid-ask spread: {bid_ask_spread_pct*100:.3f}%\n"
            f"  Slippage: {slippage_pct*100:.3f}%\n"
            f"  Expense ratio: {expense_ratio*100:.2f}% annual"
        )

    def calculate_trade_cost(
        self,
        shares: float,
        price: float,
        is_buy: bool = True
    ) -> float:
        """
        Calculate cost for a single trade.

        Parameters
        ----------
        shares : float
            Number of shares traded (absolute value)
        price : float
            Price per share
        is_buy : bool
            True for buy, False for sell

        Returns
        -------
        float
            Total transaction cost
        """
        shares = abs(shares)
        trade_value = shares * price

        # Commission cost
        commission = max(
            shares * self.commission_per_share,
            self.min_commission
        )

        # Bid-ask spread cost (always paid)
        spread_cost = trade_value * self.bid_ask_spread_pct

        # Slippage cost (market impact)
        slippage_cost = trade_value * self.slippage_pct

        total_cost = commission + spread_cost + slippage_cost

        logger.debug(
            f"Trade cost for {shares:.0f} shares @ ${price:.2f} "
            f"({'BUY' if is_buy else 'SELL'}): "
            f"Commission=${commission:.2f}, Spread=${spread_cost:.2f}, "
            f"Slippage=${slippage_cost:.2f}, Total=${total_cost:.2f}"
        )

        return total_cost

    def calculate_rebalance_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate total cost to rebalance portfolio.

        Parameters
        ----------
        current_weights : dict
            Current portfolio weights {ticker: weight}
        target_weights : dict
            Target portfolio weights {ticker: weight}
        prices : dict
            Current prices {ticker: price}
        portfolio_value : float
            Total portfolio value

        Returns
        -------
        dict
            Cost breakdown: {
                'total_cost': float,
                'commission': float,
                'spread': float,
                'slippage': float,
                'turnover': float,
                'num_trades': int
            }
        """
        total_commission = 0
        total_spread = 0
        total_slippage = 0
        turnover = 0
        num_trades = 0

        # Get all tickers
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        for ticker in all_tickers:
            current_weight = current_weights.get(ticker, 0)
            target_weight = target_weights.get(ticker, 0)
            weight_change = abs(target_weight - current_weight)

            if weight_change < 0.0001:  # Skip negligible changes
                continue

            price = prices.get(ticker, 0)
            if price == 0:
                logger.warning(f"No price for {ticker}, skipping trade")
                continue

            # Calculate trade size
            trade_value = portfolio_value * weight_change
            shares = trade_value / price

            # Calculate costs
            commission = max(
                shares * self.commission_per_share,
                self.min_commission
            )
            spread_cost = trade_value * self.bid_ask_spread_pct
            slippage_cost = trade_value * self.slippage_pct

            total_commission += commission
            total_spread += spread_cost
            total_slippage += slippage_cost
            turnover += weight_change
            num_trades += 1

        total_cost = total_commission + total_spread + total_slippage

        return {
            'total_cost': total_cost,
            'commission': total_commission,
            'spread': total_spread,
            'slippage': total_slippage,
            'turnover': turnover,
            'num_trades': num_trades,
            'cost_pct': total_cost / portfolio_value if portfolio_value > 0 else 0
        }

    def calculate_daily_expense_ratio_cost(
        self,
        portfolio_value: float
    ) -> float:
        """
        Calculate daily expense ratio cost.

        Parameters
        ----------
        portfolio_value : float
            Current portfolio value

        Returns
        -------
        float
            Daily expense ratio cost
        """
        return portfolio_value * (self.expense_ratio / 252)

    def apply_expense_ratio(
        self,
        portfolio_values: pd.Series,
        frequency: str = 'daily'
    ) -> pd.Series:
        """
        Apply expense ratio drag to portfolio values.

        Parameters
        ----------
        portfolio_values : pd.Series
            Portfolio value series
        frequency : str
            'daily' or 'annual'

        Returns
        -------
        pd.Series
            Portfolio values after expense ratio
        """
        if frequency == 'daily':
            daily_er = self.expense_ratio / 252
            drag_factor = (1 - daily_er) ** np.arange(len(portfolio_values))
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        return portfolio_values * drag_factor

    def summary_report(self, total_costs: float, portfolio_value: float) -> str:
        """
        Generate summary report of transaction costs.

        Parameters
        ----------
        total_costs : float
            Total transaction costs incurred
        portfolio_value : float
            Final portfolio value

        Returns
        -------
        str
            Formatted summary report
        """
        cost_pct = (total_costs / portfolio_value * 100) if portfolio_value > 0 else 0

        report = f"""
╔══════════════════════════════════════════════════════════╗
║           TRANSACTION COST SUMMARY                       ║
╚══════════════════════════════════════════════════════════╝

  Total Costs:              ${total_costs:,.2f}
  Final Portfolio Value:    ${portfolio_value:,.2f}
  Cost as % of Value:       {cost_pct:.3f}%

  Cost Model (Conservative 2x):
    • Commission:           ${self.commission_per_share}/share (min ${self.min_commission})
    • Bid-ask spread:       {self.bid_ask_spread_pct*100:.3f}%
    • Slippage:             {self.slippage_pct*100:.3f}%
    • Expense ratio:        {self.expense_ratio*100:.2f}% annual

  Note: All costs are 2x typical market rates (conservative)
"""
        return report
