"""
Transaction Cost Modeling

Models realistic trading costs for backtesting including:
- Commission fees
- Bid-ask spread
- Slippage
- ETF expense ratios

Uses conservative estimates (2x typical costs) as requested.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Model transaction costs for ETF trading.

    Typical ETF trading costs (as of 2024):
    - Commission: $0 (most brokers)
    - Bid-ask spread: 0.01% - 0.05% for liquid ETFs
    - Slippage: 0.01% - 0.10% depending on size
    - Market impact: minimal for ETFs with good liquidity

    This model uses 2x conservative estimates.
    """

    def __init__(self,
                 commission_per_trade: float = 0.0,
                 spread_bps: float = 2.0,  # 2 bps (0.02%) - already 2x typical
                 slippage_bps: float = 2.0,  # 2 bps - already 2x typical
                 min_trade_cost: float = 0.0):
        """
        Parameters
        ----------
        commission_per_trade : float
            Fixed commission per trade (default: $0, most brokers are commission-free)
        spread_bps : float
            Bid-ask spread in basis points (default: 2 bps = 0.02%)
            This is 2x typical spread for liquid ETFs
        slippage_bps : float
            Slippage in basis points (default: 2 bps = 0.02%)
            This is 2x typical slippage
        min_trade_cost : float
            Minimum cost per trade (default: $0)
        """
        self.commission = commission_per_trade
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.min_trade_cost = min_trade_cost

        logger.info(
            f"Transaction costs: commission=${commission_per_trade:.2f}, "
            f"spread={spread_bps:.1f}bps, slippage={slippage_bps:.1f}bps"
        )

    def calculate_trade_cost(self,
                            trade_value: float,
                            is_buy: bool = True) -> float:
        """
        Calculate cost for a single trade.

        Parameters
        ----------
        trade_value : float
            Dollar value of the trade (positive)
        is_buy : bool
            True if buying, False if selling

        Returns
        -------
        float
            Total transaction cost (always positive)
        """
        if trade_value <= 0:
            return 0.0

        # Commission (fixed)
        commission_cost = self.commission

        # Spread cost (% of trade value)
        # When buying, pay the ask (higher)
        # When selling, receive the bid (lower)
        spread_cost = trade_value * (self.spread_bps / 10000.0)

        # Slippage cost (% of trade value)
        slippage_cost = trade_value * (self.slippage_bps / 10000.0)

        # Total cost
        total_cost = commission_cost + spread_cost + slippage_cost

        # Apply minimum
        total_cost = max(total_cost, self.min_trade_cost)

        return total_cost

    def calculate_rebalance_cost(self,
                                current_holdings: Dict[str, float],
                                target_holdings: Dict[str, float],
                                prices: pd.Series) -> Dict:
        """
        Calculate total cost to rebalance portfolio.

        Parameters
        ----------
        current_holdings : dict
            Current number of shares held (ticker -> shares)
        target_holdings : dict
            Target number of shares (ticker -> shares)
        prices : pd.Series
            Current prices (ticker -> price)

        Returns
        -------
        dict
            Breakdown of rebalancing costs
        """
        all_tickers = set(current_holdings.keys()) | set(target_holdings.keys())

        buy_costs = 0.0
        sell_costs = 0.0
        buy_value = 0.0
        sell_value = 0.0
        num_buys = 0
        num_sells = 0

        trades = []

        for ticker in all_tickers:
            current = current_holdings.get(ticker, 0.0)
            target = target_holdings.get(ticker, 0.0)

            if ticker not in prices.index:
                logger.warning(f"No price available for {ticker}")
                continue

            price = prices[ticker]
            shares_diff = target - current

            if abs(shares_diff) < 0.01:  # Ignore tiny trades
                continue

            trade_value = abs(shares_diff * price)

            if shares_diff > 0:  # Buy
                cost = self.calculate_trade_cost(trade_value, is_buy=True)
                buy_costs += cost
                buy_value += trade_value
                num_buys += 1
                trades.append({
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares_diff,
                    'value': trade_value,
                    'cost': cost
                })

            else:  # Sell
                cost = self.calculate_trade_cost(trade_value, is_buy=False)
                sell_costs += cost
                sell_value += trade_value
                num_sells += 1
                trades.append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': abs(shares_diff),
                    'value': trade_value,
                    'cost': cost
                })

        total_cost = buy_costs + sell_costs
        total_value = buy_value + sell_value

        return {
            'total_cost': total_cost,
            'buy_costs': buy_costs,
            'sell_costs': sell_costs,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'total_traded_value': total_value,
            'num_buys': num_buys,
            'num_sells': num_sells,
            'num_trades': num_buys + num_sells,
            'cost_bps': (total_cost / total_value * 10000) if total_value > 0 else 0,
            'trades': trades
        }

    def calculate_expense_ratio_cost(self,
                                    portfolio_value: float,
                                    expense_ratios: pd.Series,
                                    weights: pd.Series,
                                    days: int = 1) -> float:
        """
        Calculate ETF expense ratio costs.

        Expense ratios are annual fees charged by ETFs.

        Parameters
        ----------
        portfolio_value : float
            Total portfolio value
        expense_ratios : pd.Series
            Annual expense ratios (ticker -> ratio, e.g., 0.0003 for 0.03%)
        weights : pd.Series
            Portfolio weights (ticker -> weight)
        days : int
            Number of days to calculate cost for

        Returns
        -------
        float
            Expense ratio cost for the period
        """
        # Align expense ratios and weights
        aligned_er, aligned_weights = expense_ratios.align(weights, join='inner', fill_value=0)

        # Weighted average expense ratio
        avg_expense_ratio = (aligned_er * aligned_weights).sum()

        # Daily cost
        daily_cost = portfolio_value * avg_expense_ratio / 365.25

        # Total cost for period
        total_cost = daily_cost * days

        return total_cost


class ConservativeCostModel(TransactionCostModel):
    """
    Extra conservative cost model (3x typical costs).

    For stress testing and worst-case scenarios.
    """

    def __init__(self):
        super().__init__(
            commission_per_trade=1.0,  # $1 per trade
            spread_bps=3.0,  # 3 bps spread
            slippage_bps=3.0,  # 3 bps slippage
            min_trade_cost=2.0  # $2 minimum
        )


class OptimisticCostModel(TransactionCostModel):
    """
    Optimistic cost model (typical costs, no multiplier).

    For best-case scenarios.
    """

    def __init__(self):
        super().__init__(
            commission_per_trade=0.0,
            spread_bps=1.0,  # 1 bp spread
            slippage_bps=1.0,  # 1 bp slippage
            min_trade_cost=0.0
        )


def estimate_turnover(current_weights: pd.Series,
                     new_weights: pd.Series) -> float:
    """
    Calculate portfolio turnover.

    Turnover = sum(|new_weight - old_weight|) / 2

    Parameters
    ----------
    current_weights : pd.Series
        Current portfolio weights
    new_weights : pd.Series
        New target weights

    Returns
    -------
    float
        Turnover as a fraction (0.0 to 1.0)
    """
    # Align weights
    all_tickers = current_weights.index.union(new_weights.index)

    current = current_weights.reindex(all_tickers, fill_value=0.0)
    new = new_weights.reindex(all_tickers, fill_value=0.0)

    # Turnover = sum of absolute changes / 2
    turnover = (current - new).abs().sum() / 2.0

    return turnover


def annualized_turnover(turnovers: pd.Series) -> float:
    """
    Calculate annualized turnover from a series of periodic turnovers.

    Parameters
    ----------
    turnovers : pd.Series
        Series of turnover values over time

    Returns
    -------
    float
        Annualized turnover
    """
    if len(turnovers) == 0:
        return 0.0

    # Average turnover per period
    avg_turnover = turnovers.mean()

    # Annualize based on period frequency
    # Assuming daily data - scale to annual
    periods_per_year = 252

    annual_turnover = avg_turnover * periods_per_year

    return annual_turnover
