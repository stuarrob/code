"""
Backtesting Engine

Event-driven backtesting framework with:
- Rolling window optimization
- Realistic transaction costs
- Stop-loss management
- Portfolio rebalancing
- Performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

from .metrics import PerformanceMetrics
from .costs import TransactionCostModel, estimate_turnover

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    Defaults aligned with AQR Multi-Factor Project Plan:
    - Weekly rebalancing with 5% drift threshold
    - Maximum 20 positions
    - 12% stop-loss (10-15% range per plan)
    """
    initial_capital: float = 1_000_000
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    rebalance_frequency: str = 'weekly'  # AQR plan recommends weekly
    num_positions: int = 20  # AQR plan: maximum 20 positions
    stop_loss_pct: float = 0.12  # 12% stop-loss (AQR plan: 10-15%)
    use_stop_loss: bool = True
    risk_free_rate: float = 0.02
    benchmark_ticker: str = 'SPY'


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    date: pd.Timestamp
    cash: float
    holdings: Dict[str, float] = field(default_factory=dict)  # ticker -> shares
    entry_prices: Dict[str, float] = field(default_factory=dict)  # ticker -> entry price
    value: float = 0.0

    def copy(self):
        """Create a copy of the state."""
        return PortfolioState(
            date=self.date,
            cash=self.cash,
            holdings=self.holdings.copy(),
            entry_prices=self.entry_prices.copy(),
            value=self.value
        )


class BacktestEngine:
    """
    Main backtesting engine.

    Simulates trading strategy over historical data with:
    - Rolling window factor calculation
    - Portfolio optimization
    - Rebalancing
    - Transaction costs
    - Stop-loss management
    """

    def __init__(self,
                 config: BacktestConfig,
                 cost_model: Optional[TransactionCostModel] = None):
        """
        Parameters
        ----------
        config : BacktestConfig
            Backtesting configuration
        cost_model : TransactionCostModel, optional
            Transaction cost model (uses default if not provided)
        """
        self.config = config
        self.cost_model = cost_model or TransactionCostModel()
        self.metrics_calc = PerformanceMetrics(risk_free_rate=config.risk_free_rate)

        # Results tracking
        self.portfolio_history: List[PortfolioState] = []
        self.trades_history: List[Dict] = []
        self.rebalance_dates: List[pd.Timestamp] = []

        # Performance tracking
        self.daily_values = pd.Series(dtype=float)
        self.daily_returns = pd.Series(dtype=float)
        self.turnovers = pd.Series(dtype=float)
        self.transaction_costs = pd.Series(dtype=float)

    def run(self,
            prices: pd.DataFrame,
            factor_scores: pd.DataFrame,
            optimizer,
            rebalancer,
            risk_manager,
            expense_ratios: Optional[pd.Series] = None) -> Dict:
        """
        Run backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical prices (dates x tickers)
        factor_scores : pd.DataFrame
            Factor scores over time (dates x tickers)
            Can be precomputed or calculated rolling
        optimizer : Optimizer
            Portfolio optimizer instance
        rebalancer : Rebalancer
            Rebalancing logic instance
        risk_manager : RiskManager
            Risk management instance
        expense_ratios : pd.Series, optional
            ETF expense ratios (ticker -> ratio)

        Returns
        -------
        dict
            Backtest results and metrics
        """
        logger.info("="*60)
        logger.info("STARTING BACKTEST")
        logger.info("="*60)
        logger.info(f"Initial capital: ${self.config.initial_capital:,.0f}")
        logger.info(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        logger.info(f"Rebalance frequency: {self.config.rebalance_frequency}")
        logger.info(f"Stop-loss: {self.config.stop_loss_pct:.1%}" if self.config.use_stop_loss else "No stop-loss")

        # Initialize
        start_date = self.config.start_date or prices.index[0]
        end_date = self.config.end_date or prices.index[-1]

        # Filter data
        prices = prices.loc[start_date:end_date]
        factor_scores = factor_scores.loc[start_date:end_date]

        # Initialize portfolio
        state = PortfolioState(
            date=prices.index[0],
            cash=self.config.initial_capital,
            holdings={},
            value=self.config.initial_capital
        )

        # Get initial portfolio
        initial_scores = factor_scores.iloc[0].dropna()
        if len(initial_scores) > 0:
            initial_weights = self._call_optimizer(optimizer, initial_scores, prices, 0)
            state = self._execute_rebalance(
                state, initial_weights, prices.iloc[0],
                current_weights=pd.Series(dtype=float)
            )
            self.rebalance_dates.append(state.date)

        self.portfolio_history.append(state.copy())

        # Simulate each day
        for i in range(1, len(prices)):
            current_date = prices.index[i]
            current_prices = prices.iloc[i]

            # Update portfolio value
            state.date = current_date
            state = self._update_portfolio_value(state, current_prices)

            # Check stop-loss
            if self.config.use_stop_loss:
                stop_loss_signal = self._check_stop_loss(state, current_prices)
                if stop_loss_signal['triggered']:
                    state = self._execute_stop_loss(state, stop_loss_signal, current_prices)

            # Check rebalancing
            if self._should_rebalance(current_date, prices.index):
                current_scores = factor_scores.iloc[i].dropna()

                if len(current_scores) > 0:
                    # Optimize
                    target_weights = self._call_optimizer(optimizer, current_scores, prices, i)

                    # Calculate current weights
                    current_weights = self._get_current_weights(state, current_prices)

                    # Check if rebalancer says to rebalance
                    decision = rebalancer.check_rebalance(
                        current_weights=current_weights,
                        target_weights=target_weights,
                        current_date=current_date
                    )

                    if decision.should_rebalance:
                        # Execute rebalance
                        state = self._execute_rebalance(
                            state, target_weights, current_prices, current_weights
                        )
                        rebalancer.execute_rebalance(current_date)
                        self.rebalance_dates.append(current_date)

            # Record state
            self.portfolio_history.append(state.copy())
            self.daily_values[current_date] = state.value

            # Calculate daily return
            if i > 0:
                prev_value = self.portfolio_history[i-1].value
                self.daily_returns[current_date] = (state.value / prev_value) - 1.0

        # Calculate metrics
        results = self._compile_results(prices, expense_ratios)

        logger.info("="*60)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*60)

        return results

    def _call_optimizer(self, optimizer, factor_scores: pd.Series, prices: pd.DataFrame, current_idx: int) -> pd.Series:
        """
        Call optimizer with appropriate arguments.

        Some optimizers (Simple, RankBased) only need factor_scores.
        Others (MinVar, MVO) need both factor_scores and prices.
        """
        import inspect

        # Check optimizer's optimize method signature
        sig = inspect.signature(optimizer.optimize)
        params = list(sig.parameters.keys())

        # If optimizer takes 'prices' parameter, provide historical prices up to current point
        if 'prices' in params:
            # Provide lookback window of prices
            lookback = getattr(optimizer, 'lookback', 60)
            start_idx = max(0, current_idx - lookback)
            price_window = prices.iloc[start_idx:current_idx+1]

            # Check if we have enough data
            if len(price_window) < max(20, lookback // 4):  # Need at least 20 days or 25% of lookback
                logger.warning(f"Insufficient price history ({len(price_window)} days) for optimizer at idx {current_idx}. Using simple equal-weight fallback.")
                # Fallback to equal weight
                n_pos = getattr(optimizer, 'num_positions', 20)
                top_etfs = factor_scores.nlargest(n_pos)
                return pd.Series(1.0 / len(top_etfs), index=top_etfs.index)

            return optimizer.optimize(factor_scores, price_window)
        else:
            # Simple optimizer - just needs scores
            return optimizer.optimize(factor_scores)

    def _should_rebalance(self, current_date: pd.Timestamp, all_dates: pd.DatetimeIndex) -> bool:
        """Check if it's time to rebalance based on frequency."""
        if len(self.rebalance_dates) == 0:
            return True

        last_rebalance = self.rebalance_dates[-1]

        if self.config.rebalance_frequency == 'daily':
            return True
        elif self.config.rebalance_frequency == 'weekly':
            return (current_date - last_rebalance).days >= 7
        elif self.config.rebalance_frequency == 'monthly':
            return current_date.month != last_rebalance.month or current_date.year != last_rebalance.year
        elif self.config.rebalance_frequency == 'bimonthly':
            # Every 2 months = 6 rebalances per year
            return (current_date - last_rebalance).days >= 60
        elif self.config.rebalance_frequency == 'quarterly':
            curr_quarter = (current_date.month - 1) // 3
            last_quarter = (last_rebalance.month - 1) // 3
            return curr_quarter != last_quarter or current_date.year != last_rebalance.year

        return False

    def _update_portfolio_value(self, state: PortfolioState, prices: pd.Series) -> PortfolioState:
        """Update portfolio value based on current prices."""
        holdings_value = 0.0

        for ticker, shares in state.holdings.items():
            if ticker in prices.index and not pd.isna(prices[ticker]):
                holdings_value += shares * prices[ticker]

        state.value = state.cash + holdings_value

        return state

    def _get_current_weights(self, state: PortfolioState, prices: pd.Series) -> pd.Series:
        """Calculate current portfolio weights."""
        if state.value <= 0:
            return pd.Series(dtype=float)

        weights = {}

        for ticker, shares in state.holdings.items():
            if ticker in prices.index and not pd.isna(prices[ticker]):
                value = shares * prices[ticker]
                weights[ticker] = value / state.value

        return pd.Series(weights)

    def _execute_rebalance(self,
                          state: PortfolioState,
                          target_weights: pd.Series,
                          prices: pd.Series,
                          current_weights: pd.Series) -> PortfolioState:
        """Execute portfolio rebalancing."""
        # Calculate turnover
        turnover = 0.0
        if len(current_weights) > 0:
            turnover = estimate_turnover(current_weights, target_weights)
            self.turnovers[state.date] = turnover

        # Calculate target holdings in shares
        target_holdings = {}
        for ticker, weight in target_weights.items():
            if ticker in prices.index and not pd.isna(prices[ticker]):
                target_value = state.value * weight
                target_shares = target_value / prices[ticker]
                target_holdings[ticker] = target_shares

        # Calculate costs
        cost_info = self.cost_model.calculate_rebalance_cost(
            current_holdings=state.holdings,
            target_holdings=target_holdings,
            prices=prices
        )

        # Deduct costs from cash
        state.cash -= cost_info['total_cost']
        self.transaction_costs[state.date] = cost_info['total_cost']

        # Update holdings
        state.holdings = target_holdings.copy()

        # Update entry prices for new positions
        for ticker in target_holdings.keys():
            if ticker not in state.entry_prices or state.entry_prices.get(ticker, 0) == 0:
                if ticker in prices.index:
                    state.entry_prices[ticker] = prices[ticker]

        # Record trades
        for trade in cost_info['trades']:
            self.trades_history.append({
                'date': state.date,
                **trade
            })

        # Recalculate cash (what's left after buying/selling)
        # Buy value should come from cash, sell value adds to cash
        cash_flow = cost_info['sell_value'] - cost_info['buy_value'] - cost_info['total_cost']
        state.cash += cash_flow

        logger.info(
            f"Rebalanced on {state.date.date()}: "
            f"{cost_info['num_trades']} trades, "
            f"turnover={turnover:.1%}, "
            f"cost=${cost_info['total_cost']:.2f}"
        )

        return state

    def _check_stop_loss(self, state: PortfolioState, prices: pd.Series) -> Dict:
        """Check if any positions hit stop-loss."""
        triggered_positions = []

        for ticker, shares in state.holdings.items():
            if ticker not in state.entry_prices or shares <= 0:
                continue

            if ticker not in prices.index or pd.isna(prices[ticker]):
                continue

            entry_price = state.entry_prices[ticker]
            current_price = prices[ticker]

            # Calculate loss from entry
            loss = (entry_price - current_price) / entry_price

            if loss > self.config.stop_loss_pct:
                triggered_positions.append(ticker)

        return {
            'triggered': len(triggered_positions) > 0,
            'positions': triggered_positions
        }

    def _execute_stop_loss(self, state: PortfolioState, signal: Dict, prices: pd.Series) -> PortfolioState:
        """Execute stop-loss by closing positions."""
        for ticker in signal['positions']:
            if ticker in state.holdings:
                shares = state.holdings[ticker]
                price = prices[ticker]

                # Sell position
                sale_value = shares * price
                trade_cost = self.cost_model.calculate_trade_cost(sale_value, is_buy=False)

                state.cash += sale_value - trade_cost

                # Record trade
                self.trades_history.append({
                    'date': state.date,
                    'ticker': ticker,
                    'action': 'STOP_LOSS',
                    'shares': shares,
                    'value': sale_value,
                    'cost': trade_cost
                })

                # Remove from holdings
                del state.holdings[ticker]
                if ticker in state.entry_prices:
                    del state.entry_prices[ticker]

                logger.warning(
                    f"Stop-loss triggered for {ticker} on {state.date.date()}: "
                    f"sold {shares:.2f} shares @ ${price:.2f}"
                )

        return state

    def _compile_results(self, prices: pd.DataFrame, expense_ratios: Optional[pd.Series]) -> Dict:
        """Compile final backtest results."""
        # Calculate returns
        returns = self.daily_returns.dropna()

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            portfolio_values=self.daily_values,
            returns=returns
        )

        # Add backtest-specific metrics
        metrics['num_rebalances'] = len(self.rebalance_dates)
        metrics['total_transaction_costs'] = self.transaction_costs.sum()
        metrics['avg_turnover'] = self.turnovers.mean() if len(self.turnovers) > 0 else 0.0
        metrics['num_trades'] = len(self.trades_history)

        # Final portfolio state
        final_state = self.portfolio_history[-1]
        metrics['final_value'] = final_state.value
        metrics['final_cash'] = final_state.cash
        metrics['final_positions'] = len(final_state.holdings)

        # Extract holdings history from portfolio states
        holdings_history = {}
        for state in self.portfolio_history:
            if len(state.holdings) > 0:
                holdings_history[state.date] = state.holdings.copy()

        return {
            'metrics': metrics,
            'daily_values': self.daily_values,
            'portfolio_values': self.daily_values,  # Alias for backward compatibility
            'daily_returns': returns,
            'holdings_history': holdings_history,
            'trades': pd.DataFrame(self.trades_history) if self.trades_history else pd.DataFrame(),
            'rebalance_dates': self.rebalance_dates,
            'turnovers': self.turnovers,
            'transaction_costs': self.transaction_costs,
            'config': self.config
        }
