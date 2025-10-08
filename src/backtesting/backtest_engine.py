"""
Backtesting Engine

Rolling window optimization with:
- Transaction costs
- Stop-loss protection
- Rebalancing strategies
- Performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from src.optimization.cvxpy_optimizer import create_optimizer
from src.signals.composite_signal import CompositeSignalGenerator
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.transaction_costs import TransactionCostModel
from src.backtesting.stop_loss import StopLossManager, StopLossEvent

logger = logging.getLogger(__name__)


@dataclass
class RebalanceEvent:
    """Record of a rebalancing event."""
    date: pd.Timestamp
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    turnover: float
    cost: float
    portfolio_value: float


class BacktestEngine:
    """
    Rolling window backtest engine for ETF portfolio strategies.

    Features:
    - Rolling window optimization
    - Transaction costs
    - Stop-loss protection
    - Multiple rebalancing frequencies
    - Contributions and withdrawals
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        rebalance_frequency: str = 'monthly',  # 'weekly', 'monthly', 'quarterly'
        lookback_period: int = 252,            # Days for optimization window
        min_history: int = 63,                 # Minimum history required (3 months)
        variant: str = 'balanced',
        enable_stop_loss: bool = True,
        stop_loss_pct: float = 0.10,
        enable_transaction_costs: bool = True,
        risk_free_rate: float = 0.04,
        asset_class_map: Dict[str, str] = None
    ):
        """
        Initialize backtest engine.

        Parameters
        ----------
        initial_capital : float
            Starting portfolio value
        rebalance_frequency : str
            How often to rebalance ('weekly', 'monthly', 'quarterly')
        lookback_period : int
            Days of history for optimization window
        min_history : int
            Minimum days required before first optimization
        variant : str
            Optimization variant ('max_sharpe', 'balanced', 'min_drawdown')
        enable_stop_loss : bool
            Enable stop-loss protection
        stop_loss_pct : float
            Stop-loss percentage (default 10%)
        enable_transaction_costs : bool
            Include transaction costs
        risk_free_rate : float
            Risk-free rate for Sharpe calculation
        asset_class_map : dict
            Ticker to asset class mapping
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.lookback_period = lookback_period
        self.min_history = min_history
        self.variant = variant
        self.enable_stop_loss = enable_stop_loss
        self.enable_transaction_costs = enable_transaction_costs
        self.asset_class_map = asset_class_map or {}

        # Initialize components
        self.signal_generator = CompositeSignalGenerator()
        self.performance = PerformanceMetrics(risk_free_rate)
        self.cost_model = TransactionCostModel() if enable_transaction_costs else None
        self.stop_loss = StopLossManager(stop_loss_pct) if enable_stop_loss else None

        # Track portfolio state
        self.current_weights: Dict[str, float] = {}
        self.current_shares: Dict[str, float] = {}
        self.cash = initial_capital

        # Track events
        self.rebalance_events: List[RebalanceEvent] = []
        self.portfolio_history = []
        self.weights_history = []

        logger.info(
            f"Backtest Engine initialized:\n"
            f"  Initial capital:     ${initial_capital:,.0f}\n"
            f"  Rebalance:           {rebalance_frequency}\n"
            f"  Lookback:            {lookback_period} days\n"
            f"  Variant:             {variant}\n"
            f"  Stop-loss:           {'Enabled' if enable_stop_loss else 'Disabled'}\n"
            f"  Transaction costs:   {'Enabled' if enable_transaction_costs else 'Disabled'}"
        )

    def should_rebalance(
        self,
        current_date: pd.Timestamp,
        last_rebalance: Optional[pd.Timestamp]
    ) -> bool:
        """
        Check if portfolio should be rebalanced.

        Parameters
        ----------
        current_date : pd.Timestamp
            Current date
        last_rebalance : pd.Timestamp or None
            Date of last rebalance

        Returns
        -------
        bool
            True if should rebalance
        """
        if last_rebalance is None:
            return True

        if self.rebalance_frequency == 'weekly':
            return (current_date - last_rebalance).days >= 7
        elif self.rebalance_frequency == 'monthly':
            return current_date.month != last_rebalance.month
        elif self.rebalance_frequency == 'quarterly':
            return current_date.quarter != last_rebalance.quarter
        else:
            return False

    def optimize_portfolio(
        self,
        prices: pd.DataFrame,
        end_date: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Run optimization on historical data up to end_date.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical prices
        end_date : pd.Timestamp
            End of optimization window

        Returns
        -------
        dict or None
            Optimal weights {ticker: weight} or None if failed
        """
        # Get lookback window
        window_start = end_date - pd.Timedelta(days=self.lookback_period)
        window_prices = prices.loc[:end_date].loc[window_start:]

        if len(window_prices) < self.min_history:
            logger.warning(f"Insufficient history at {end_date}: {len(window_prices)} days")
            return None

        # Calculate returns
        returns = window_prices.pct_change().dropna()

        if returns.empty:
            return None

        # Generate signals
        signals = {}
        for ticker in returns.columns:
            try:
                ticker_prices = window_prices[ticker].dropna()
                df = pd.DataFrame({
                    'Open': ticker_prices,
                    'High': ticker_prices,
                    'Low': ticker_prices,
                    'Close': ticker_prices,
                    'Volume': 1000000
                })
                scores_df = self.signal_generator.generate_signals_for_etf(df)
                if scores_df is not None and 'composite_score' in scores_df.columns:
                    signals[ticker] = scores_df['composite_score'].iloc[-1]
            except Exception as e:
                logger.debug(f"Signal generation failed for {ticker}: {e}")
                continue

        if not signals:
            logger.warning(f"No signals generated at {end_date}")
            return None

        signals_series = pd.Series(signals)

        # Align data
        common_tickers = returns.columns.intersection(signals_series.index)
        returns = returns[common_tickers]
        signals_series = signals_series[common_tickers]

        if len(common_tickers) < 10:
            logger.warning(f"Too few tickers at {end_date}: {len(common_tickers)}")
            return None

        # Run optimization
        try:
            optimizer = create_optimizer(
                variant=self.variant,
                max_positions=20,
                max_weight=0.15,
                min_weight=0.02,
                asset_class_map=self.asset_class_map,
                max_asset_class_weight=0.20,
                prefilter_top_n=min(400, len(common_tickers)),
                use_ledoit_wolf=True,
                solver="ECOS",
                solver_tolerance=1e-4
            )

            result = optimizer.optimize(returns, signals_series)

            if result and 'weights' in result and not result['weights'].empty:
                return result['weights'].to_dict()
            else:
                logger.warning(f"Optimization failed at {end_date}")
                return None

        except Exception as e:
            logger.error(f"Optimization error at {end_date}: {e}")
            return None

    def execute_rebalance(
        self,
        date: pd.Timestamp,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """
        Execute portfolio rebalance.

        Parameters
        ----------
        date : pd.Timestamp
            Rebalance date
        target_weights : dict
            Target weights {ticker: weight}
        prices : dict
            Current prices {ticker: price}
        portfolio_value : float
            Current portfolio value

        Returns
        -------
        float
            Transaction cost
        """
        # Calculate costs
        cost = 0
        if self.enable_transaction_costs:
            cost_breakdown = self.cost_model.calculate_rebalance_cost(
                self.current_weights,
                target_weights,
                prices,
                portfolio_value
            )
            cost = cost_breakdown['total_cost']

            logger.info(
                f"Rebalance cost: ${cost:,.2f} ({cost_breakdown['cost_pct']*100:.3f}%), "
                f"{cost_breakdown['num_trades']} trades, "
                f"turnover={cost_breakdown['turnover']*100:.1f}%"
            )

        # Record event
        turnover = sum(abs(target_weights.get(t, 0) - self.current_weights.get(t, 0))
                      for t in set(target_weights.keys()) | set(self.current_weights.keys()))

        event = RebalanceEvent(
            date=date,
            old_weights=self.current_weights.copy(),
            new_weights=target_weights.copy(),
            turnover=turnover,
            cost=cost,
            portfolio_value=portfolio_value
        )
        self.rebalance_events.append(event)

        # Update portfolio
        portfolio_value_after_cost = portfolio_value - cost

        # Calculate new shares
        new_shares = {}
        for ticker, weight in target_weights.items():
            price = prices.get(ticker, 0)
            if price > 0:
                target_value = portfolio_value_after_cost * weight
                new_shares[ticker] = target_value / price

        self.current_weights = target_weights.copy()
        self.current_shares = new_shares

        # Update stop-loss positions
        if self.enable_stop_loss:
            for ticker, shares in new_shares.items():
                price = prices[ticker]
                self.stop_loss.update_position(ticker, shares, price)

            # Remove sold positions
            for ticker in list(self.stop_loss.positions.keys()):
                if ticker not in new_shares:
                    self.stop_loss.remove_position(ticker)

        return cost

    def run(
        self,
        prices: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        monthly_contribution: float = 0,
        annual_contribution: float = 0
    ) -> Dict:
        """
        Run backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data
        start_date : pd.Timestamp, optional
            Backtest start date
        end_date : pd.Timestamp, optional
            Backtest end date
        monthly_contribution : float
            Monthly contribution amount (default 0)
        annual_contribution : float
            Annual contribution amount (default 0)

        Returns
        -------
        dict
            Backtest results
        """
        # Date range
        if start_date is None:
            start_date = prices.index[self.lookback_period]
        if end_date is None:
            end_date = prices.index[-1]

        test_dates = prices.loc[start_date:end_date].index

        logger.info(
            f"\nStarting backtest:\n"
            f"  Period: {start_date.date()} to {end_date.date()}\n"
            f"  Days: {len(test_dates)}\n"
            f"  Monthly contribution: ${monthly_contribution:,.0f}\n"
            f"  Annual contribution: ${annual_contribution:,.0f}"
        )

        last_rebalance = None
        total_transaction_costs = 0
        last_month = None
        last_year = None

        for i, date in enumerate(test_dates):
            # Add contributions
            if monthly_contribution > 0 and (last_month is None or date.month != last_month):
                self.cash += monthly_contribution
                last_month = date.month
                logger.debug(f"Monthly contribution: ${monthly_contribution:,.0f}")

            if annual_contribution > 0 and (last_year is None or date.year != last_year):
                self.cash += annual_contribution
                last_year = date.year
                logger.info(f"Annual contribution: ${annual_contribution:,.0f}")

            # Get current prices
            current_prices = prices.loc[date].to_dict()

            # Calculate portfolio value
            holdings_value = sum(
                self.current_shares.get(ticker, 0) * current_prices.get(ticker, 0)
                for ticker in self.current_shares
            )
            portfolio_value = holdings_value + self.cash

            # Check stop-losses
            if self.enable_stop_loss and self.current_shares:
                triggered = self.stop_loss.check_stops(date, current_prices)

                if triggered:
                    # Sell triggered positions
                    for ticker in triggered:
                        shares = self.current_shares.get(ticker, 0)
                        price = current_prices.get(ticker, 0)
                        proceeds = shares * price

                        # Transaction cost
                        if self.enable_transaction_costs:
                            cost = self.cost_model.calculate_trade_cost(shares, price, is_buy=False)
                            proceeds -= cost
                            total_transaction_costs += cost

                        self.cash += proceeds
                        if ticker in self.current_shares:
                            del self.current_shares[ticker]
                        if ticker in self.current_weights:
                            del self.current_weights[ticker]

                    # Recalculate portfolio value
                    holdings_value = sum(
                        self.current_shares.get(ticker, 0) * current_prices.get(ticker, 0)
                        for ticker in self.current_shares
                    )
                    portfolio_value = holdings_value + self.cash

            # Check if should rebalance
            if self.should_rebalance(date, last_rebalance):
                logger.info(f"\n📅 Rebalancing on {date.date()}, Portfolio: ${portfolio_value:,.0f}")

                # Optimize
                target_weights = self.optimize_portfolio(prices, date)

                if target_weights:
                    # Execute rebalance
                    cost = self.execute_rebalance(date, target_weights, current_prices, portfolio_value)
                    total_transaction_costs += cost

                    # Update cash (all proceeds go to cash, then reallocate)
                    self.cash = portfolio_value - cost
                    portfolio_value = self.cash

                    # Allocate to new positions
                    for ticker, weight in target_weights.items():
                        price = current_prices.get(ticker, 0)
                        if price > 0:
                            target_value = portfolio_value * weight
                            shares = target_value / price
                            self.current_shares[ticker] = shares
                            self.cash -= target_value

                    last_rebalance = date
                else:
                    # Optimization failed - still update last_rebalance to avoid retrying every day
                    logger.warning(f"Skipping rebalance on {date.date()} due to optimization failure")
                    last_rebalance = date

            # Apply daily expense ratio
            if self.enable_transaction_costs:
                daily_er_cost = self.cost_model.calculate_daily_expense_ratio_cost(portfolio_value)
                self.cash -= daily_er_cost
                total_transaction_costs += daily_er_cost

            # Record history
            final_value = sum(
                self.current_shares.get(ticker, 0) * current_prices.get(ticker, 0)
                for ticker in self.current_shares
            ) + self.cash

            self.portfolio_history.append({
                'date': date,
                'value': final_value,
                'cash': self.cash,
                'holdings_value': final_value - self.cash
            })

            self.weights_history.append({
                'date': date,
                **self.current_weights
            })

            if i % 63 == 0:  # Log every quarter
                logger.info(f"  {date.date()}: ${final_value:,.0f}")

        # Calculate performance
        results = self._calculate_results(total_transaction_costs)

        logger.info("\n✅ Backtest complete")

        return results

    def _calculate_results(self, total_transaction_costs: float) -> Dict:
        """Calculate and package backtest results."""
        # Convert to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index('date')
        weights_df = pd.DataFrame(self.weights_history).set_index('date')

        # Calculate returns
        returns = portfolio_df['value'].pct_change().dropna()

        # Performance metrics
        metrics = self.performance.calculate_all_metrics(returns, portfolio_df['value'])

        # Additional info
        metrics['total_transaction_costs'] = total_transaction_costs
        metrics['num_rebalances'] = len(self.rebalance_events)
        metrics['avg_turnover'] = np.mean([e.turnover for e in self.rebalance_events]) if self.rebalance_events else 0

        # Stop-loss summary
        if self.enable_stop_loss:
            stop_summary = self.stop_loss.get_stop_summary()
            metrics.update({f'stop_loss_{k}': v for k, v in stop_summary.items()})

        return {
            'metrics': metrics,
            'portfolio_values': portfolio_df,
            'weights': weights_df,
            'returns': returns,
            'rebalance_events': self.rebalance_events,
            'stop_loss_events': self.stop_loss.events if self.enable_stop_loss else []
        }
