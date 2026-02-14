"""
Backtest: Growth Portfolio with Contributions

Portfolio with:
- Initial capital: $100,000
- Monthly contribution: $10,000
- Annual bonus contribution: $100,000
- Goal: Reach $3,000,000

Calculates time to reach target and tracks performance metrics.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.backtesting import (
    BacktestEngine,
    BacktestConfig,
    PerformanceMetrics,
    TransactionCostModel
)
from src.portfolio import (
    SimpleOptimizer,
    ThresholdRebalancer,
    StopLossManager
)

# Setup logging
log_file = project_root / "logs" / f"backtest_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GrowthPortfolioEngine(BacktestEngine):
    """
    Extended backtest engine with contribution tracking.

    Adds monthly and annual contributions to the portfolio.
    """

    def __init__(self,
                 config: BacktestConfig,
                 cost_model: TransactionCostModel,
                 monthly_contribution: float = 10_000,
                 annual_contribution: float = 100_000,
                 target_value: float = 3_000_000):
        """
        Parameters
        ----------
        config : BacktestConfig
            Backtesting configuration
        cost_model : TransactionCostModel
            Transaction cost model
        monthly_contribution : float
            Monthly contribution amount
        annual_contribution : float
            Annual bonus contribution
        target_value : float
            Target portfolio value
        """
        super().__init__(config, cost_model)

        self.monthly_contribution = monthly_contribution
        self.annual_contribution = annual_contribution
        self.target_value = target_value

        # Track contributions
        self.total_contributions = config.initial_capital
        self.contributions_history = []
        self.target_reached_date = None

    def run(self, prices, factor_scores, optimizer, rebalancer, risk_manager, expense_ratios=None):
        """Run backtest with contributions."""
        logger.info("="*60)
        logger.info("STARTING GROWTH PORTFOLIO BACKTEST")
        logger.info("="*60)
        logger.info(f"Initial capital: ${self.config.initial_capital:,.0f}")
        logger.info(f"Monthly contribution: ${self.monthly_contribution:,.0f}")
        logger.info(f"Annual contribution: ${self.annual_contribution:,.0f}")
        logger.info(f"Target value: ${self.target_value:,.0f}")

        # Run base backtest but inject contributions
        start_date = self.config.start_date or prices.index[0]
        end_date = self.config.end_date or prices.index[-1]

        # Filter data
        prices = prices.loc[start_date:end_date]
        factor_scores = factor_scores.loc[start_date:end_date]

        # Initialize portfolio
        from src.backtesting.engine import PortfolioState

        state = PortfolioState(
            date=prices.index[0],
            cash=self.config.initial_capital,
            holdings={},
            value=self.config.initial_capital
        )

        # Get initial portfolio
        initial_scores = factor_scores.iloc[0].dropna()
        if len(initial_scores) > 0:
            initial_weights = optimizer.optimize(initial_scores)
            state = self._execute_rebalance(
                state, initial_weights, prices.iloc[0],
                current_weights=pd.Series(dtype=float)
            )
            self.rebalance_dates.append(state.date)

        self.portfolio_history.append(state.copy())

        last_contribution_month = start_date.month
        last_contribution_year = start_date.year

        # Simulate each day
        for i in range(1, len(prices)):
            current_date = prices.index[i]
            current_prices = prices.iloc[i]

            # Check for monthly contribution
            if current_date.month != last_contribution_month:
                state.cash += self.monthly_contribution
                self.total_contributions += self.monthly_contribution
                self.contributions_history.append({
                    'date': current_date,
                    'type': 'monthly',
                    'amount': self.monthly_contribution
                })
                last_contribution_month = current_date.month

                logger.info(
                    f"Monthly contribution on {current_date.date()}: "
                    f"${self.monthly_contribution:,.0f} "
                    f"(total contributions: ${self.total_contributions:,.0f})"
                )

            # Check for annual contribution
            if current_date.year != last_contribution_year:
                state.cash += self.annual_contribution
                self.total_contributions += self.annual_contribution
                self.contributions_history.append({
                    'date': current_date,
                    'type': 'annual',
                    'amount': self.annual_contribution
                })
                last_contribution_year = current_date.year

                logger.info(
                    f"Annual contribution on {current_date.date()}: "
                    f"${self.annual_contribution:,.0f} "
                    f"(total contributions: ${self.total_contributions:,.0f})"
                )

            # Update portfolio value
            state.date = current_date
            state = self._update_portfolio_value(state, current_prices)

            # Check if target reached
            if self.target_reached_date is None and state.value >= self.target_value:
                self.target_reached_date = current_date
                years_to_target = (current_date - start_date).days / 365.25

                logger.info("="*60)
                logger.info(f"TARGET REACHED on {current_date.date()}!")
                logger.info(f"Portfolio value: ${state.value:,.0f}")
                logger.info(f"Time to target: {years_to_target:.1f} years")
                logger.info(f"Total contributions: ${self.total_contributions:,.0f}")
                logger.info(f"Investment gains: ${state.value - self.total_contributions:,.0f}")
                logger.info("="*60)

            # Check stop-loss
            if self.config.use_stop_loss:
                stop_loss_signal = self._check_stop_loss(state, current_prices)
                if stop_loss_signal['triggered']:
                    state = self._execute_stop_loss(state, stop_loss_signal, current_prices)

            # Check rebalancing
            if self._should_rebalance(current_date, prices.index):
                current_scores = factor_scores.iloc[i].dropna()

                if len(current_scores) > 0:
                    target_weights = optimizer.optimize(current_scores)
                    current_weights = self._get_current_weights(state, current_prices)

                    decision = rebalancer.check_rebalance(
                        current_weights=current_weights,
                        target_weights=target_weights,
                        current_date=current_date
                    )

                    if decision.should_rebalance:
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

        # Compile results
        results = self._compile_results(prices, expense_ratios)

        # Add contribution tracking
        results['total_contributions'] = self.total_contributions
        results['contributions_history'] = pd.DataFrame(self.contributions_history)
        results['target_reached_date'] = self.target_reached_date

        # Calculate contribution-adjusted metrics
        final_value = results['metrics']['final_value']
        investment_gain = final_value - self.total_contributions
        roi = investment_gain / self.total_contributions if self.total_contributions > 0 else 0

        results['metrics']['total_contributions'] = self.total_contributions
        results['metrics']['investment_gain'] = investment_gain
        results['metrics']['roi_on_contributions'] = roi

        if self.target_reached_date:
            years_to_target = (self.target_reached_date - start_date).days / 365.25
            results['metrics']['years_to_target'] = years_to_target
        else:
            results['metrics']['years_to_target'] = None

        logger.info("="*60)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*60)

        return results


def load_data():
    """Load factor scores and generate price data."""
    data_dir = project_root / "data"

    # Load factor scores
    factor_file = data_dir / "factor_scores_latest.parquet"
    if not factor_file.exists():
        raise FileNotFoundError(
            f"Factor scores not found. "
            "Run scripts/03_test_portfolio_construction.py first."
        )

    factor_scores = pd.read_parquet(factor_file)

    # Generate synthetic price data
    # Longer timeframe to potentially reach $3M goal
    np.random.seed(42)
    num_days = 2520  # ~10 years
    dates = pd.date_range('2015-01-01', periods=num_days, freq='D')

    tickers = factor_scores.index.tolist()

    returns_data = []
    for ticker in tickers:
        score = factor_scores.loc[ticker, 'integrated']
        base_drift = 0.0005  # 12% annual
        factor_alpha = score * 0.0002
        total_drift = base_drift + factor_alpha

        returns = np.random.randn(num_days) * 0.01 + total_drift
        returns_data.append(returns)

    returns_array = np.array(returns_data).T

    prices = pd.DataFrame(
        100 * (1 + returns_array).cumprod(axis=0),
        columns=tickers,
        index=dates
    )

    logger.info(f"Generated price data: {len(dates)} days, {len(tickers)} ETFs")

    # Expand factor scores
    expanded_scores = pd.DataFrame(
        [factor_scores['integrated'].values] * len(dates),
        columns=tickers,
        index=dates
    )

    return prices, expanded_scores


def display_results(results: dict):
    """Display growth portfolio results."""
    metrics = results['metrics']

    logger.info("\n" + "="*60)
    logger.info("GROWTH PORTFOLIO RESULTS")
    logger.info("="*60)

    # Portfolio Value Progression
    logger.info("\n" + "="*60)
    logger.info("PORTFOLIO VALUE PROGRESSION")
    logger.info("="*60)
    logger.info(f"Initial Value:       ${results['config'].initial_capital:>15,.0f}")
    logger.info(f"Final Value:         ${metrics['final_value']:>15,.0f}")
    logger.info(f"Target Value:        ${3_000_000:>15,.0f}")

    # Contribution Summary
    logger.info("\n" + "="*60)
    logger.info("CONTRIBUTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Contributed:   ${metrics['total_contributions']:>15,.0f}")
    logger.info(f"Investment Gain:     ${metrics['investment_gain']:>15,.0f}")
    logger.info(f"ROI on Capital:      {metrics['roi_on_contributions']:>15.2%}")

    # Time to Target
    logger.info("\n" + "="*60)
    logger.info("TIME TO REACH $3M TARGET")
    logger.info("="*60)
    if results['target_reached_date']:
        logger.info(f"Target reached on:   {results['target_reached_date'].date()}")
        logger.info(f"Time to target:      {metrics['years_to_target']:>15.1f} years")
        logger.info(f"Status:              ✓ TARGET REACHED")
    else:
        logger.info(f"Target reached:      NOT YET")
        logger.info(f"Current value:       ${metrics['final_value']:>15,.0f}")
        shortfall = 3_000_000 - metrics['final_value']
        logger.info(f"Shortfall:           ${shortfall:>15,.0f}")

    # Performance Metrics
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*60)
    logger.info(f"CAGR:                {metrics['cagr']:>15.2%}")
    logger.info(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>15.2f}")
    logger.info(f"Max Drawdown:        {metrics['max_drawdown']:>15.2%}")
    logger.info(f"Win Rate:            {metrics['win_rate']:>15.2%}")

    # Trading Activity
    logger.info("\n" + "="*60)
    logger.info("TRADING ACTIVITY")
    logger.info("="*60)
    logger.info(f"Rebalances:          {metrics['num_rebalances']:>15.0f}")
    logger.info(f"Total Trades:        {metrics['num_trades']:>15.0f}")
    logger.info(f"Avg Turnover:        {metrics['avg_turnover']:>15.2%}")
    logger.info(f"Transaction Costs:   ${metrics['total_transaction_costs']:>14,.0f}")

    logger.info("\n" + "="*60)


def save_results(results: dict, output_dir: Path):
    """Save growth portfolio results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save daily values
    daily_values_file = output_dir / f"daily_values_{timestamp}.csv"
    results['daily_values'].to_csv(daily_values_file)

    # Save contributions history
    if len(results['contributions_history']) > 0:
        contributions_file = output_dir / f"contributions_{timestamp}.csv"
        results['contributions_history'].to_csv(contributions_file, index=False)
        logger.info(f"Saved contributions to: {contributions_file}")

    # Save metrics
    metrics_file = output_dir / f"metrics_{timestamp}.csv"
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(metrics_file, index=False)

    logger.info(f"Saved results to: {output_dir}")


def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("GROWTH PORTFOLIO BACKTEST")
    logger.info("Initial: $100k + $10k/month + $100k/year")
    logger.info("="*60)
    logger.info(f"Started: {datetime.now()}")

    # Load data
    prices, factor_scores = load_data()

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100_000,
        start_date=prices.index[0],
        end_date=prices.index[-1],
        rebalance_frequency='weekly',
        num_positions=20,
        stop_loss_pct=0.12,
        use_stop_loss=True
    )

    # Create components
    optimizer = SimpleOptimizer(num_positions=config.num_positions)
    rebalancer = ThresholdRebalancer(drift_threshold=0.05)
    risk_manager = StopLossManager(position_stop_loss=0.10)
    cost_model = TransactionCostModel()

    # Run growth portfolio backtest
    engine = GrowthPortfolioEngine(
        config=config,
        cost_model=cost_model,
        monthly_contribution=10_000,
        annual_contribution=100_000,
        target_value=3_000_000
    )

    results = engine.run(
        prices=prices,
        factor_scores=factor_scores,
        optimizer=optimizer,
        rebalancer=rebalancer,
        risk_manager=risk_manager
    )

    # Display and save results
    display_results(results)

    output_dir = Path.home() / "trading" / "backtest_growth"
    save_results(results, output_dir)

    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Completed: {datetime.now()}")
    logger.info(f"Log file: {log_file}")

    return results


if __name__ == '__main__':
    results = main()
