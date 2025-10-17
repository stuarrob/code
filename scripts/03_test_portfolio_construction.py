"""
Test Portfolio Construction Components

Creates synthetic factor scores and tests optimizer, rebalancer, and risk manager.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.portfolio import (
    SimpleOptimizer,
    RankBasedOptimizer,
    ThresholdRebalancer,
    PeriodicRebalancer,
    StopLossManager,
    VolatilityManager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_factor_scores(num_etfs: int = 300) -> pd.DataFrame:
    """Generate synthetic factor scores for testing."""
    np.random.seed(42)

    tickers = [f'ETF_{i:03d}' for i in range(num_etfs)]

    # Generate correlated factor scores
    # Momentum and quality are positively correlated
    momentum = np.random.randn(num_etfs)
    quality = 0.3 * momentum + 0.7 * np.random.randn(num_etfs)
    value = np.random.randn(num_etfs)
    low_volatility = -0.2 * momentum + 0.8 * np.random.randn(num_etfs)  # Negative correlation with momentum

    # Normalize to z-scores
    momentum = (momentum - momentum.mean()) / momentum.std()
    quality = (quality - quality.mean()) / quality.std()
    value = (value - value.mean()) / value.std()
    low_volatility = (low_volatility - low_volatility.mean()) / low_volatility.std()

    # Integrated score (geometric mean of z-scores)
    # Convert to positive scale
    weights = {'momentum': 0.35, 'quality': 0.30, 'value': 0.15, 'low_volatility': 0.20}

    integrated = (
        weights['momentum'] * momentum +
        weights['quality'] * quality +
        weights['value'] * value +
        weights['low_volatility'] * low_volatility
    )

    factor_scores = pd.DataFrame({
        'momentum': momentum,
        'quality': quality,
        'value': value,
        'low_volatility': low_volatility,
        'integrated': integrated
    }, index=tickers)

    return factor_scores


def test_optimizers(factor_scores: pd.DataFrame, num_positions: int = 30):
    """Test different optimizer implementations."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING OPTIMIZERS")
    logger.info(f"{'='*60}")

    integrated_scores = factor_scores['integrated']

    optimizers = {
        'Equal-Weighted': SimpleOptimizer(num_positions=num_positions),
        'Rank Linear': RankBasedOptimizer(num_positions=num_positions, weighting_scheme='linear'),
        'Rank Exponential': RankBasedOptimizer(num_positions=num_positions, weighting_scheme='exponential')
    }

    results = {}

    for name, optimizer in optimizers.items():
        logger.info(f"\n{name}:")
        weights = optimizer.optimize(integrated_scores)

        logger.info(f"  Positions: {len(weights)}")
        logger.info(f"  Max weight: {weights.max():.2%}")
        logger.info(f"  Min weight: {weights.min():.2%}")
        logger.info(f"  Weight std: {weights.std():.3f}")
        logger.info(f"  Top 10 concentration: {weights.nlargest(10).sum():.1%}")

        results[name] = weights

    return results


def test_rebalancer(weights: pd.Series):
    """Test rebalancing logic."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING REBALANCER")
    logger.info(f"{'='*60}")

    # Simulate drift
    np.random.seed(42)
    returns = np.random.randn(len(weights)) * 0.02  # 2% random returns
    current_weights = weights * (1 + returns)
    current_weights = current_weights / current_weights.sum()

    target_weights = weights

    # Calculate drift
    drift = (current_weights - target_weights).abs()
    logger.info(f"\nPortfolio Drift:")
    logger.info(f"  Max drift: {drift.max():.2%}")
    logger.info(f"  Mean drift: {drift.mean():.2%}")
    logger.info(f"  Positions with >2% drift: {(drift > 0.02).sum()}")

    # Test threshold rebalancer
    rebalancer = ThresholdRebalancer(
        drift_threshold=0.05,
        min_trade_size=0.01
    )

    decision = rebalancer.check_rebalance(
        current_weights=current_weights,
        target_weights=target_weights,
        current_date=pd.Timestamp('2024-01-15')
    )

    logger.info(f"\nRebalancing Decision:")
    logger.info(f"  Should rebalance: {decision.should_rebalance}")
    logger.info(f"  Reason: {decision.reason}")
    logger.info(f"  Drift: {decision.drift:.2%}")
    logger.info(f"  Trades needed: {len(decision.trades)}")

    if decision.should_rebalance and len(decision.trades) > 0:
        logger.info(f"\n  Top 5 Trades:")
        top_trades = decision.trades.abs().nlargest(5)
        for ticker in top_trades.index:
            trade = decision.trades[ticker]
            logger.info(f"    {ticker}: {trade:+.3%}")


def test_risk_manager(weights: pd.Series):
    """Test risk management."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING RISK MANAGER")
    logger.info(f"{'='*60}")

    # Create stop-loss manager
    stop_loss = StopLossManager(
        position_stop_loss=0.15,
        portfolio_stop_loss=0.20
    )

    # Simulate positions with losses
    np.random.seed(42)
    entry_prices = pd.Series(
        np.random.uniform(100, 200, len(weights)),
        index=weights.index
    )

    # Some positions have losses
    price_changes = np.random.uniform(0.70, 1.10, len(weights))
    current_prices = entry_prices * price_changes

    # Add positions
    for ticker, price in entry_prices.items():
        stop_loss.add_position(ticker, price)

    # Calculate losses
    losses = (entry_prices - current_prices) / entry_prices

    logger.info(f"\nPosition Losses:")
    logger.info(f"  Mean loss: {losses.mean():.2%}")
    logger.info(f"  Max loss: {losses.max():.2%}")
    logger.info(f"  Min loss: {losses.min():.2%}")
    logger.info(f"  Positions with >15% loss: {(losses > 0.15).sum()}")

    # Check risk
    signal = stop_loss.check_risk(
        current_prices=current_prices,
        weights=weights,
        portfolio_value=100000
    )

    logger.info(f"\nRisk Signal:")
    logger.info(f"  Action: {signal.action}")
    logger.info(f"  Reason: {signal.reason}")
    logger.info(f"  Severity: {signal.severity}")
    logger.info(f"  Positions to close: {len(signal.positions_to_close)}")
    logger.info(f"  Positions to reduce: {len(signal.positions_to_reduce)}")

    if len(signal.positions_to_close) > 0:
        logger.info(f"\n  Stop-loss triggered for:")
        for ticker in signal.positions_to_close[:5]:  # Show first 5
            logger.info(f"    {ticker}: {losses[ticker]:.2%} loss")


def test_volatility_manager():
    """Test volatility-based exposure management."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING VOLATILITY MANAGER")
    logger.info(f"{'='*60}")

    vol_mgr = VolatilityManager(
        target_volatility=0.15,
        lookback=60
    )

    # Test different volatility regimes
    regimes = {
        'Low (10%)': 0.0063,
        'Target (15%)': 0.0095,
        'High (25%)': 0.0157,
        'Crisis (40%)': 0.0252
    }

    logger.info(f"\nVolatility Targeting (Target: 15%):")

    for name, daily_vol in regimes.items():
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * daily_vol)

        exposure = vol_mgr.calculate_exposure(returns)
        realized_vol = returns.std() * np.sqrt(252)

        logger.info(f"  {name}: realized={realized_vol:.1%}, exposure={exposure:.1%}")


def test_at_scale(factor_scores: pd.DataFrame):
    """Test portfolio construction at different scales."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING AT SCALE")
    logger.info(f"{'='*60}")

    scales = [
        ('Pilot', 100),
        ('Medium', 200),
        ('Full', len(factor_scores))
    ]

    optimizer = SimpleOptimizer(num_positions=30)

    for name, num_etfs in scales:
        eligible = factor_scores['integrated'].nlargest(num_etfs)
        weights = optimizer.optimize(eligible)

        logger.info(f"\n{name} ({num_etfs} ETFs):")
        logger.info(f"  Portfolio positions: {len(weights)}")
        logger.info(f"  Min factor score: {eligible[weights.index].min():.3f}")
        logger.info(f"  Mean factor score: {eligible[weights.index].mean():.3f}")
        logger.info(f"  Max factor score: {eligible[weights.index].max():.3f}")


def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("PORTFOLIO CONSTRUCTION TEST")
    logger.info("="*60)

    # Generate synthetic data
    logger.info("\nGenerating synthetic factor scores...")
    factor_scores = generate_synthetic_factor_scores(num_etfs=300)

    logger.info(f"Generated factor scores for {len(factor_scores)} ETFs")
    logger.info(f"Factor columns: {factor_scores.columns.tolist()}")

    # Save for notebook
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "factor_scores_latest.parquet"
    factor_scores.to_parquet(output_file)
    logger.info(f"Saved factor scores to: {output_file}")

    # Test optimizers
    optimizer_results = test_optimizers(factor_scores, num_positions=30)

    # Test rebalancer
    test_rebalancer(optimizer_results['Equal-Weighted'])

    # Test risk manager
    test_risk_manager(optimizer_results['Equal-Weighted'])

    # Test volatility manager
    test_volatility_manager()

    # Test at scale
    test_at_scale(factor_scores)

    logger.info(f"\n{'='*60}")
    logger.info("ALL TESTS PASSED")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
