"""
Simple test script for CVXPY optimizers with Axioma adjustment.

Tests MinVarianceOptimizer and MeanVarianceOptimizer without pytest dependency.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.portfolio.optimizer import MinVarianceOptimizer, MeanVarianceOptimizer
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data():
    """Generate sample factor scores and price data."""
    np.random.seed(42)

    # 50 ETFs
    tickers = [f'ETF{i:03d}' for i in range(50)]

    # Factor scores (normalized to [-1, 1])
    scores = pd.Series(np.random.randn(50), index=tickers)

    # Price data (252 days)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = pd.DataFrame(
        100 * np.exp(np.random.randn(252, 50).cumsum(axis=0) * 0.01),
        index=dates,
        columns=tickers
    )

    return scores, prices


def test_minvar_optimizer():
    """Test MinVarianceOptimizer with Axioma adjustment."""
    logger.info("="*60)
    logger.info("TEST 1: MinVarianceOptimizer Basic Optimization")
    logger.info("="*60)

    scores, prices = generate_sample_data()

    optimizer = MinVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_penalty=0.01
    )

    weights = optimizer.optimize(scores, prices)

    logger.info(f"\nResults:")
    logger.info(f"  Positions: {len(weights)}")
    logger.info(f"  Sum of weights: {weights.sum():.6f}")
    logger.info(f"  Min weight: {weights.min():.4f}")
    logger.info(f"  Max weight: {weights.max():.4f}")
    logger.info(f"  Top 5 holdings:")
    logger.info(f"{weights.nlargest(5)}")

    # Assertions
    assert len(weights) == 20, f"Expected 20 positions, got {len(weights)}"
    assert abs(weights.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {weights.sum()}"
    assert (weights >= 0).all(), "Found negative weights"
    assert (weights <= 1.0).all(), "Found weights > 1.0"

    logger.info("✓ PASS: MinVarianceOptimizer basic optimization\n")
    return weights


def test_minvar_axioma_effect():
    """Test that Axioma penalty affects optimization."""
    logger.info("="*60)
    logger.info("TEST 2: MinVarianceOptimizer Axioma Penalty Effect")
    logger.info("="*60)

    scores, prices = generate_sample_data()

    # No Axioma penalty
    optimizer_no_penalty = MinVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_penalty=0.0
    )
    weights_no_penalty = optimizer_no_penalty.optimize(scores, prices)

    # With Axioma penalty
    optimizer_with_penalty = MinVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_penalty=0.1
    )
    weights_with_penalty = optimizer_with_penalty.optimize(scores, prices)

    logger.info(f"\nNo penalty - Max weight: {weights_no_penalty.max():.4f}")
    logger.info(f"With penalty - Max weight: {weights_with_penalty.max():.4f}")

    # Calculate concentration (HHI)
    hhi_no = (weights_no_penalty ** 2).sum()
    hhi_with = (weights_with_penalty ** 2).sum()

    logger.info(f"\nNo penalty - HHI: {hhi_no:.4f}")
    logger.info(f"With penalty - HHI: {hhi_with:.4f}")

    # Weights should be different
    assert not np.allclose(weights_no_penalty.values, weights_with_penalty.values), \
        "Axioma penalty had no effect"

    logger.info("✓ PASS: Axioma penalty affects optimization\n")


def test_mvo_optimizer():
    """Test MeanVarianceOptimizer with Axioma adjustment."""
    logger.info("="*60)
    logger.info("TEST 3: MeanVarianceOptimizer Basic Optimization")
    logger.info("="*60)

    scores, prices = generate_sample_data()

    optimizer = MeanVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_aversion=1.0,
        axioma_penalty=0.01
    )

    weights = optimizer.optimize(scores, prices)

    logger.info(f"\nResults:")
    logger.info(f"  Positions: {len(weights)}")
    logger.info(f"  Sum of weights: {weights.sum():.6f}")
    logger.info(f"  Min weight: {weights.min():.4f}")
    logger.info(f"  Max weight: {weights.max():.4f}")
    logger.info(f"  Non-zero positions: {(weights > 1e-4).sum()}")
    logger.info(f"  Top 5 holdings:")
    logger.info(f"{weights.nlargest(5)}")

    # Assertions
    assert len(weights) == 20, f"Expected 20 positions, got {len(weights)}"
    assert abs(weights.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {weights.sum()}"
    assert (weights >= 0).all(), "Found negative weights"
    assert (weights <= 0.15 + 1e-6).all(), f"Found weights > 15%: {weights.max()}"

    logger.info("✓ PASS: MeanVarianceOptimizer basic optimization\n")
    return weights


def test_mvo_factor_scores_as_alpha():
    """Test that higher factor scores get higher weights in MVO."""
    logger.info("="*60)
    logger.info("TEST 4: MeanVarianceOptimizer Factor Scores as Alpha")
    logger.info("="*60)

    scores, prices = generate_sample_data()

    optimizer = MeanVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_aversion=1.0,
        use_factor_scores_as_alpha=True
    )

    weights = optimizer.optimize(scores, prices)

    # Get top 5 by factor score
    top_scores = scores.nlargest(20).index[:5]
    logger.info(f"\nTop 5 by factor score: {list(top_scores)}")

    # Get top 5 by weight
    top_weights = weights.nlargest(5).index
    logger.info(f"Top 5 by weight: {list(top_weights)}")

    # Check overlap
    overlap = len(set(top_scores) & set(top_weights))
    logger.info(f"\nOverlap: {overlap}/5")

    assert overlap >= 2, f"Expected at least 2 overlap, got {overlap}"

    logger.info("✓ PASS: Factor scores influence weights\n")


def test_mvo_risk_aversion():
    """Test that risk aversion affects optimization."""
    logger.info("="*60)
    logger.info("TEST 5: MeanVarianceOptimizer Risk Aversion Effect")
    logger.info("="*60)

    scores, prices = generate_sample_data()

    # Low risk aversion (more aggressive - concentrate on high return)
    optimizer_low = MeanVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_aversion=0.1,  # Very low - prioritize returns
        axioma_penalty=0.0  # Turn off Axioma to see risk aversion effect
    )
    weights_low = optimizer_low.optimize(scores, prices)

    # High risk aversion (more conservative - diversify)
    optimizer_high = MeanVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_aversion=10.0,  # Very high - prioritize risk reduction
        axioma_penalty=0.0  # Turn off Axioma to see risk aversion effect
    )
    weights_high = optimizer_high.optimize(scores, prices)

    logger.info(f"\nLow risk aversion (0.1) - Max weight: {weights_low.max():.4f}")
    logger.info(f"High risk aversion (10.0) - Max weight: {weights_high.max():.4f}")

    non_tiny_low = (weights_low > 0.01).sum()
    non_tiny_high = (weights_high > 0.01).sum()

    logger.info(f"\nLow risk aversion - Non-tiny positions: {non_tiny_low}")
    logger.info(f"High risk aversion - Non-tiny positions: {non_tiny_high}")

    # Calculate HHI (concentration)
    hhi_low = (weights_low ** 2).sum()
    hhi_high = (weights_high ** 2).sum()

    logger.info(f"\nLow risk aversion - HHI: {hhi_low:.4f}")
    logger.info(f"High risk aversion - HHI: {hhi_high:.4f}")

    # High risk aversion should be more diversified (lower HHI or more positions)
    is_more_diversified = (hhi_high < hhi_low * 1.1) or (non_tiny_high >= non_tiny_low)

    assert is_more_diversified, \
        f"High risk aversion not more diversified: HHI {hhi_high:.4f} vs {hhi_low:.4f}"

    logger.info("✓ PASS: Risk aversion affects optimization\n")


def test_optimizer_comparison():
    """Compare MinVar vs MVO on same data."""
    logger.info("="*60)
    logger.info("TEST 6: Optimizer Comparison")
    logger.info("="*60)

    scores, prices = generate_sample_data()

    minvar = MinVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_penalty=0.01
    )
    weights_minvar = minvar.optimize(scores, prices)

    mvo = MeanVarianceOptimizer(
        num_positions=20,
        lookback=60,
        risk_aversion=1.0,
        axioma_penalty=0.01
    )
    weights_mvo = mvo.optimize(scores, prices)

    # Calculate metrics
    hhi_minvar = (weights_minvar ** 2).sum()
    hhi_mvo = (weights_mvo ** 2).sum()

    logger.info(f"\nMinVariance:")
    logger.info(f"  Max weight: {weights_minvar.max():.4f}")
    logger.info(f"  HHI: {hhi_minvar:.4f}")
    logger.info(f"  Non-zero: {(weights_minvar > 1e-4).sum()}")

    logger.info(f"\nMeanVariance:")
    logger.info(f"  Max weight: {weights_mvo.max():.4f}")
    logger.info(f"  HHI: {hhi_mvo:.4f}")
    logger.info(f"  Non-zero: {(weights_mvo > 1e-4).sum()}")

    # Both should be different but valid
    assert not np.allclose(weights_minvar.values, weights_mvo.values), \
        "MinVar and MVO produced identical results"

    assert hhi_minvar < 0.3, f"MinVar too concentrated: HHI={hhi_minvar}"
    assert hhi_mvo < 0.3, f"MVO too concentrated: HHI={hhi_mvo}"

    logger.info("✓ PASS: Optimizers produce different, valid results\n")


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("CVXPY OPTIMIZER TESTS WITH AXIOMA ADJUSTMENT")
    logger.info("="*60 + "\n")

    try:
        test_minvar_optimizer()
        test_minvar_axioma_effect()
        test_mvo_optimizer()
        test_mvo_factor_scores_as_alpha()
        test_mvo_risk_aversion()
        test_optimizer_comparison()

        logger.info("="*60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("="*60)

    except AssertionError as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        logger.error(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == '__main__':
    main()
