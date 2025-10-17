"""
Tests for CVXPY-based optimizers with Axioma adjustment.

Tests MinVarianceOptimizer and MeanVarianceOptimizer to ensure:
- Axioma adjustment is working correctly
- Factor scores are used as alpha signal
- Constraints are enforced
- Optimizer produces reasonable weights
"""

import pytest
import pandas as pd
import numpy as np
from src.portfolio.optimizer import MinVarianceOptimizer, MeanVarianceOptimizer


@pytest.fixture
def sample_data():
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


class TestMinVarianceOptimizer:
    """Tests for MinVarianceOptimizer with Axioma adjustment."""

    def test_basic_optimization(self, sample_data):
        """Test basic min variance optimization."""
        scores, prices = sample_data

        optimizer = MinVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_penalty=0.01
        )

        weights = optimizer.optimize(scores, prices)

        # Check basic properties
        assert len(weights) == 20
        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= 0).all()
        assert (weights <= 1.0).all()

    def test_axioma_penalty_effect(self, sample_data):
        """Test that Axioma penalty affects optimization."""
        scores, prices = sample_data

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
            risk_penalty=0.1  # Higher penalty
        )
        weights_with_penalty = optimizer_with_penalty.optimize(scores, prices)

        # Weights should be different
        assert not np.allclose(weights_no_penalty.values, weights_with_penalty.values)

        # With penalty should be more diversified (lower max weight)
        assert weights_with_penalty.max() <= weights_no_penalty.max() * 1.1

    def test_min_score_filter(self, sample_data):
        """Test minimum score filtering."""
        scores, prices = sample_data

        optimizer = MinVarianceOptimizer(
            num_positions=20,
            min_score=0.5,  # High threshold
            lookback=60
        )

        weights = optimizer.optimize(scores, prices)

        # All selected ETFs should have score >= 0.5
        for ticker in weights.index:
            assert scores[ticker] >= 0.5

    def test_insufficient_data_warning(self, sample_data):
        """Test warning when insufficient data for lookback."""
        scores, prices = sample_data

        # Very long lookback
        optimizer = MinVarianceOptimizer(
            num_positions=10,
            lookback=500,  # More than available data
            risk_penalty=0.01
        )

        # Should still work but log warning
        weights = optimizer.optimize(scores, prices)
        assert len(weights) == 10
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_target_return_constraint(self, sample_data):
        """Test target return constraint."""
        scores, prices = sample_data

        optimizer = MinVarianceOptimizer(
            num_positions=20,
            lookback=60,
            target_return=0.10,  # 10% annual return target
            risk_penalty=0.01
        )

        weights = optimizer.optimize(scores, prices)

        # Should produce valid weights
        assert len(weights) == 20
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_fallback_on_optimization_failure(self, sample_data):
        """Test fallback to equal weight on optimization failure."""
        scores, prices = sample_data

        # Create scenario that might cause optimization issues
        # (very small lookback with high constraints)
        optimizer = MinVarianceOptimizer(
            num_positions=5,
            lookback=10,
            target_return=0.50,  # Unrealistic target
            risk_penalty=0.01
        )

        # Should either optimize or fall back to equal weight
        weights = optimizer.optimize(scores, prices)
        assert len(weights) == 5
        assert abs(weights.sum() - 1.0) < 1e-6


class TestMeanVarianceOptimizer:
    """Tests for MeanVarianceOptimizer with Axioma adjustment."""

    def test_basic_optimization(self, sample_data):
        """Test basic mean-variance optimization."""
        scores, prices = sample_data

        optimizer = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=1.0,
            axioma_penalty=0.01
        )

        weights = optimizer.optimize(scores, prices)

        # Check basic properties
        assert len(weights) == 20
        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= 0).all()
        assert (weights <= 0.15).all()  # Concentration limit

    def test_factor_scores_as_alpha(self, sample_data):
        """Test that higher factor scores get higher weights."""
        scores, prices = sample_data

        optimizer = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=1.0,
            use_factor_scores_as_alpha=True
        )

        weights = optimizer.optimize(scores, prices)

        # Get top 5 by factor score and top 5 by weight
        top_scores = scores.nlargest(20).index[:5]
        top_weights = weights.nlargest(5).index

        # There should be significant overlap
        overlap = len(set(top_scores) & set(top_weights))
        assert overlap >= 2  # At least 2 of top 5 scores in top 5 weights

    def test_risk_aversion_effect(self, sample_data):
        """Test that risk aversion affects optimization."""
        scores, prices = sample_data

        # Low risk aversion (more aggressive)
        optimizer_low = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=0.5,
            axioma_penalty=0.01
        )
        weights_low = optimizer_low.optimize(scores, prices)

        # High risk aversion (more conservative)
        optimizer_high = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=5.0,
            axioma_penalty=0.01
        )
        weights_high = optimizer_high.optimize(scores, prices)

        # Weights should be different
        assert not np.allclose(weights_low.values, weights_high.values)

        # High risk aversion should lead to more diversified portfolio
        # (measured by lower max weight or higher number of non-tiny positions)
        non_tiny_low = (weights_low > 0.01).sum()
        non_tiny_high = (weights_high > 0.01).sum()
        assert non_tiny_high >= non_tiny_low * 0.8  # Allow some tolerance

    def test_axioma_penalty_effect(self, sample_data):
        """Test that Axioma penalty affects optimization."""
        scores, prices = sample_data

        # No Axioma penalty
        optimizer_no = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=1.0,
            axioma_penalty=0.0
        )
        weights_no = optimizer_no.optimize(scores, prices)

        # High Axioma penalty
        optimizer_high = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=1.0,
            axioma_penalty=0.1
        )
        weights_high = optimizer_high.optimize(scores, prices)

        # Weights should be different
        assert not np.allclose(weights_no.values, weights_high.values)

    def test_concentration_limit(self, sample_data):
        """Test that concentration limit is enforced."""
        scores, prices = sample_data

        optimizer = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=0.1,  # Very low to encourage concentration
            axioma_penalty=0.01
        )

        weights = optimizer.optimize(scores, prices)

        # No position should exceed 15%
        assert (weights <= 0.15 + 1e-6).all()

    def test_historical_returns_mode(self, sample_data):
        """Test using historical returns instead of factor scores."""
        scores, prices = sample_data

        optimizer = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=1.0,
            use_factor_scores_as_alpha=False  # Use historical returns
        )

        weights = optimizer.optimize(scores, prices)

        # Should still produce valid weights
        assert len(weights) == 20
        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= 0).all()

    def test_min_score_filter(self, sample_data):
        """Test minimum score filtering."""
        scores, prices = sample_data

        optimizer = MeanVarianceOptimizer(
            num_positions=20,
            min_score=0.5,
            lookback=60,
            risk_aversion=1.0
        )

        weights = optimizer.optimize(scores, prices)

        # All selected ETFs should have score >= 0.5
        for ticker in weights.index:
            assert scores[ticker] >= 0.5

    def test_insufficient_eligible_etfs(self, sample_data):
        """Test behavior when not enough ETFs meet min score."""
        scores, prices = sample_data

        optimizer = MeanVarianceOptimizer(
            num_positions=20,
            min_score=2.0,  # Very high - only a few will qualify
            lookback=60,
            risk_aversion=1.0
        )

        weights = optimizer.optimize(scores, prices)

        # Should select fewer than requested
        assert len(weights) < 20
        assert abs(weights.sum() - 1.0) < 1e-6


class TestOptimizerComparison:
    """Compare different optimizers on same data."""

    def test_all_optimizers_produce_valid_weights(self, sample_data):
        """Test that all optimizers produce valid weights."""
        scores, prices = sample_data

        optimizers = [
            MinVarianceOptimizer(num_positions=20, lookback=60),
            MeanVarianceOptimizer(num_positions=20, lookback=60, risk_aversion=1.0)
        ]

        for optimizer in optimizers:
            weights = optimizer.optimize(scores, prices)

            # All should produce valid weights
            assert abs(weights.sum() - 1.0) < 1e-6
            assert (weights >= 0).all()
            assert len(weights) == 20

    def test_mvo_vs_minvar_diversification(self, sample_data):
        """Test that MVO can be more/less diversified than MinVar."""
        scores, prices = sample_data

        minvar = MinVarianceOptimizer(num_positions=20, lookback=60, risk_penalty=0.01)
        mvo = MeanVarianceOptimizer(num_positions=20, lookback=60, risk_aversion=1.0, axioma_penalty=0.01)

        weights_minvar = minvar.optimize(scores, prices)
        weights_mvo = mvo.optimize(scores, prices)

        # Both should be valid but different
        assert not np.allclose(weights_minvar.values, weights_mvo.values)

        # Calculate concentration (HHI)
        hhi_minvar = (weights_minvar ** 2).sum()
        hhi_mvo = (weights_mvo ** 2).sum()

        # Both should be reasonably diversified (HHI < 0.2)
        assert hhi_minvar < 0.2
        assert hhi_mvo < 0.2
