"""
Unit tests for Portfolio Optimizer
"""

import pytest
import pandas as pd
import numpy as np
from src.portfolio.optimizer import (
    SimpleOptimizer,
    RankBasedOptimizer,
    MinVarianceOptimizer
)


@pytest.fixture
def sample_factor_scores():
    """Create sample factor scores for testing."""
    np.random.seed(42)

    scores = pd.Series({
        f'ETF_{i:03d}': np.random.randn()
        for i in range(100)
    })

    return scores


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')

    tickers = [f'ETF_{i:03d}' for i in range(100)]

    # Generate random walk prices
    returns = np.random.randn(252, 100) * 0.01
    prices = pd.DataFrame(
        100 * (1 + returns).cumprod(axis=0),
        columns=tickers,
        index=dates
    )

    return prices


class TestSimpleOptimizer:
    """Test SimpleOptimizer class."""

    @pytest.mark.unit
    def test_basic_optimization(self, sample_factor_scores):
        """Test basic equal-weight optimization."""
        optimizer = SimpleOptimizer(num_positions=30)
        weights = optimizer.optimize(sample_factor_scores)

        # Check basic properties
        assert len(weights) == 30
        assert abs(weights.sum() - 1.0) < 1e-6  # Sums to 1
        assert (weights >= 0).all()  # All positive
        assert (weights <= 1.0).all()  # All <= 1

        # Equal weight
        expected_weight = 1.0 / 30
        assert abs(weights.mean() - expected_weight) < 1e-6

    @pytest.mark.unit
    def test_min_score_filter(self, sample_factor_scores):
        """Test minimum score filtering."""
        optimizer = SimpleOptimizer(num_positions=30, min_score=0.5)
        weights = optimizer.optimize(sample_factor_scores)

        # All selected ETFs should have score >= 0.5
        for ticker in weights.index:
            assert sample_factor_scores[ticker] >= 0.5

    @pytest.mark.unit
    def test_fewer_etfs_than_positions(self):
        """Test when fewer ETFs available than target positions."""
        scores = pd.Series({'A': 1.0, 'B': 0.5, 'C': 0.3})

        optimizer = SimpleOptimizer(num_positions=10)
        weights = optimizer.optimize(scores)

        # Should only have 3 positions
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-6

    @pytest.mark.unit
    def test_weight_constraints(self, sample_factor_scores):
        """Test max/min weight constraints."""
        optimizer = SimpleOptimizer(
            num_positions=30,
            max_weight=0.05,  # 5% max per position
            min_weight=0.02   # 2% min per position
        )

        weights = optimizer.optimize(sample_factor_scores)

        # Check constraints
        assert (weights >= 0.02).all()
        assert (weights <= 0.05).all()
        assert abs(weights.sum() - 1.0) < 1e-3

    @pytest.mark.unit
    def test_empty_scores(self):
        """Test error handling with empty scores."""
        optimizer = SimpleOptimizer(num_positions=30)

        with pytest.raises(ValueError, match="No factor scores"):
            optimizer.optimize(pd.Series(dtype=float))

    @pytest.mark.unit
    def test_position_info(self, sample_factor_scores):
        """Test position info generation."""
        optimizer = SimpleOptimizer(num_positions=30)
        weights = optimizer.optimize(sample_factor_scores)

        info = optimizer.get_position_info(weights, sample_factor_scores)

        assert len(info) == 30
        assert 'weight' in info.columns
        assert 'factor_score' in info.columns
        assert (info['weight'] > 0).all()


class TestRankBasedOptimizer:
    """Test RankBasedOptimizer class."""

    @pytest.mark.unit
    def test_linear_weighting(self, sample_factor_scores):
        """Test linear rank-based weighting."""
        optimizer = RankBasedOptimizer(
            num_positions=30,
            weighting_scheme='linear'
        )

        weights = optimizer.optimize(sample_factor_scores)

        # Check properties
        assert len(weights) == 30
        assert abs(weights.sum() - 1.0) < 1e-6

        # Top ranked should have highest weight
        top_etf = weights.idxmax()
        top_score = sample_factor_scores[top_etf]

        # Verify top etf has one of the highest scores
        assert top_score >= sample_factor_scores.nlargest(30).min()

        # Weights should be decreasing by rank
        sorted_weights = weights.sort_values(ascending=False)
        diffs = sorted_weights.diff().dropna()
        assert (diffs <= 0).all()  # Non-increasing

    @pytest.mark.unit
    def test_exponential_weighting(self, sample_factor_scores):
        """Test exponential rank-based weighting."""
        optimizer = RankBasedOptimizer(
            num_positions=30,
            weighting_scheme='exponential'
        )

        weights = optimizer.optimize(sample_factor_scores)

        assert len(weights) == 30
        assert abs(weights.sum() - 1.0) < 1e-6

        # Exponential should have more concentration in top positions
        top_10_pct = weights.nlargest(10).sum()
        assert top_10_pct > 0.4  # Top 10 should have >40% weight

    @pytest.mark.unit
    def test_invalid_scheme(self, sample_factor_scores):
        """Test invalid weighting scheme."""
        optimizer = RankBasedOptimizer(
            num_positions=30,
            weighting_scheme='invalid'
        )

        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            optimizer.optimize(sample_factor_scores)


class TestMinVarianceOptimizer:
    """Test MinVarianceOptimizer class."""

    @pytest.mark.unit
    def test_basic_optimization(self, sample_factor_scores, sample_prices):
        """Test basic min variance optimization."""
        try:
            import cvxpy
        except ImportError:
            pytest.skip("cvxpy not installed")

        optimizer = MinVarianceOptimizer(num_positions=30, lookback=60)
        weights = optimizer.optimize(sample_factor_scores, sample_prices)

        # Check properties
        assert len(weights) <= 30
        assert abs(weights.sum() - 1.0) < 1e-3
        assert (weights >= 0).all()

    @pytest.mark.unit
    def test_concentrates_on_low_vol(self, sample_prices):
        """Test that min variance prefers low volatility assets."""
        try:
            import cvxpy
        except ImportError:
            pytest.skip("cvxpy not installed")

        # Create scores for 10 ETFs
        scores = pd.Series({
            f'ETF_{i:03d}': 1.0 for i in range(10)
        })

        # Modify prices so some ETFs have very low volatility
        prices = sample_prices[scores.index].copy()

        # Make first 3 ETFs low volatility
        for i in range(3):
            ticker = f'ETF_{i:03d}'
            prices[ticker] = prices[ticker].rolling(10).mean()  # Smooth out

        optimizer = MinVarianceOptimizer(num_positions=10, lookback=60)
        weights = optimizer.optimize(scores, prices)

        # Low vol ETFs should get higher weights
        low_vol_weight = weights[[f'ETF_{i:03d}' for i in range(3)]].sum()
        assert low_vol_weight > 0.3  # At least 30% in low vol

    @pytest.mark.unit
    def test_insufficient_price_data(self, sample_factor_scores):
        """Test handling of insufficient price data."""
        try:
            import cvxpy
        except ImportError:
            pytest.skip("cvxpy not installed")

        # Create very short price history
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        prices = pd.DataFrame(
            np.random.randn(30, 10) + 100,
            columns=[f'ETF_{i:03d}' for i in range(10)],
            index=dates
        )

        scores = pd.Series({f'ETF_{i:03d}': 1.0 for i in range(10)})

        optimizer = MinVarianceOptimizer(num_positions=10, lookback=60)

        # Should still work but log warning
        weights = optimizer.optimize(scores, prices)
        assert len(weights) == 10


class TestOptimizerIntegration:
    """Integration tests for optimizers."""

    @pytest.mark.integration
    def test_optimizer_comparison(self, sample_factor_scores, sample_prices):
        """Compare different optimizer outputs."""
        # Simple optimizer
        simple_opt = SimpleOptimizer(num_positions=30)
        simple_weights = simple_opt.optimize(sample_factor_scores)

        # Rank-based optimizer
        rank_opt = RankBasedOptimizer(num_positions=30)
        rank_weights = rank_opt.optimize(sample_factor_scores)

        # All should have 30 positions
        assert len(simple_weights) == 30
        assert len(rank_weights) == 30

        # Simple should be equal-weighted
        assert simple_weights.std() < rank_weights.std()

        # Rank-based should concentrate more in top picks
        assert rank_weights.max() > simple_weights.max()

    @pytest.mark.integration
    def test_score_selection_quality(self, sample_factor_scores):
        """Test that optimizers select high-scoring ETFs."""
        optimizer = SimpleOptimizer(num_positions=30)
        weights = optimizer.optimize(sample_factor_scores)

        # Selected ETFs should be in top 30 by score
        top_30_scores = sample_factor_scores.nlargest(30)

        for ticker in weights.index:
            assert ticker in top_30_scores.index


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
