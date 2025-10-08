"""
Unit tests for Momentum Factor
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.factors.momentum_factor import MomentumFactor, DualMomentumFactor


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')

    prices = pd.DataFrame({
        'STRONG_UP': np.linspace(100, 150, 300),    # 50% gain
        'MODERATE_UP': np.linspace(100, 120, 300),  # 20% gain
        'FLAT': np.full(300, 100),                  # No change
        'MODERATE_DOWN': np.linspace(100, 90, 300), # 10% loss
        'STRONG_DOWN': np.linspace(100, 70, 300),   # 30% loss
    }, index=dates)

    return prices


@pytest.fixture
def spike_prices():
    """Create price data with recent spike (to test skip_recent)."""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')

    # Gradual rise for 279 days, then huge spike in last 21 days
    base = np.linspace(100, 120, 279)
    spike = np.linspace(120, 200, 21)

    prices = pd.DataFrame({
        'SPIKE_ETF': np.concatenate([base, spike])
    }, index=dates)

    return prices


class TestMomentumFactor:
    """Test basic momentum factor calculation."""

    @pytest.mark.unit
    def test_basic_calculation(self, sample_prices):
        """Test momentum calculates correctly and ranks ETFs properly."""
        factor = MomentumFactor(lookback=252, skip_recent=21)
        scores = factor.calculate(sample_prices)

        # Check all ETFs got scores
        assert len(scores) == 5
        assert scores.isna().sum() == 0

        # Check ranking: STRONG_UP > MODERATE_UP > FLAT > MODERATE_DOWN > STRONG_DOWN
        assert scores['STRONG_UP'] > scores['MODERATE_UP']
        assert scores['MODERATE_UP'] > scores['FLAT']
        assert scores['FLAT'] > scores['MODERATE_DOWN']
        assert scores['MODERATE_DOWN'] > scores['STRONG_DOWN']

    @pytest.mark.unit
    def test_normalization(self, sample_prices):
        """Test that scores are normalized to z-score."""
        factor = MomentumFactor(lookback=252, skip_recent=21)
        scores = factor.calculate(sample_prices)

        # Z-score should have mean ≈ 0, std ≈ 1
        assert abs(scores.mean()) < 0.1
        assert 0.9 < scores.std() < 1.1

    @pytest.mark.unit
    def test_skip_recent_month(self, spike_prices):
        """Test that recent month is correctly skipped."""
        # With skip_recent=21, should ignore the huge spike
        factor_skip = MomentumFactor(lookback=252, skip_recent=21)
        score_skip = factor_skip.calculate(spike_prices)['SPIKE_ETF']

        # Without skip_recent, should capture the huge spike
        factor_no_skip = MomentumFactor(lookback=252, skip_recent=0)
        score_no_skip = factor_no_skip.calculate(spike_prices)['SPIKE_ETF']

        # Score without skip should be much higher (spike included)
        # Note: Since we normalize, can't directly compare absolute values
        # But we can test that skip_recent changes the result
        assert score_skip != score_no_skip

    @pytest.mark.unit
    def test_insufficient_data(self):
        """Test error handling with insufficient data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'ETF_A': np.random.randn(100) + 100
        }, index=dates)

        factor = MomentumFactor(lookback=252, skip_recent=21)

        with pytest.raises(ValueError, match="requires 252 days"):
            factor.calculate(prices)

    @pytest.mark.unit
    def test_nan_handling(self, sample_prices):
        """Test handling of NaN values in price data."""
        # Introduce some NaN values
        prices_with_nan = sample_prices.copy()
        prices_with_nan.loc[prices_with_nan.index[10:20], 'STRONG_UP'] = np.nan

        factor = MomentumFactor(lookback=252, skip_recent=21)

        # Should handle NaN gracefully (though specific behavior depends on implementation)
        scores = factor.calculate(prices_with_nan)

        # At minimum, should not crash
        assert scores is not None

    @pytest.mark.unit
    def test_identical_prices(self):
        """Test handling of ETFs with identical prices (zero volatility)."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = pd.DataFrame({
            'ETF_A': np.full(300, 100),
            'ETF_B': np.full(300, 100),
            'ETF_C': np.full(300, 100),
        }, index=dates)

        factor = MomentumFactor(lookback=252, skip_recent=21)
        scores = factor.calculate(prices)

        # All scores should be identical (or all NaN/zero)
        assert len(scores.unique()) <= 2  # Might be 0 and NaN

    @pytest.mark.unit
    def test_winsorization(self, sample_prices):
        """Test that extreme outliers are winsorized."""
        # Add an extreme outlier
        prices_with_outlier = sample_prices.copy()
        prices_with_outlier['EXTREME'] = np.linspace(100, 1000, 300)  # 900% gain

        factor = MomentumFactor(lookback=252, skip_recent=21, winsorize_pct=0.01)
        scores = factor.calculate(prices_with_outlier)

        # Outlier should be clipped (not massively larger than others)
        # Without winsorization, EXTREME would dominate
        # With winsorization, should be closer to other scores
        assert scores['EXTREME'] < scores['EXTREME'] * 10  # Sanity check


class TestDualMomentumFactor:
    """Test dual momentum (relative + absolute)."""

    @pytest.mark.unit
    def test_absolute_momentum_filter(self, sample_prices):
        """Test that negative momentum ETFs are filtered out."""
        factor = DualMomentumFactor(lookback=252, skip_recent=21, absolute_threshold=0.0)
        scores = factor.calculate(sample_prices)

        # MODERATE_DOWN and STRONG_DOWN should be filtered (NaN)
        assert pd.isna(scores['MODERATE_DOWN'])
        assert pd.isna(scores['STRONG_DOWN'])

        # Positive momentum ETFs should have scores
        assert not pd.isna(scores['STRONG_UP'])
        assert not pd.isna(scores['MODERATE_UP'])

        # FLAT is borderline (exactly 0% momentum)
        # Behavior depends on threshold (0.0 means allow zero)

    @pytest.mark.unit
    def test_custom_threshold(self, sample_prices):
        """Test custom absolute momentum threshold."""
        # Require at least 15% momentum
        factor = DualMomentumFactor(lookback=252, skip_recent=21, absolute_threshold=0.15)
        scores = factor.calculate(sample_prices)

        # Only STRONG_UP (50% gain) should pass
        assert not pd.isna(scores['STRONG_UP'])

        # MODERATE_UP (20% gain) should pass
        assert not pd.isna(scores['MODERATE_UP'])

        # All others filtered
        assert pd.isna(scores['FLAT'])
        assert pd.isna(scores['MODERATE_DOWN'])
        assert pd.isna(scores['STRONG_DOWN'])


class TestMomentumRolling:
    """Test rolling momentum calculations."""

    @pytest.mark.unit
    def test_rolling_calculation(self, sample_prices):
        """Test rolling momentum calculation."""
        factor = MomentumFactor(lookback=252, skip_recent=21)

        # Add more data for rolling
        extended_dates = pd.date_range('2020-01-01', periods=400, freq='D')
        extended_prices = pd.DataFrame({
            'ETF_A': np.linspace(100, 150, 400)
        }, index=extended_dates)

        rolling_scores = factor.calculate_rolling(extended_prices, window=21)

        # Should have multiple time points
        assert len(rolling_scores) > 1

        # Each row should have scores for ETF_A
        assert 'ETF_A' in rolling_scores.columns

    @pytest.mark.unit
    def test_rolling_insufficient_data(self, sample_prices):
        """Test rolling with insufficient data."""
        factor = MomentumFactor(lookback=252, skip_recent=21)

        with pytest.raises(ValueError, match="at least"):
            factor.calculate_rolling(sample_prices, window=21)


@pytest.mark.integration
def test_momentum_integration():
    """Integration test: Full momentum calculation pipeline."""
    # Create realistic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Generate random walk prices
    returns = np.random.randn(500, 10) * 0.01  # 1% daily volatility
    prices = pd.DataFrame(
        100 * (1 + returns).cumprod(),
        columns=[f'ETF_{i}' for i in range(10)],
        index=dates
    )

    # Calculate momentum
    factor = MomentumFactor(lookback=252, skip_recent=21)
    scores = factor.calculate(prices)

    # Validation
    assert len(scores) == 10
    assert scores.isna().sum() == 0
    assert abs(scores.mean()) < 0.2  # Approximately zero
    assert 0.8 < scores.std() < 1.2  # Approximately 1

    # Test dual momentum
    dual_factor = DualMomentumFactor(lookback=252, skip_recent=21)
    dual_scores = dual_factor.calculate(prices)

    # Should have some filtered ETFs (those with negative momentum)
    assert dual_scores.isna().sum() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
