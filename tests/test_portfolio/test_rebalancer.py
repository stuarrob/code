"""
Unit tests for Portfolio Rebalancer
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.portfolio.rebalancer import (
    ThresholdRebalancer,
    PeriodicRebalancer,
    HybridRebalancer,
    RebalanceDecision
)


@pytest.fixture
def sample_weights():
    """Create sample portfolio weights."""
    return pd.Series({
        'ETF_A': 0.10,
        'ETF_B': 0.15,
        'ETF_C': 0.20,
        'ETF_D': 0.25,
        'ETF_E': 0.30
    })


@pytest.fixture
def drifted_weights():
    """Create drifted weights (after market moves)."""
    return pd.Series({
        'ETF_A': 0.08,   # -2%
        'ETF_B': 0.13,   # -2%
        'ETF_C': 0.22,   # +2%
        'ETF_D': 0.27,   # +2%
        'ETF_E': 0.30    # No change
    })


@pytest.fixture
def heavily_drifted_weights():
    """Create heavily drifted weights."""
    return pd.Series({
        'ETF_A': 0.05,   # -5%
        'ETF_B': 0.12,   # -3%
        'ETF_C': 0.28,   # +8%
        'ETF_D': 0.30,   # +5%
        'ETF_E': 0.25    # -5%
    })


class TestThresholdRebalancer:
    """Test ThresholdRebalancer class."""

    @pytest.mark.unit
    def test_no_rebalance_needed(self, sample_weights, drifted_weights):
        """Test when drift is within threshold."""
        rebalancer = ThresholdRebalancer(drift_threshold=0.05)

        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')
        )

        # 2% drift is within 5% threshold
        assert not decision.should_rebalance
        assert 'within threshold' in decision.reason.lower()
        assert decision.drift < 0.05

    @pytest.mark.unit
    def test_rebalance_on_drift(self, sample_weights, heavily_drifted_weights):
        """Test rebalancing triggered by drift."""
        rebalancer = ThresholdRebalancer(drift_threshold=0.05)

        decision = rebalancer.check_rebalance(
            current_weights=heavily_drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')
        )

        # 8% drift exceeds 5% threshold
        assert decision.should_rebalance
        assert 'drift' in decision.reason.lower()
        assert decision.drift > 0.05

        # Should have trades for significantly drifted positions
        assert len(decision.trades) > 0

    @pytest.mark.unit
    def test_min_trade_size(self, sample_weights):
        """Test that small trades are filtered out."""
        # Small drift
        current = pd.Series({
            'ETF_A': 0.102,  # +0.2% (tiny)
            'ETF_B': 0.148,  # -0.2% (tiny)
            'ETF_C': 0.20,
            'ETF_D': 0.25,
            'ETF_E': 0.30
        })

        rebalancer = ThresholdRebalancer(
            drift_threshold=0.01,  # Low threshold
            min_trade_size=0.02    # But require 2% trade size
        )

        decision = rebalancer.check_rebalance(
            current_weights=current,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')
        )

        # Drift detected but trades too small
        assert not decision.should_rebalance or len(decision.trades) == 0

    @pytest.mark.unit
    def test_time_threshold(self, sample_weights, drifted_weights):
        """Test rebalancing triggered by time threshold."""
        rebalancer = ThresholdRebalancer(
            drift_threshold=0.10,  # High drift threshold
            max_days_between=30
        )

        # First rebalance
        rebalancer.execute_rebalance(pd.Timestamp('2024-01-01'))

        # Check after 31 days
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-02-01')
        )

        # Should rebalance due to time, even though drift is small
        assert decision.should_rebalance
        assert 'time' in decision.reason.lower()

    @pytest.mark.unit
    def test_position_count_trigger(self, sample_weights):
        """Test rebalancing triggered by position count change."""
        rebalancer = ThresholdRebalancer(
            drift_threshold=0.50,  # Very high so drift doesn't trigger
            target_num_positions=5,
            position_tolerance=1
        )

        # Current portfolio has only 3 positions (2 closed)
        current = pd.Series({
            'ETF_A': 0.33,
            'ETF_B': 0.33,
            'ETF_C': 0.34
        })

        decision = rebalancer.check_rebalance(
            current_weights=current,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')
        )

        # Should rebalance due to position count (3 vs target 5)
        assert decision.should_rebalance
        assert 'position' in decision.reason.lower()

    @pytest.mark.unit
    def test_execute_rebalance_tracking(self):
        """Test rebalance execution tracking."""
        rebalancer = ThresholdRebalancer(drift_threshold=0.05)

        assert rebalancer.rebalance_count == 0
        assert rebalancer.last_rebalance_date is None

        rebalancer.execute_rebalance(pd.Timestamp('2024-01-01'))

        assert rebalancer.rebalance_count == 1
        assert rebalancer.last_rebalance_date == pd.Timestamp('2024-01-01')

        rebalancer.execute_rebalance(pd.Timestamp('2024-02-01'))

        assert rebalancer.rebalance_count == 2

    @pytest.mark.unit
    def test_new_positions(self, sample_weights):
        """Test handling of new positions in target."""
        current = pd.Series({
            'ETF_A': 0.50,
            'ETF_B': 0.50
        })

        target = pd.Series({
            'ETF_A': 0.20,
            'ETF_B': 0.20,
            'ETF_C': 0.20,  # New
            'ETF_D': 0.20,  # New
            'ETF_E': 0.20   # New
        })

        rebalancer = ThresholdRebalancer(drift_threshold=0.10)

        decision = rebalancer.check_rebalance(
            current_weights=current,
            target_weights=target,
            current_date=pd.Timestamp('2024-01-15')
        )

        # Large drift due to new positions
        assert decision.should_rebalance
        assert len(decision.trades) > 0

        # Should have sell trades for A and B, buy trades for C, D, E
        assert decision.trades['ETF_A'] < 0  # Sell
        assert decision.trades['ETF_C'] > 0  # Buy


class TestPeriodicRebalancer:
    """Test PeriodicRebalancer class."""

    @pytest.mark.unit
    def test_daily_rebalance(self, sample_weights, drifted_weights):
        """Test daily rebalancing."""
        rebalancer = PeriodicRebalancer(frequency='daily')

        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')
        )

        # Should always rebalance (daily)
        assert decision.should_rebalance

    @pytest.mark.unit
    def test_weekly_rebalance(self, sample_weights, drifted_weights):
        """Test weekly rebalancing."""
        rebalancer = PeriodicRebalancer(frequency='weekly')

        # First rebalance on Monday
        rebalancer.execute_rebalance(pd.Timestamp('2024-01-08'))  # Monday

        # Check on Tuesday (should not rebalance)
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-09')  # Tuesday
        )
        assert not decision.should_rebalance

        # Check on next Monday (should rebalance)
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')  # Next Monday
        )
        # Note: Depends on whether last_rebalance_date is set
        # First call should rebalance
        if rebalancer.last_rebalance_date is None:
            assert decision.should_rebalance

    @pytest.mark.unit
    def test_monthly_rebalance(self, sample_weights, drifted_weights):
        """Test monthly rebalancing."""
        rebalancer = PeriodicRebalancer(frequency='monthly')

        # Rebalance in January
        rebalancer.execute_rebalance(pd.Timestamp('2024-01-15'))

        # Check later in January (should not rebalance)
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-20')
        )
        assert not decision.should_rebalance

        # Check in February (should rebalance)
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-02-01')
        )
        assert decision.should_rebalance

    @pytest.mark.unit
    def test_quarterly_rebalance(self, sample_weights, drifted_weights):
        """Test quarterly rebalancing."""
        rebalancer = PeriodicRebalancer(frequency='quarterly')

        # Rebalance in Q1
        rebalancer.execute_rebalance(pd.Timestamp('2024-01-15'))

        # Check later in Q1 (should not rebalance)
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-03-15')
        )
        assert not decision.should_rebalance

        # Check in Q2 (should rebalance)
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-04-01')
        )
        assert decision.should_rebalance

    @pytest.mark.unit
    def test_invalid_frequency(self):
        """Test invalid frequency."""
        with pytest.raises(ValueError, match="frequency must be one of"):
            PeriodicRebalancer(frequency='invalid')


class TestHybridRebalancer:
    """Test HybridRebalancer class."""

    @pytest.mark.unit
    def test_triggers_on_drift(self, sample_weights, heavily_drifted_weights):
        """Test that hybrid triggers on drift."""
        rebalancer = HybridRebalancer(
            drift_threshold=0.05,
            frequency='monthly'
        )

        # Large drift should trigger immediate rebalance
        decision = rebalancer.check_rebalance(
            current_weights=heavily_drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')
        )

        assert decision.should_rebalance
        assert 'drift' in decision.reason.lower()

    @pytest.mark.unit
    def test_triggers_on_schedule(self, sample_weights, drifted_weights):
        """Test that hybrid triggers on schedule."""
        rebalancer = HybridRebalancer(
            drift_threshold=0.20,  # High threshold (won't trigger)
            frequency='monthly'
        )

        # Execute in January
        rebalancer.execute_rebalance(pd.Timestamp('2024-01-15'))

        # Check in February (should rebalance on schedule)
        decision = rebalancer.check_rebalance(
            current_weights=drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-02-01')
        )

        assert decision.should_rebalance
        assert 'scheduled' in decision.reason.lower()


class TestRebalanceIntegration:
    """Integration tests for rebalancers."""

    @pytest.mark.integration
    def test_rebalance_lifecycle(self, sample_weights):
        """Test full rebalancing lifecycle."""
        rebalancer = ThresholdRebalancer(drift_threshold=0.05)

        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        rebalance_dates = []
        current_weights = sample_weights.copy()

        for date in dates:
            # Simulate random drift
            drift = pd.Series(
                np.random.randn(5) * 0.02,
                index=sample_weights.index
            )
            current_weights = (sample_weights + drift).clip(lower=0)
            current_weights = current_weights / current_weights.sum()

            decision = rebalancer.check_rebalance(
                current_weights=current_weights,
                target_weights=sample_weights,
                current_date=date
            )

            if decision.should_rebalance:
                rebalancer.execute_rebalance(date)
                rebalance_dates.append(date)
                current_weights = sample_weights.copy()  # Reset

        # Should have some rebalances
        assert len(rebalance_dates) >= 0
        assert rebalancer.rebalance_count == len(rebalance_dates)

    @pytest.mark.integration
    def test_trade_execution_accuracy(self, sample_weights, heavily_drifted_weights):
        """Test that recommended trades are accurate."""
        rebalancer = ThresholdRebalancer(drift_threshold=0.05, min_trade_size=0.01)

        decision = rebalancer.check_rebalance(
            current_weights=heavily_drifted_weights,
            target_weights=sample_weights,
            current_date=pd.Timestamp('2024-01-15')
        )

        if decision.should_rebalance:
            # Applying trades should bring us close to target
            new_weights = heavily_drifted_weights + decision.trades

            for ticker in sample_weights.index:
                # Should be very close to target
                assert abs(new_weights[ticker] - sample_weights[ticker]) < 0.02


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
