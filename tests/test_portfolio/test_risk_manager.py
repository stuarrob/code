"""
Unit tests for Risk Manager
"""

import pytest
import pandas as pd
import numpy as np
from src.portfolio.risk_manager import (
    StopLossManager,
    VolatilityManager,
    RiskBudgetManager,
    RiskSignal
)


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    return pd.Series({
        'ETF_A': 100.0,
        'ETF_B': 150.0,
        'ETF_C': 200.0,
        'ETF_D': 120.0,
        'ETF_E': 90.0
    })


@pytest.fixture
def sample_weights():
    """Create sample portfolio weights."""
    return pd.Series({
        'ETF_A': 0.20,
        'ETF_B': 0.20,
        'ETF_C': 0.20,
        'ETF_D': 0.20,
        'ETF_E': 0.20
    })


@pytest.fixture
def sample_returns():
    """Create sample return history."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')

    returns = pd.Series(
        np.random.randn(252) * 0.01,  # 1% daily vol
        index=dates
    )

    return returns


class TestStopLossManager:
    """Test StopLossManager class."""

    @pytest.mark.unit
    def test_no_risk_violation(self, sample_prices, sample_weights):
        """Test when no risk violations occur."""
        manager = StopLossManager(
            position_stop_loss=0.15,
            portfolio_stop_loss=0.20
        )

        # Add positions at current prices
        for ticker, price in sample_prices.items():
            manager.add_position(ticker, price)

        signal = manager.check_risk(
            current_prices=sample_prices,
            weights=sample_weights,
            portfolio_value=100000
        )

        assert signal.action == 'hold'
        assert len(signal.positions_to_close) == 0
        assert signal.severity == 'low'

    @pytest.mark.unit
    def test_position_stop_loss(self, sample_prices, sample_weights):
        """Test position stop-loss trigger."""
        manager = StopLossManager(position_stop_loss=0.15)

        # Add positions at higher entry prices
        for ticker in sample_prices.index:
            manager.add_position(ticker, sample_prices[ticker] * 1.30)  # 30% higher

        # Current prices represent 23% loss from entry
        signal = manager.check_risk(
            current_prices=sample_prices,
            weights=sample_weights,
            portfolio_value=100000
        )

        # Should trigger stop-loss
        assert signal.action == 'stop_loss'
        assert len(signal.positions_to_close) > 0
        assert signal.severity == 'medium'

    @pytest.mark.unit
    def test_portfolio_drawdown(self, sample_prices, sample_weights):
        """Test portfolio drawdown trigger."""
        manager = StopLossManager(
            position_stop_loss=0.50,  # High (won't trigger)
            portfolio_stop_loss=0.20
        )

        # Set high water mark
        manager.portfolio_high = 150000

        # Current value is 25% below high water mark
        signal = manager.check_risk(
            current_prices=sample_prices,
            weights=sample_weights,
            portfolio_value=112500
        )

        # Should trigger portfolio risk reduction
        assert signal.action == 'reduce'
        assert len(signal.positions_to_reduce) > 0
        assert 'drawdown' in signal.reason.lower()
        assert signal.severity == 'high'

    @pytest.mark.unit
    def test_trailing_stop(self, sample_prices, sample_weights):
        """Test trailing stop-loss."""
        manager = StopLossManager(
            position_stop_loss=0.10,
            trailing_stop=True,
            trailing_distance=0.10
        )

        # Add positions
        for ticker in sample_prices.index:
            manager.add_position(ticker, 100.0)

        # Prices rise (update high water marks)
        high_prices = sample_prices * 1.50

        manager.check_risk(
            current_prices=high_prices,
            weights=sample_weights,
            portfolio_value=150000
        )

        # Now prices fall 15% from high (but still above entry)
        current_prices = high_prices * 0.85

        signal = manager.check_risk(
            current_prices=current_prices,
            weights=sample_weights,
            portfolio_value=127500
        )

        # Should trigger trailing stop
        assert signal.action == 'stop_loss'
        assert len(signal.positions_to_close) > 0

    @pytest.mark.unit
    def test_add_and_close_positions(self):
        """Test adding and closing position tracking."""
        manager = StopLossManager()

        # Add positions
        manager.add_position('ETF_A', 100.0)
        manager.add_position('ETF_B', 150.0)

        assert 'ETF_A' in manager.entry_prices
        assert 'ETF_B' in manager.entry_prices

        # Close position
        manager.close_position('ETF_A')

        assert 'ETF_A' not in manager.entry_prices
        assert 'ETF_B' in manager.entry_prices

    @pytest.mark.unit
    def test_risk_metrics(self, sample_prices):
        """Test risk metrics calculation."""
        manager = StopLossManager(position_stop_loss=0.15)

        # Add positions with some losses
        manager.add_position('ETF_A', 120.0)  # Now 100, 16.7% loss
        manager.add_position('ETF_B', 150.0)  # No change
        manager.add_position('ETF_C', 200.0)  # No change

        manager.portfolio_high = 110000

        metrics = manager.get_risk_metrics(
            current_prices=sample_prices,
            portfolio_value=100000
        )

        assert 'portfolio_drawdown' in metrics
        assert 'position_losses' in metrics
        assert 'max_position_loss' in metrics
        assert 'positions_at_risk' in metrics

        # Check that ETF_A shows loss
        assert metrics['position_losses']['ETF_A'] > 0.15


class TestVolatilityManager:
    """Test VolatilityManager class."""

    @pytest.mark.unit
    def test_exposure_calculation(self):
        """Test exposure calculation based on volatility."""
        manager = VolatilityManager(
            target_volatility=0.15,
            lookback=60
        )

        # Generate returns with 20% volatility
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.0126)  # 20% annual

        exposure = manager.calculate_exposure(returns)

        # Should recommend reducing exposure (15% / 20% = 0.75)
        assert 0.7 < exposure < 0.9

    @pytest.mark.unit
    def test_low_volatility_increase_exposure(self):
        """Test that low volatility increases exposure."""
        manager = VolatilityManager(
            target_volatility=0.15,
            lookback=60,
            max_leverage=1.0
        )

        # Generate returns with 10% volatility
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.0063)  # 10% annual

        exposure = manager.calculate_exposure(returns)

        # Should recommend 100% exposure (capped at max)
        assert exposure == 1.0

    @pytest.mark.unit
    def test_adjust_weights(self, sample_weights):
        """Test weight adjustment based on volatility."""
        manager = VolatilityManager(
            target_volatility=0.15,
            lookback=60
        )

        # High volatility returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.02)  # 30% annual

        adjusted_weights = manager.adjust_weights(sample_weights, returns)

        # Weights should be scaled down
        assert adjusted_weights.sum() < sample_weights.sum()
        assert (adjusted_weights < sample_weights).all()

    @pytest.mark.unit
    def test_insufficient_data(self, sample_weights):
        """Test handling of insufficient data."""
        manager = VolatilityManager(target_volatility=0.15, lookback=60)

        # Only 30 days of data
        returns = pd.Series(np.random.randn(30) * 0.01)

        # Should return max leverage (default)
        exposure = manager.calculate_exposure(returns)
        assert exposure == manager.max_leverage


class TestRiskBudgetManager:
    """Test RiskBudgetManager class."""

    @pytest.mark.unit
    def test_no_violation(self, sample_weights):
        """Test when risk budget is not violated."""
        manager = RiskBudgetManager(max_position_risk=0.25)

        # Create balanced covariance matrix
        cov_matrix = pd.DataFrame(
            np.eye(5) * 0.01,  # Low correlation
            index=sample_weights.index,
            columns=sample_weights.index
        )

        signal = manager.check_risk_budget(sample_weights, cov_matrix)

        assert signal.action == 'hold'
        assert len(signal.positions_to_reduce) == 0

    @pytest.mark.unit
    def test_concentrated_risk(self):
        """Test when one position has concentrated risk."""
        # Create concentrated weights
        weights = pd.Series({
            'ETF_A': 0.50,  # Very large position
            'ETF_B': 0.20,
            'ETF_C': 0.15,
            'ETF_D': 0.10,
            'ETF_E': 0.05
        })

        manager = RiskBudgetManager(max_position_risk=0.25)

        # ETF_A is also highly volatile
        cov_matrix = pd.DataFrame(
            np.eye(5) * 0.01,
            index=weights.index,
            columns=weights.index
        )
        cov_matrix.loc['ETF_A', 'ETF_A'] = 0.04  # 2x more volatile

        signal = manager.check_risk_budget(weights, cov_matrix)

        # Should recommend reducing ETF_A
        assert signal.action == 'reduce'
        assert 'ETF_A' in signal.positions_to_reduce

    @pytest.mark.unit
    def test_zero_volatility(self, sample_weights):
        """Test handling of zero volatility."""
        manager = RiskBudgetManager(max_position_risk=0.25)

        # Zero covariance matrix
        cov_matrix = pd.DataFrame(
            np.zeros((5, 5)),
            index=sample_weights.index,
            columns=sample_weights.index
        )

        signal = manager.check_risk_budget(sample_weights, cov_matrix)

        # Should handle gracefully
        assert signal.action == 'hold'


class TestRiskManagerIntegration:
    """Integration tests for risk managers."""

    @pytest.mark.integration
    def test_combined_risk_checks(self, sample_prices, sample_weights):
        """Test using multiple risk managers together."""
        stop_loss = StopLossManager(position_stop_loss=0.15)
        vol_manager = VolatilityManager(target_volatility=0.15)

        # Add positions
        for ticker, price in sample_prices.items():
            stop_loss.add_position(ticker, price)

        # Check stop-loss
        sl_signal = stop_loss.check_risk(
            current_prices=sample_prices,
            weights=sample_weights,
            portfolio_value=100000
        )

        # Generate returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)

        # Check volatility
        exposure = vol_manager.calculate_exposure(returns)

        # Both should work independently
        assert sl_signal is not None
        assert 0 < exposure <= 1.0

    @pytest.mark.integration
    def test_risk_lifecycle(self, sample_prices, sample_weights):
        """Test risk management over time."""
        manager = StopLossManager(position_stop_loss=0.15)

        # Initialize positions
        for ticker, price in sample_prices.items():
            manager.add_position(ticker, price * 1.10)  # 10% below entry

        portfolio_value = 100000

        # Simulate price movements
        for day in range(20):
            # Prices drift down
            current_prices = sample_prices * (1 - day * 0.01)

            signal = manager.check_risk(
                current_prices=current_prices,
                weights=sample_weights,
                portfolio_value=portfolio_value
            )

            if signal.action == 'stop_loss':
                # Close triggered positions
                for ticker in signal.positions_to_close:
                    manager.close_position(ticker)
                    sample_weights = sample_weights.drop(ticker)

                # Renormalize weights
                if len(sample_weights) > 0:
                    sample_weights = sample_weights / sample_weights.sum()

            portfolio_value *= 0.99  # 1% daily loss

        # Should have closed some positions
        assert len(manager.entry_prices) < 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
