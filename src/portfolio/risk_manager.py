"""
Risk Manager

Monitors portfolio risk and implements stop-loss rules
to protect against drawdowns and excessive losses.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskSignal:
    """
    Risk management signal.

    Attributes
    ----------
    action : str
        Risk action: 'hold', 'reduce', 'close', 'stop_loss'
    positions_to_close : list
        List of tickers to close
    positions_to_reduce : dict
        Dict of ticker -> new_weight for positions to reduce
    reason : str
        Reason for risk action
    severity : str
        Risk severity: 'low', 'medium', 'high'
    """
    action: str
    positions_to_close: List[str]
    positions_to_reduce: Dict[str, float]
    reason: str
    severity: str


class StopLossManager:
    """
    Stop-loss risk manager.

    Monitors individual positions and portfolio for excessive losses.
    Triggers stop-loss when:
    1. Individual position drops below threshold
    2. Portfolio drawdown exceeds limit
    3. Volatility spikes above threshold

    **Dynamic VIX-Based Stop-Loss** (Recommended):
    When use_vix_adjustment=True, stop-loss thresholds adapt to market volatility:
    - VIX < 15 (low vol): Use 15% stop-loss (wider, allow normal fluctuations)
    - VIX 15-25 (normal): Use 12% stop-loss (standard)
    - VIX > 25 (high vol): Use 10% stop-loss (tighter, protect capital)

    This prevents premature stops in calm markets while protecting during volatility.

    Parameters
    ----------
    position_stop_loss : float
        Base maximum loss per position before stop-loss (e.g., 0.12 = 12%)
        Adjusted dynamically if use_vix_adjustment=True
    portfolio_stop_loss : float
        Maximum portfolio drawdown before reducing exposure
    trailing_stop : bool
        Use trailing stop-loss (follows high water mark)
    trailing_distance : float
        Distance below high water mark for trailing stop
    max_volatility : float, optional
        Maximum volatility before risk-off (e.g., 0.40 = 40% annual)
    use_vix_adjustment : bool
        Enable VIX-based dynamic stop-loss adjustment (default: True)
    vix_ticker : str
        Ticker symbol for VIX data (default: '^VIX')
    """

    def __init__(self,
                 position_stop_loss: float = 0.12,
                 portfolio_stop_loss: float = 0.20,
                 trailing_stop: bool = True,
                 trailing_distance: float = 0.10,
                 max_volatility: Optional[float] = 0.40,
                 use_vix_adjustment: bool = True,
                 vix_ticker: str = '^VIX'):

        self.base_position_stop_loss = position_stop_loss
        self.position_stop_loss = position_stop_loss
        self.portfolio_stop_loss = portfolio_stop_loss
        self.trailing_stop = trailing_stop
        self.trailing_distance = trailing_distance
        self.max_volatility = max_volatility
        self.use_vix_adjustment = use_vix_adjustment
        self.vix_ticker = vix_ticker

        # Track high water marks for trailing stops
        self.position_highs: Dict[str, float] = {}

        # Track VIX history for dynamic adjustment
        self.vix_history: Optional[pd.Series] = None
        self.portfolio_high: float = 0.0

        # Track entry prices for positions
        self.entry_prices: Dict[str, float] = {}

    def update_vix_stop_loss(self, vix_value: Optional[float] = None) -> float:
        """
        Update position stop-loss threshold based on current VIX level.

        VIX-based dynamic stop-loss adjustment:
        - VIX < 15: 15% stop-loss (low volatility, allow wider swings)
        - VIX 15-25: 12% stop-loss (normal volatility, standard protection)
        - VIX > 25: 10% stop-loss (high volatility, tighter protection)

        Parameters
        ----------
        vix_value : float, optional
            Current VIX value. If None and vix_history available, uses latest value.

        Returns
        -------
        float
            Adjusted stop-loss threshold
        """
        if not self.use_vix_adjustment:
            return self.base_position_stop_loss

        # Get VIX value
        if vix_value is None:
            if self.vix_history is not None and len(self.vix_history) > 0:
                vix_value = self.vix_history.iloc[-1]
            else:
                # No VIX data available, use base threshold
                logger.debug("No VIX data available, using base stop-loss threshold")
                return self.base_position_stop_loss

        # Adjust stop-loss based on VIX level
        if vix_value < 15:
            # Low volatility: wider stop-loss
            adjusted_stop = 0.15
            regime = "low volatility"
        elif vix_value <= 25:
            # Normal volatility: standard stop-loss
            adjusted_stop = 0.12
            regime = "normal volatility"
        else:
            # High volatility: tighter stop-loss
            adjusted_stop = 0.10
            regime = "high volatility"

        if adjusted_stop != self.position_stop_loss:
            logger.info(f"VIX-adjusted stop-loss: {adjusted_stop:.1%} (VIX={vix_value:.1f}, {regime})")

        self.position_stop_loss = adjusted_stop
        return adjusted_stop

    def set_vix_data(self, vix_series: pd.Series):
        """
        Set VIX historical data for dynamic stop-loss adjustment.

        Parameters
        ----------
        vix_series : pd.Series
            Historical VIX values with DatetimeIndex
        """
        self.vix_history = vix_series
        logger.info(f"VIX data set: {len(vix_series)} days, current VIX={vix_series.iloc[-1]:.1f}")

        # Update stop-loss immediately
        if self.use_vix_adjustment:
            self.update_vix_stop_loss()

    def check_risk(self,
                   current_prices: pd.Series,
                   weights: pd.Series,
                   portfolio_value: float) -> RiskSignal:
        """
        Check for risk violations and generate signals.

        Parameters
        ----------
        current_prices : pd.Series
            Current prices for all positions
        weights : pd.Series
            Current portfolio weights
        portfolio_value : float
            Current portfolio value

        Returns
        -------
        RiskSignal
            Risk management signal
        """
        # Update VIX-based stop-loss if enabled
        if self.use_vix_adjustment:
            self.update_vix_stop_loss()

        positions_to_close = []
        positions_to_reduce = {}

        # Update high water marks
        self.portfolio_high = max(self.portfolio_high, portfolio_value)

        for ticker in weights.index:
            if ticker not in self.position_highs:
                self.position_highs[ticker] = current_prices[ticker]

            self.position_highs[ticker] = max(
                self.position_highs[ticker],
                current_prices[ticker]
            )

        # Check 1: Individual position stop-loss
        for ticker in weights.index:
            if ticker not in self.entry_prices:
                # Initialize entry price
                self.entry_prices[ticker] = current_prices[ticker]
                continue

            current_price = current_prices[ticker]
            entry_price = self.entry_prices[ticker]

            # Calculate loss
            if self.trailing_stop:
                reference_price = self.position_highs[ticker]
                loss = (reference_price - current_price) / reference_price
            else:
                loss = (entry_price - current_price) / entry_price

            # Trigger stop-loss
            if loss > self.position_stop_loss:
                positions_to_close.append(ticker)
                logger.warning(
                    f"Stop-loss triggered for {ticker}: "
                    f"{loss:.2%} loss (threshold: {self.position_stop_loss:.2%})"
                )

        # Check 2: Portfolio drawdown
        portfolio_drawdown = (self.portfolio_high - portfolio_value) / self.portfolio_high

        if portfolio_drawdown > self.portfolio_stop_loss:
            # Reduce exposure on all positions
            reduction_factor = 0.5  # Reduce to 50% of current weight

            for ticker in weights.index:
                if ticker not in positions_to_close:
                    positions_to_reduce[ticker] = weights[ticker] * reduction_factor

            return RiskSignal(
                action='reduce',
                positions_to_close=positions_to_close,
                positions_to_reduce=positions_to_reduce,
                reason=f"Portfolio drawdown {portfolio_drawdown:.2%} exceeds {self.portfolio_stop_loss:.2%}",
                severity='high'
            )

        # Return signal
        if len(positions_to_close) > 0:
            return RiskSignal(
                action='stop_loss',
                positions_to_close=positions_to_close,
                positions_to_reduce={},
                reason=f"Stop-loss triggered on {len(positions_to_close)} positions",
                severity='medium'
            )

        # No risk violations
        return RiskSignal(
            action='hold',
            positions_to_close=[],
            positions_to_reduce={},
            reason='All risk checks passed',
            severity='low'
        )

    def add_position(self, ticker: str, entry_price: float) -> None:
        """
        Record a new position entry.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        entry_price : float
            Entry price
        """
        self.entry_prices[ticker] = entry_price
        self.position_highs[ticker] = entry_price
        logger.info(f"New position: {ticker} @ {entry_price:.2f}")

    def close_position(self, ticker: str) -> None:
        """
        Remove position from tracking.

        Parameters
        ----------
        ticker : str
            Ticker to close
        """
        if ticker in self.entry_prices:
            del self.entry_prices[ticker]
        if ticker in self.position_highs:
            del self.position_highs[ticker]
        logger.info(f"Position closed: {ticker}")

    def reset_portfolio_high(self) -> None:
        """Reset portfolio high water mark (e.g., after withdrawal)."""
        self.portfolio_high = 0.0

    def get_risk_metrics(self,
                        current_prices: pd.Series,
                        portfolio_value: float) -> Dict:
        """
        Calculate risk metrics for current portfolio.

        Parameters
        ----------
        current_prices : pd.Series
            Current prices
        portfolio_value : float
            Current portfolio value

        Returns
        -------
        dict
            Risk metrics
        """
        metrics = {}

        # Portfolio drawdown
        if self.portfolio_high > 0:
            metrics['portfolio_drawdown'] = (
                (self.portfolio_high - portfolio_value) / self.portfolio_high
            )
        else:
            metrics['portfolio_drawdown'] = 0.0

        # Position losses
        position_losses = {}
        for ticker, entry_price in self.entry_prices.items():
            if ticker in current_prices.index:
                current_price = current_prices[ticker]
                loss = (entry_price - current_price) / entry_price
                position_losses[ticker] = loss

        metrics['position_losses'] = position_losses
        metrics['max_position_loss'] = max(position_losses.values()) if position_losses else 0.0
        metrics['positions_at_risk'] = sum(
            1 for loss in position_losses.values()
            if loss > self.position_stop_loss * 0.8  # Within 80% of stop
        )

        return metrics


class VolatilityManager:
    """
    Volatility-based risk manager.

    Adjusts portfolio exposure based on realized volatility.
    Reduces exposure in high volatility environments.

    Parameters
    ----------
    target_volatility : float
        Target portfolio volatility (e.g., 0.15 = 15% annual)
    lookback : int
        Days of history for volatility calculation
    min_leverage : float
        Minimum leverage/exposure
    max_leverage : float
        Maximum leverage/exposure
    """

    def __init__(self,
                 target_volatility: float = 0.15,
                 lookback: int = 60,
                 min_leverage: float = 0.5,
                 max_leverage: float = 1.0):

        self.target_volatility = target_volatility
        self.lookback = lookback
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage

    def calculate_exposure(self,
                          returns: pd.Series) -> float:
        """
        Calculate recommended portfolio exposure based on volatility.

        Parameters
        ----------
        returns : pd.Series
            Historical portfolio returns

        Returns
        -------
        float
            Recommended exposure/leverage (0.5 = 50%, 1.0 = 100%)
        """
        if len(returns) < self.lookback:
            logger.warning(
                f"Insufficient history for volatility calculation: "
                f"{len(returns)}/{self.lookback} days"
            )
            return self.max_leverage

        # Calculate realized volatility
        recent_returns = returns.tail(self.lookback)
        realized_vol = recent_returns.std() * np.sqrt(252)

        # Calculate exposure: target_vol / realized_vol
        if realized_vol > 0:
            exposure = self.target_volatility / realized_vol
        else:
            exposure = self.max_leverage

        # Clip to min/max
        exposure = np.clip(exposure, self.min_leverage, self.max_leverage)

        logger.info(
            f"Volatility targeting: realized_vol={realized_vol:.2%}, "
            f"target_vol={self.target_volatility:.2%}, exposure={exposure:.2%}"
        )

        return exposure

    def adjust_weights(self,
                      weights: pd.Series,
                      returns: pd.Series) -> pd.Series:
        """
        Adjust portfolio weights based on volatility.

        Parameters
        ----------
        weights : pd.Series
            Target portfolio weights
        returns : pd.Series
            Historical portfolio returns

        Returns
        -------
        pd.Series
            Adjusted portfolio weights
        """
        exposure = self.calculate_exposure(returns)

        # Scale all weights by exposure
        adjusted_weights = weights * exposure

        logger.info(
            f"Weights adjusted by {exposure:.2%} exposure "
            f"(cash allocation: {1 - adjusted_weights.sum():.2%})"
        )

        return adjusted_weights


class RiskBudgetManager:
    """
    Risk budget manager.

    Allocates risk budget across positions based on their contribution
    to portfolio risk (volatility).

    Parameters
    ----------
    max_position_risk : float
        Maximum risk contribution per position
    risk_parity : bool
        Use risk parity approach (equal risk contribution)
    """

    def __init__(self,
                 max_position_risk: float = 0.25,
                 risk_parity: bool = False):

        self.max_position_risk = max_position_risk
        self.risk_parity = risk_parity

    def check_risk_budget(self,
                         weights: pd.Series,
                         cov_matrix: pd.DataFrame) -> RiskSignal:
        """
        Check if positions violate risk budget constraints.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
        cov_matrix : pd.DataFrame
            Covariance matrix of returns

        Returns
        -------
        RiskSignal
            Risk signal with recommended adjustments
        """
        # Calculate marginal contribution to risk
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        if portfolio_vol == 0:
            return RiskSignal(
                action='hold',
                positions_to_close=[],
                positions_to_reduce={},
                reason='Zero portfolio volatility',
                severity='low'
            )

        # Marginal contribution to risk (MCR)
        mcr = (cov_matrix @ weights) / portfolio_vol

        # Risk contribution (RC) = weight * MCR
        risk_contribution = weights * mcr
        risk_contribution_pct = risk_contribution / risk_contribution.sum()

        # Check violations
        violations = risk_contribution_pct[risk_contribution_pct > self.max_position_risk]

        if len(violations) > 0:
            # Scale down violators
            positions_to_reduce = {}

            for ticker in violations.index:
                # Target risk contribution
                target_rc = self.max_position_risk * 0.9  # 90% of max for buffer
                current_rc = risk_contribution_pct[ticker]

                # Scale weight proportionally
                scale_factor = target_rc / current_rc
                positions_to_reduce[ticker] = weights[ticker] * scale_factor

            return RiskSignal(
                action='reduce',
                positions_to_close=[],
                positions_to_reduce=positions_to_reduce,
                reason=f"{len(violations)} positions exceed risk budget",
                severity='medium'
            )

        return RiskSignal(
            action='hold',
            positions_to_close=[],
            positions_to_reduce={},
            reason='All positions within risk budget',
            severity='low'
        )
