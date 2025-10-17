"""
Portfolio Rebalancer

Threshold-based rebalancing logic that determines when and how
to rebalance the portfolio based on drift from target weights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RebalanceDecision:
    """
    Result of a rebalance check.

    Attributes
    ----------
    should_rebalance : bool
        Whether rebalancing is recommended
    reason : str
        Reason for rebalancing (or not)
    drift : float
        Maximum weight drift from target
    trades : pd.Series
        Recommended trades (ticker -> weight change)
    """
    should_rebalance: bool
    reason: str
    drift: float
    trades: pd.Series


class ThresholdRebalancer:
    """
    Threshold-based portfolio rebalancer.

    Triggers rebalancing when:
    1. Any position drifts beyond threshold from target
    2. Time since last rebalance exceeds maximum
    3. Portfolio has too many/few positions

    Parameters
    ----------
    drift_threshold : float
        Maximum allowed drift from target weight (e.g., 0.05 = 5%)
    min_trade_size : float
        Minimum trade size as fraction of portfolio (ignore tiny trades)
    max_days_between : int, optional
        Maximum days between rebalances (force rebalance if exceeded)
    target_num_positions : int, optional
        Target number of positions (rebalance if deviate too much)
    position_tolerance : int
        Allowed deviation from target number of positions
    """

    def __init__(self,
                 drift_threshold: float = 0.05,
                 min_trade_size: float = 0.01,
                 max_days_between: Optional[int] = None,
                 target_num_positions: Optional[int] = None,
                 position_tolerance: int = 3):

        self.drift_threshold = drift_threshold
        self.min_trade_size = min_trade_size
        self.max_days_between = max_days_between
        self.target_num_positions = target_num_positions
        self.position_tolerance = position_tolerance

        # Track rebalance history
        self.last_rebalance_date = None
        self.rebalance_count = 0

    def check_rebalance(self,
                        current_weights: pd.Series,
                        target_weights: pd.Series,
                        current_date: pd.Timestamp) -> RebalanceDecision:
        """
        Check if rebalancing is needed.

        Parameters
        ----------
        current_weights : pd.Series
            Current portfolio weights (after market moves)
        target_weights : pd.Series
            Target portfolio weights from optimizer
        current_date : pd.Timestamp
            Current date

        Returns
        -------
        RebalanceDecision
            Decision about whether to rebalance
        """
        # Align indices (some positions may have been closed)
        all_tickers = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_tickers, fill_value=0.0)
        target = target_weights.reindex(all_tickers, fill_value=0.0)

        # Calculate trades needed
        trades = target - current

        # Calculate maximum drift
        drift = (current - target).abs().max()

        # Check 1: Drift threshold
        if drift > self.drift_threshold:
            # Only rebalance positions that exceed min_trade_size
            significant_trades = trades[trades.abs() > self.min_trade_size]

            if len(significant_trades) > 0:
                return RebalanceDecision(
                    should_rebalance=True,
                    reason=f"Drift {drift:.2%} exceeds threshold {self.drift_threshold:.2%}",
                    drift=drift,
                    trades=significant_trades
                )

        # Check 2: Time since last rebalance
        if self.max_days_between is not None and self.last_rebalance_date is not None:
            days_since = (current_date - self.last_rebalance_date).days

            if days_since >= self.max_days_between:
                significant_trades = trades[trades.abs() > self.min_trade_size]

                return RebalanceDecision(
                    should_rebalance=True,
                    reason=f"Time threshold: {days_since} days since last rebalance",
                    drift=drift,
                    trades=significant_trades
                )

        # Check 3: Number of positions
        if self.target_num_positions is not None:
            current_num_positions = (current > 0.001).sum()
            target_num_positions = (target > 0.001).sum()

            position_diff = abs(current_num_positions - self.target_num_positions)

            if position_diff > self.position_tolerance:
                significant_trades = trades[trades.abs() > self.min_trade_size]

                return RebalanceDecision(
                    should_rebalance=True,
                    reason=f"Position count: {current_num_positions} vs target {self.target_num_positions}",
                    drift=drift,
                    trades=significant_trades
                )

        # No rebalancing needed
        return RebalanceDecision(
            should_rebalance=False,
            reason=f"Drift {drift:.2%} within threshold",
            drift=drift,
            trades=pd.Series(dtype=float)
        )

    def execute_rebalance(self,
                         current_date: pd.Timestamp) -> None:
        """
        Record that rebalancing was executed.

        Parameters
        ----------
        current_date : pd.Timestamp
            Date of rebalance
        """
        self.last_rebalance_date = current_date
        self.rebalance_count += 1
        logger.info(f"Rebalance #{self.rebalance_count} executed on {current_date.date()}")

    def get_rebalance_stats(self) -> Dict:
        """
        Get statistics about rebalancing history.

        Returns
        -------
        dict
            Rebalancing statistics
        """
        return {
            'total_rebalances': self.rebalance_count,
            'last_rebalance_date': self.last_rebalance_date,
            'drift_threshold': self.drift_threshold,
            'min_trade_size': self.min_trade_size
        }


class PeriodicRebalancer:
    """
    Simple periodic rebalancer.

    Rebalances on a fixed schedule (e.g., monthly, quarterly).

    Parameters
    ----------
    frequency : str
        Rebalancing frequency: 'daily', 'weekly', 'monthly', 'quarterly'
    min_trade_size : float
        Minimum trade size to execute
    """

    def __init__(self,
                 frequency: str = 'monthly',
                 min_trade_size: float = 0.01):

        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly']
        if frequency not in valid_frequencies:
            raise ValueError(f"frequency must be one of {valid_frequencies}")

        self.frequency = frequency
        self.min_trade_size = min_trade_size

        self.last_rebalance_date = None
        self.rebalance_count = 0

    def check_rebalance(self,
                        current_weights: pd.Series,
                        target_weights: pd.Series,
                        current_date: pd.Timestamp) -> RebalanceDecision:
        """
        Check if it's time to rebalance based on schedule.

        Parameters
        ----------
        current_weights : pd.Series
            Current portfolio weights
        target_weights : pd.Series
            Target portfolio weights
        current_date : pd.Timestamp
            Current date

        Returns
        -------
        RebalanceDecision
            Decision about whether to rebalance
        """
        # Calculate trades
        all_tickers = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_tickers, fill_value=0.0)
        target = target_weights.reindex(all_tickers, fill_value=0.0)
        trades = target - current
        drift = (current - target).abs().max()

        # Check if it's time based on frequency
        should_rebalance = self._is_rebalance_date(current_date)

        if should_rebalance:
            significant_trades = trades[trades.abs() > self.min_trade_size]

            return RebalanceDecision(
                should_rebalance=True,
                reason=f"Scheduled {self.frequency} rebalance",
                drift=drift,
                trades=significant_trades
            )
        else:
            return RebalanceDecision(
                should_rebalance=False,
                reason=f"Not a scheduled rebalance date (frequency={self.frequency})",
                drift=drift,
                trades=pd.Series(dtype=float)
            )

    def _is_rebalance_date(self, current_date: pd.Timestamp) -> bool:
        """Check if current date is a rebalance date."""
        if self.last_rebalance_date is None:
            return True

        if self.frequency == 'daily':
            return True

        elif self.frequency == 'weekly':
            # Rebalance every Monday (weekday == 0)
            return current_date.weekday() == 0

        elif self.frequency == 'monthly':
            # Rebalance on first business day of month
            return (current_date.month != self.last_rebalance_date.month or
                    current_date.year != self.last_rebalance_date.year)

        elif self.frequency == 'quarterly':
            # Rebalance on first business day of quarter
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (self.last_rebalance_date.month - 1) // 3

            return (current_quarter != last_quarter or
                    current_date.year != self.last_rebalance_date.year)

        return False

    def execute_rebalance(self, current_date: pd.Timestamp) -> None:
        """Record that rebalancing was executed."""
        self.last_rebalance_date = current_date
        self.rebalance_count += 1
        logger.info(f"Rebalance #{self.rebalance_count} executed on {current_date.date()}")


class HybridRebalancer:
    """
    Hybrid rebalancer combining threshold and periodic approaches.

    Rebalances if either:
    1. Drift exceeds threshold (opportunistic)
    2. Scheduled period arrives (periodic)

    Parameters
    ----------
    drift_threshold : float
        Drift threshold for opportunistic rebalancing
    frequency : str
        Periodic rebalancing frequency
    min_trade_size : float
        Minimum trade size
    """

    def __init__(self,
                 drift_threshold: float = 0.10,
                 frequency: str = 'monthly',
                 min_trade_size: float = 0.01):

        self.threshold_rebalancer = ThresholdRebalancer(
            drift_threshold=drift_threshold,
            min_trade_size=min_trade_size
        )

        self.periodic_rebalancer = PeriodicRebalancer(
            frequency=frequency,
            min_trade_size=min_trade_size
        )

    def check_rebalance(self,
                        current_weights: pd.Series,
                        target_weights: pd.Series,
                        current_date: pd.Timestamp) -> RebalanceDecision:
        """
        Check if rebalancing is needed (threshold or periodic).

        Parameters
        ----------
        current_weights : pd.Series
            Current portfolio weights
        target_weights : pd.Series
            Target portfolio weights
        current_date : pd.Timestamp
            Current date

        Returns
        -------
        RebalanceDecision
            Decision about whether to rebalance
        """
        # Check threshold first (more urgent)
        threshold_decision = self.threshold_rebalancer.check_rebalance(
            current_weights, target_weights, current_date
        )

        if threshold_decision.should_rebalance:
            return threshold_decision

        # Check periodic schedule
        periodic_decision = self.periodic_rebalancer.check_rebalance(
            current_weights, target_weights, current_date
        )

        return periodic_decision

    def execute_rebalance(self, current_date: pd.Timestamp) -> None:
        """Record rebalance in both rebalancers."""
        self.threshold_rebalancer.execute_rebalance(current_date)
        self.periodic_rebalancer.execute_rebalance(current_date)
