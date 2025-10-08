"""
Portfolio Constraints Module

Implements various constraints for portfolio optimization:
- Position limits (max/min weight per asset)
- Cardinality constraints (max number of positions)
- Risk constraints (CVaR, drawdown limits)
- Sector/asset class diversification
- Turnover limits
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioConstraints:
    """Portfolio constraint enforcement and validation."""

    def __init__(
        self,
        max_positions: int = 20,
        max_weight: float = 0.15,
        min_weight: float = 0.02,
        max_sector_weight: Optional[float] = None,
        max_turnover: Optional[float] = None
    ):
        """
        Initialize portfolio constraints.

        Parameters
        ----------
        max_positions : int
            Maximum number of positions
        max_weight : float
            Maximum weight per position (0.15 = 15%)
        min_weight : float
            Minimum weight if position selected (0.02 = 2%)
        max_sector_weight : float, optional
            Maximum weight per sector (e.g., 0.30 = 30%)
        max_turnover : float, optional
            Maximum portfolio turnover per rebalance (e.g., 0.50 = 50%)
        """
        self.max_positions = max_positions
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_sector_weight = max_sector_weight
        self.max_turnover = max_turnover

    def validate_weights(
        self,
        weights: pd.Series,
        sectors: Optional[pd.Series] = None,
        previous_weights: Optional[pd.Series] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate portfolio weights against constraints.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
        sectors : pd.Series, optional
            Sector classification for each ticker
        previous_weights : pd.Series, optional
            Previous portfolio weights for turnover check

        Returns
        -------
        tuple
            (is_valid, violation_messages)
        """
        violations = []

        # Sum to 1 (fully invested)
        weight_sum = weights.sum()
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            violations.append(f"Weights sum to {weight_sum:.4f}, not 1.0")

        # Long-only
        if (weights < -1e-6).any():
            violations.append("Negative weights found (not long-only)")

        # Number of positions
        n_positions = (weights > 1e-6).sum()
        if n_positions > self.max_positions:
            violations.append(
                f"Too many positions: {n_positions} > {self.max_positions}"
            )

        # Max weight constraint
        max_w = weights.max()
        if max_w > self.max_weight + 1e-6:
            violations.append(
                f"Max weight {max_w:.4f} exceeds limit {self.max_weight}"
            )

        # Min weight constraint (for selected positions)
        selected = weights > 1e-6
        if selected.any():
            min_w = weights[selected].min()
            if min_w < self.min_weight - 1e-6:
                violations.append(
                    f"Min weight {min_w:.4f} below limit {self.min_weight}"
                )

        # Sector concentration (if provided)
        if sectors is not None and self.max_sector_weight is not None:
            sector_weights = weights.groupby(sectors).sum()
            max_sector_w = sector_weights.max()
            if max_sector_w > self.max_sector_weight + 1e-6:
                worst_sector = sector_weights.idxmax()
                violations.append(
                    f"Sector {worst_sector} weight {max_sector_w:.4f} "
                    f"exceeds limit {self.max_sector_weight}"
                )

        # Turnover constraint (if provided)
        if previous_weights is not None and self.max_turnover is not None:
            # Align indices
            aligned_prev = previous_weights.reindex(weights.index, fill_value=0)
            turnover = (weights - aligned_prev).abs().sum()
            if turnover > self.max_turnover + 1e-6:
                violations.append(
                    f"Turnover {turnover:.4f} exceeds limit {self.max_turnover}"
                )

        is_valid = len(violations) == 0

        if not is_valid:
            logger.warning(f"Constraint violations: {violations}")

        return is_valid, violations

    def enforce_cardinality(
        self,
        weights: np.ndarray,
        scores: np.ndarray
    ) -> np.ndarray:
        """
        Enforce maximum number of positions by keeping top K by score.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        scores : np.ndarray
            Score for each position (higher = better)

        Returns
        -------
        np.ndarray
            Weights with cardinality constraint enforced
        """
        n_positions = np.sum(weights > 1e-6)

        if n_positions <= self.max_positions:
            return weights

        # Keep top max_positions by score
        threshold_idx = np.argsort(scores)[-self.max_positions]
        threshold = scores[threshold_idx]

        new_weights = weights.copy()
        new_weights[scores < threshold] = 0

        logger.info(
            f"Cardinality constraint: reduced from {n_positions} "
            f"to {self.max_positions} positions"
        )

        return new_weights

    def enforce_min_weight(self, weights: np.ndarray) -> np.ndarray:
        """
        Enforce minimum weight by removing small positions.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights

        Returns
        -------
        np.ndarray
            Weights with min weight constraint enforced
        """
        new_weights = weights.copy()
        small_positions = (weights > 1e-6) & (weights < self.min_weight)

        if np.any(small_positions):
            new_weights[small_positions] = 0
            n_removed = np.sum(small_positions)
            logger.info(
                f"Min weight constraint: removed {n_removed} positions "
                f"below {self.min_weight}"
            )

        return new_weights

    def enforce_turnover(
        self,
        new_weights: np.ndarray,
        prev_weights: np.ndarray
    ) -> np.ndarray:
        """
        Enforce turnover limit by adjusting towards previous weights.

        Parameters
        ----------
        new_weights : np.ndarray
            Proposed new weights
        prev_weights : np.ndarray
            Previous portfolio weights

        Returns
        -------
        np.ndarray
            Adjusted weights respecting turnover limit
        """
        if self.max_turnover is None:
            return new_weights

        turnover = np.sum(np.abs(new_weights - prev_weights))

        if turnover <= self.max_turnover:
            return new_weights

        # Scale move towards new weights to respect turnover limit
        # w_adjusted = w_prev + α * (w_new - w_prev)
        # where α is chosen such that turnover = max_turnover
        alpha = self.max_turnover / turnover
        adjusted_weights = prev_weights + alpha * (new_weights - prev_weights)

        logger.info(
            f"Turnover constraint: scaled move by {alpha:.2f} "
            f"(turnover {turnover:.4f} -> {self.max_turnover})"
        )

        return adjusted_weights


class RiskConstraints:
    """Risk-based constraints (CVaR, drawdown, etc.)."""

    def __init__(
        self,
        max_cvar: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize risk constraints.

        Parameters
        ----------
        max_cvar : float, optional
            Maximum CVaR (Conditional Value at Risk) at confidence level
            E.g., 0.15 means max 15% expected loss in worst 5% scenarios
        max_drawdown : float, optional
            Maximum historical drawdown allowed (e.g., 0.15 = 15%)
        confidence_level : float
            Confidence level for CVaR (default 0.95 = 95%)
        """
        self.max_cvar = max_cvar
        self.max_drawdown = max_drawdown
        self.confidence_level = confidence_level

    def calculate_cvar(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio CVaR (Conditional Value at Risk).

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame
            Historical returns

        Returns
        -------
        float
            CVaR (expected loss in worst (1-confidence_level)% scenarios)
        """
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # VaR threshold
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)

        # CVaR = mean of returns below VaR
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return abs(cvar)

    def calculate_drawdown(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio maximum drawdown.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame
            Historical returns

        Returns
        -------
        float
            Maximum drawdown
        """
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Cumulative returns
        cumulative = (1 + portfolio_returns).cumprod()

        # Running maximum
        running_max = cumulative.expanding().max()

        # Drawdown
        drawdown = (cumulative - running_max) / running_max

        return abs(drawdown.min())

    def check_risk_constraints(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio satisfies risk constraints.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame
            Historical returns

        Returns
        -------
        tuple
            (is_valid, violation_messages)
        """
        violations = []

        # CVaR constraint
        if self.max_cvar is not None:
            cvar = self.calculate_cvar(weights, returns)
            if cvar > self.max_cvar + 1e-6:
                violations.append(
                    f"CVaR {cvar:.4f} exceeds limit {self.max_cvar} "
                    f"at {self.confidence_level:.0%} confidence"
                )

        # Drawdown constraint
        if self.max_drawdown is not None:
            drawdown = self.calculate_drawdown(weights, returns)
            if drawdown > self.max_drawdown + 1e-6:
                violations.append(
                    f"Max drawdown {drawdown:.4f} exceeds limit {self.max_drawdown}"
                )

        is_valid = len(violations) == 0

        if not is_valid:
            logger.warning(f"Risk constraint violations: {violations}")

        return is_valid, violations
