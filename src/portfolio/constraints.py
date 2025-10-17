"""
Portfolio Constraints

Defines constraints for portfolio construction including:
- Asset class limits
- Sector diversification
- Geographic exposure
- Position sizing limits
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """
    Portfolio-level constraints.

    Attributes:
        max_position_weight: Maximum weight for any single position
        min_position_weight: Minimum weight for any position
        max_sector_weight: Maximum weight for any sector
        max_asset_class_weight: Maximum weight for any asset class
        max_country_weight: Maximum weight for any country
        min_num_positions: Minimum number of positions
        max_num_positions: Maximum number of positions
    """
    max_position_weight: float = 0.15  # 15% per position max
    min_position_weight: float = 0.01  # 1% per position min
    max_sector_weight: float = 0.40    # 40% per sector max
    max_asset_class_weight: float = 1.0  # No limit by default
    max_country_weight: float = 1.0    # No limit by default
    min_num_positions: int = 10
    max_num_positions: int = 30

    def __post_init__(self):
        """Validate constraints."""
        if self.max_position_weight < self.min_position_weight:
            raise ValueError("max_position_weight must be >= min_position_weight")

        if self.max_num_positions < self.min_num_positions:
            raise ValueError("max_num_positions must be >= min_num_positions")

        if self.min_position_weight * self.max_num_positions > 1.0:
            logger.warning(
                f"min_position_weight ({self.min_position_weight}) * "
                f"max_num_positions ({self.max_num_positions}) > 1.0. "
                f"This may make the problem infeasible."
            )


class ConstraintChecker:
    """
    Checks if portfolio weights satisfy constraints.

    Parameters:
        constraints: PortfolioConstraints object
        etf_metadata: DataFrame with ETF metadata (sector, asset_class, country, etc.)
    """

    def __init__(self,
                 constraints: PortfolioConstraints,
                 etf_metadata: Optional[pd.DataFrame] = None):
        self.constraints = constraints
        self.etf_metadata = etf_metadata

        logger.info("ConstraintChecker initialized")
        logger.info(f"  Max position weight: {constraints.max_position_weight:.1%}")
        logger.info(f"  Max sector weight: {constraints.max_sector_weight:.1%}")
        logger.info(f"  Position range: {constraints.min_num_positions}-{constraints.max_num_positions}")

    def check_position_weights(self, weights: pd.Series) -> Dict[str, bool]:
        """Check if individual position weights satisfy constraints."""
        results = {}

        # Check max weight
        max_weight = weights.max()
        results['max_weight_ok'] = max_weight <= self.constraints.max_position_weight
        if not results['max_weight_ok']:
            logger.warning(
                f"Max position weight violated: {max_weight:.2%} > "
                f"{self.constraints.max_position_weight:.2%}"
            )

        # Check min weight (for non-zero positions)
        non_zero_weights = weights[weights > 1e-6]
        if len(non_zero_weights) > 0:
            min_weight = non_zero_weights.min()
            results['min_weight_ok'] = min_weight >= self.constraints.min_position_weight
            if not results['min_weight_ok']:
                logger.warning(
                    f"Min position weight violated: {min_weight:.2%} < "
                    f"{self.constraints.min_position_weight:.2%}"
                )
        else:
            results['min_weight_ok'] = True

        # Check number of positions
        num_positions = len(non_zero_weights)
        results['num_positions_ok'] = (
            self.constraints.min_num_positions <= num_positions <=
            self.constraints.max_num_positions
        )
        if not results['num_positions_ok']:
            logger.warning(
                f"Number of positions violated: {num_positions} not in "
                f"[{self.constraints.min_num_positions}, {self.constraints.max_num_positions}]"
            )

        return results

    def check_sector_weights(self, weights: pd.Series) -> Dict[str, bool]:
        """Check if sector weights satisfy constraints."""
        if self.etf_metadata is None or 'sector' not in self.etf_metadata.columns:
            logger.debug("No sector metadata available, skipping sector check")
            return {'sector_weights_ok': True}

        results = {}

        # Calculate sector weights
        sector_weights = {}
        for ticker, weight in weights.items():
            if ticker in self.etf_metadata.index:
                sector = self.etf_metadata.loc[ticker, 'sector']
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Check each sector
        all_ok = True
        for sector, weight in sector_weights.items():
            if weight > self.constraints.max_sector_weight:
                logger.warning(
                    f"Sector weight violated: {sector} = {weight:.2%} > "
                    f"{self.constraints.max_sector_weight:.2%}"
                )
                all_ok = False

        results['sector_weights_ok'] = all_ok
        return results

    def check_asset_class_weights(self, weights: pd.Series) -> Dict[str, bool]:
        """Check if asset class weights satisfy constraints."""
        if self.etf_metadata is None or 'asset_class' not in self.etf_metadata.columns:
            logger.debug("No asset class metadata available, skipping check")
            return {'asset_class_weights_ok': True}

        results = {}

        # Calculate asset class weights
        asset_class_weights = {}
        for ticker, weight in weights.items():
            if ticker in self.etf_metadata.index:
                asset_class = self.etf_metadata.loc[ticker, 'asset_class']
                asset_class_weights[asset_class] = asset_class_weights.get(asset_class, 0) + weight

        # Check each asset class
        all_ok = True
        for asset_class, weight in asset_class_weights.items():
            if weight > self.constraints.max_asset_class_weight:
                logger.warning(
                    f"Asset class weight violated: {asset_class} = {weight:.2%} > "
                    f"{self.constraints.max_asset_class_weight:.2%}"
                )
                all_ok = False

        results['asset_class_weights_ok'] = all_ok
        return results

    def check_country_weights(self, weights: pd.Series) -> Dict[str, bool]:
        """Check if country weights satisfy constraints."""
        if self.etf_metadata is None or 'country' not in self.etf_metadata.columns:
            logger.debug("No country metadata available, skipping check")
            return {'country_weights_ok': True}

        results = {}

        # Calculate country weights
        country_weights = {}
        for ticker, weight in weights.items():
            if ticker in self.etf_metadata.index:
                country = self.etf_metadata.loc[ticker, 'country']
                country_weights[country] = country_weights.get(country, 0) + weight

        # Check each country
        all_ok = True
        for country, weight in country_weights.items():
            if weight > self.constraints.max_country_weight:
                logger.warning(
                    f"Country weight violated: {country} = {weight:.2%} > "
                    f"{self.constraints.max_country_weight:.2%}"
                )
                all_ok = False

        results['country_weights_ok'] = all_ok
        return results

    def check_all(self, weights: pd.Series) -> Dict[str, bool]:
        """
        Check all constraints.

        Returns:
            Dictionary with results for each constraint check
        """
        results = {}

        # Position weights
        results.update(self.check_position_weights(weights))

        # Sector weights
        results.update(self.check_sector_weights(weights))

        # Asset class weights
        results.update(self.check_asset_class_weights(weights))

        # Country weights
        results.update(self.check_country_weights(weights))

        # Overall result
        results['all_constraints_ok'] = all(results.values())

        if results['all_constraints_ok']:
            logger.info("✓ All constraints satisfied")
        else:
            failed = [k for k, v in results.items() if not v and k != 'all_constraints_ok']
            logger.warning(f"✗ Constraints violated: {', '.join(failed)}")

        return results

    def get_violation_summary(self, weights: pd.Series) -> str:
        """Get a human-readable summary of constraint violations."""
        results = self.check_all(weights)

        if results['all_constraints_ok']:
            return "All constraints satisfied ✓"

        violations = []

        if not results['max_weight_ok']:
            max_weight = weights.max()
            max_ticker = weights.idxmax()
            violations.append(
                f"  - Max position: {max_ticker} = {max_weight:.2%} "
                f"(limit: {self.constraints.max_position_weight:.2%})"
            )

        if not results['min_weight_ok']:
            non_zero = weights[weights > 1e-6]
            min_weight = non_zero.min()
            min_ticker = non_zero.idxmin()
            violations.append(
                f"  - Min position: {min_ticker} = {min_weight:.2%} "
                f"(limit: {self.constraints.min_position_weight:.2%})"
            )

        if not results['num_positions_ok']:
            num_pos = len(weights[weights > 1e-6])
            violations.append(
                f"  - Number of positions: {num_pos} "
                f"(range: {self.constraints.min_num_positions}-{self.constraints.max_num_positions})"
            )

        if not results.get('sector_weights_ok', True) and self.etf_metadata is not None:
            violations.append("  - Sector concentration limits")

        if not results.get('asset_class_weights_ok', True) and self.etf_metadata is not None:
            violations.append("  - Asset class concentration limits")

        if not results.get('country_weights_ok', True) and self.etf_metadata is not None:
            violations.append("  - Country concentration limits")

        return "Constraint violations:\n" + "\n".join(violations)


def apply_position_limits(weights: pd.Series,
                         max_weight: float = 0.15,
                         min_weight: float = 0.01) -> pd.Series:
    """
    Apply position size limits to weights, renormalizing to sum to 1.

    Args:
        weights: Portfolio weights (should sum to 1)
        max_weight: Maximum weight per position
        min_weight: Minimum weight per position (or zero)

    Returns:
        Adjusted weights that sum to 1 and respect limits
    """
    adjusted = weights.copy()

    # Clip to max weight
    adjusted = adjusted.clip(upper=max_weight)

    # Remove positions below min weight
    adjusted[adjusted < min_weight] = 0.0

    # Renormalize
    if adjusted.sum() > 0:
        adjusted = adjusted / adjusted.sum()

    logger.debug(f"Applied position limits: {max_weight:.1%} max, {min_weight:.1%} min")

    return adjusted


def diversification_score(weights: pd.Series) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI) for concentration.

    Lower is more diversified:
    - HHI = 1.0: Fully concentrated (1 position)
    - HHI = 0.05: Well diversified (20 equal positions)
    - HHI = 0.01: Very diversified (100 equal positions)

    Args:
        weights: Portfolio weights

    Returns:
        HHI score (0-1)
    """
    return (weights ** 2).sum()


def effective_num_positions(weights: pd.Series) -> float:
    """
    Calculate effective number of positions (1 / HHI).

    This represents the "equivalent" number of equal-weighted positions.

    Args:
        weights: Portfolio weights

    Returns:
        Effective number of positions
    """
    hhi = diversification_score(weights)
    return 1.0 / hhi if hhi > 0 else 0.0
