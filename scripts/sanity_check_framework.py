"""
Sanity Check Framework for Portfolio Metrics

Validates that performance metrics are within realistic bounds based on
validated backtests from TECHNICAL_INVESTMENT_DOCUMENT.tex.

This prevents the reporting of impossible or unrealistic metrics that
would undermine confidence in the system.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceSanityCheck:
    """
    Validates that metrics are within realistic bounds.

    Based on validated performance from TECHNICAL_INVESTMENT_DOCUMENT.tex:
    - CAGR: 17.0% (5-year validated)
    - Sharpe: 1.07 (validated)
    - Max DD: -15.7% (validated)
    - Volatility: ~12-15% (typical for ETF portfolio)
    """

    # Realistic bounds for each metric
    BOUNDS = {
        'sharpe_ratio': (0.0, 2.5),           # Sharpe > 2.5 is extremely rare
        'volatility': (0.05, 0.40),           # 5-40% reasonable range for ETFs
        'max_drawdown': (-0.50, 0.0),         # -50% to 0%
        'cagr': (-0.30, 0.50),                # -30% to +50% reasonable
        'drift': (0.0, 1.00),                 # 0-100% drift (200% is theoretical max)
        'turnover_per_rebalance': (0.0, 0.80), # 0-80% turnover per rebalance
        'total_return': (-0.50, 5.0),         # -50% to +500% over full period
        'win_rate': (0.0, 1.0),               # 0-100% probability
    }

    # Critical thresholds that indicate calculation errors
    CRITICAL_THRESHOLDS = {
        'sharpe_ratio': 2.0,      # Sharpe > 2.0 is suspicious
        'volatility_min': 0.08,   # Vol < 8% is unrealistic for equities
        'volatility_max': 0.30,   # Vol > 30% suggests extreme strategy
        'drift_max': 1.00,        # Drift > 100% indicates bug
    }

    def __init__(self, strict: bool = True):
        """
        Parameters
        ----------
        strict : bool
            If True, raise exceptions on violations
            If False, log warnings only
        """
        self.strict = strict

    def check_sharpe(self, sharpe: float, context: str = "") -> bool:
        """
        Check if Sharpe ratio is realistic.

        Sharpe > 2.0 is extremely rare and suggests calculation error.
        Historical data shows validated Sharpe of 1.07.
        """
        min_val, max_val = self.BOUNDS['sharpe_ratio']

        if not (min_val <= sharpe <= max_val):
            msg = f"Sharpe ratio {sharpe:.2f} outside bounds [{min_val}, {max_val}]"
            if context:
                msg += f" ({context})"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        # Additional check for suspicious values
        if sharpe > self.CRITICAL_THRESHOLDS['sharpe_ratio']:
            msg = f"⚠️  Sharpe ratio {sharpe:.2f} > {self.CRITICAL_THRESHOLDS['sharpe_ratio']:.1f} is suspicious"
            if context:
                msg += f" ({context})"
            msg += "\n   Validated backtest showed Sharpe ~1.07"
            msg += "\n   This may indicate a calculation error (e.g., using z-scores as returns)"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        return True

    def check_volatility(self, vol: float, context: str = "") -> bool:
        """
        Check if volatility is realistic.

        Volatility < 8% is unrealistic for equity portfolios.
        Typical range is 12-15% for diversified ETF portfolios.
        """
        min_val, max_val = self.BOUNDS['volatility']

        if not (min_val <= vol <= max_val):
            msg = f"Volatility {vol:.2%} outside bounds [{min_val:.0%}, {max_val:.0%}]"
            if context:
                msg += f" ({context})"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        # Check for suspiciously low volatility
        if vol < self.CRITICAL_THRESHOLDS['volatility_min']:
            msg = f"⚠️  Volatility {vol:.2%} < {self.CRITICAL_THRESHOLDS['volatility_min']:.0%} is unrealistic"
            if context:
                msg += f" ({context})"
            msg += "\n   Expected: ~12-15% for ETF portfolio"
            msg += "\n   This may indicate daily vs annual unit confusion"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        # Check for very high volatility
        if vol > self.CRITICAL_THRESHOLDS['volatility_max']:
            msg = f"⚠️  Volatility {vol:.2%} > {self.CRITICAL_THRESHOLDS['volatility_max']:.0%} is very high"
            if context:
                msg += f" ({context})"
            logger.warning(msg)
            # Don't fail on high vol - just warn

        return True

    def check_drift(self, drift: float, context: str = "") -> bool:
        """
        Check if portfolio drift is realistic.

        Drift > 100% indicates a calculation bug.
        Typical drift: 5-20% between rebalances.
        """
        min_val, max_val = self.BOUNDS['drift']

        if not (min_val <= drift <= max_val):
            msg = f"Drift {drift:.0%} outside bounds [{min_val:.0%}, {max_val:.0%}]"
            if context:
                msg += f" ({context})"
            msg += "\n   Drift > 100% indicates calculation error"
            msg += "\n   Check that weights are normalized and aligned correctly"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        return True

    def check_turnover(self, turnover: float, context: str = "") -> bool:
        """
        Check if turnover per rebalance is realistic.

        Turnover > 80% per rebalance is very high and may indicate
        optimizer instability or incorrect rebalancing logic.
        """
        min_val, max_val = self.BOUNDS['turnover_per_rebalance']

        if not (min_val <= turnover <= max_val):
            msg = f"Turnover {turnover:.0%} outside bounds [{min_val:.0%}, {max_val:.0%}]"
            if context:
                msg += f" ({context})"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        return True

    def check_cagr(self, cagr: float, context: str = "") -> bool:
        """Check if CAGR is realistic."""
        min_val, max_val = self.BOUNDS['cagr']

        if not (min_val <= cagr <= max_val):
            msg = f"CAGR {cagr:.2%} outside bounds [{min_val:.0%}, {max_val:.0%}]"
            if context:
                msg += f" ({context})"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        return True

    def check_max_drawdown(self, dd: float, context: str = "") -> bool:
        """Check if max drawdown is realistic."""
        min_val, max_val = self.BOUNDS['max_drawdown']

        if not (min_val <= dd <= max_val):
            msg = f"Max drawdown {dd:.2%} outside bounds [{min_val:.0%}, {max_val:.0%}]"
            if context:
                msg += f" ({context})"

            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        return True

    def check_all(self, metrics: Dict[str, float], context: str = "") -> bool:
        """
        Check all provided metrics at once.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric_name -> value
        context : str
            Context for error messages

        Returns
        -------
        bool
            True if all checks pass, False otherwise
        """
        all_passed = True

        for metric_name, value in metrics.items():
            if value is None:
                continue

            if metric_name == 'sharpe_ratio':
                all_passed &= self.check_sharpe(value, context)
            elif metric_name == 'volatility':
                all_passed &= self.check_volatility(value, context)
            elif metric_name == 'drift':
                all_passed &= self.check_drift(value, context)
            elif metric_name == 'turnover':
                all_passed &= self.check_turnover(value, context)
            elif metric_name == 'cagr':
                all_passed &= self.check_cagr(value, context)
            elif metric_name == 'max_drawdown':
                all_passed &= self.check_max_drawdown(value, context)
            else:
                # Unknown metric - skip
                pass

        return all_passed


def validate_backtest_results(results: Dict[str, float],
                              optimizer_name: str = "",
                              strict: bool = False) -> bool:
    """
    Convenience function to validate backtest results.

    Parameters
    ----------
    results : dict
        Dictionary of metric names -> values
    optimizer_name : str
        Name of optimizer for context
    strict : bool
        If True, raise exceptions on violations

    Returns
    -------
    bool
        True if all checks pass

    Example
    -------
    >>> results = {
    ...     'cagr': 0.17,
    ...     'sharpe_ratio': 1.07,
    ...     'volatility': 0.14,
    ...     'max_drawdown': -0.157
    ... }
    >>> validate_backtest_results(results, optimizer_name="MVO")
    True
    """
    checker = PerformanceSanityCheck(strict=strict)
    context = f"{optimizer_name}" if optimizer_name else ""

    return checker.check_all(results, context=context)


if __name__ == "__main__":
    # Example usage and tests
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("SANITY CHECK FRAMEWORK TESTS")
    print("=" * 80)

    checker = PerformanceSanityCheck(strict=False)

    # Test 1: Validated performance (should pass)
    print("\n1. Testing validated performance (should PASS):")
    validated = {
        'cagr': 0.17,
        'sharpe_ratio': 1.07,
        'volatility': 0.14,
        'max_drawdown': -0.157
    }
    result = checker.check_all(validated, context="Validated MVO")
    print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")

    # Test 2: Impossible Sharpe (should fail)
    print("\n2. Testing impossible Sharpe ratio (should FAIL):")
    impossible = {
        'sharpe_ratio': 21.46,
        'volatility': 0.005,  # 0.5%
    }
    result = checker.check_all(impossible, context="Buggy optimizer")
    print(f"   Result: {'✓ PASS' if result else '✗ FAIL (expected)'}")

    # Test 3: Drift > 100% (should fail)
    print("\n3. Testing drift > 100% (should FAIL):")
    bad_drift = {'drift': 1.80}
    result = checker.check_drift(bad_drift['drift'], context="Portfolio rebalance")
    print(f"   Result: {'✓ PASS' if result else '✗ FAIL (expected)'}")

    # Test 4: High turnover (should fail)
    print("\n4. Testing excessive turnover (should FAIL):")
    high_turnover = {'turnover': 0.85}
    result = checker.check_turnover(high_turnover['turnover'], context="Weekly rebalance")
    print(f"   Result: {'✓ PASS' if result else '✗ FAIL (expected)'}")

    # Test 5: Realistic range (should pass)
    print("\n5. Testing realistic performance range (should PASS):")
    realistic = {
        'cagr': 0.22,
        'sharpe_ratio': 1.4,
        'volatility': 0.13,
        'max_drawdown': -0.11
    }
    result = checker.check_all(realistic, context="Period 3 MVO")
    print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")

    print("\n" + "=" * 80)
    print("✓ Sanity check framework tests complete")
    print("=" * 80)
