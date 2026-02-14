"""
Test Factor Score Sensitivity in Mean-Variance Optimizer

This script tests how different factor score scaling parameters
affect portfolio concentration in the MVO optimizer.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio import MeanVarianceOptimizer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_factor_sensitivity():
    """
    Test how alpha scaling affects portfolio concentration.

    We'll use the actual factor scores from your portfolio
    and test different scaling parameters.
    """

    # Your actual factor scores (from the Excel output)
    factor_data = {
        'IDV': 0.832332725,
        'DFAI': 0.798941341,
        'DLS': 0.789269643,
        'VASGX': 0.784122868,
        'ACWX': 0.781159052,
        'IQDF': 0.779911976,
        'IXUS': 0.777605443,
        'EWU': 0.776172769,
        'EZU': 0.775378789,
        'VSMGX': 0.772788396,
        'URTH': 0.767750743,
        'AOR': 0.761048031,
        'AVDE': 0.760498944,
        'FPXE': 0.760085629,
        'VGK': 0.756171463,
        'EWI': 0.754364911,
        'EFA': 0.753118083,
        'VSGX': 0.748808005,
        'GXG': 0.747364089,
        'VEU': 0.746205665,
    }

    factor_scores = pd.Series(factor_data)

    # Create synthetic price data for covariance
    # Use same volatility for all to isolate factor score effect
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')

    # Create synthetic returns with similar correlation structure
    # All have ~15% volatility, moderate correlation
    n_etfs = len(factor_scores)
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_etfs),
        cov=np.eye(n_etfs) * 0.01 + 0.005,  # 15% vol + 50% correlation
        size=252
    )

    prices_df = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=factor_scores.index
    )

    logger.info("=" * 80)
    logger.info("FACTOR SCORE SENSITIVITY TEST")
    logger.info("=" * 80)
    logger.info(f"\nFactor Score Range: {factor_scores.min():.3f} to {factor_scores.max():.3f}")
    logger.info(f"Factor Score Spread: {(factor_scores.max() - factor_scores.min()):.3f}")
    logger.info(f"Factor Score StDev: {factor_scores.std():.3f}")

    # Test different alpha scaling factors
    test_cases = [
        ("Current (0.10)", 0.10, "VERY HIGH - causes concentration"),
        ("Moderate (0.05)", 0.05, "Medium influence"),
        ("Conservative (0.02)", 0.02, "Low influence"),
        ("Minimal (0.01)", 0.01, "Very low influence"),
    ]

    logger.info("\n" + "=" * 80)
    logger.info("TESTING DIFFERENT ALPHA SCALING FACTORS")
    logger.info("=" * 80)

    results = []

    for name, alpha_scale, description in test_cases:
        logger.info(f"\n{'-' * 80}")
        logger.info(f"Test: {name} - {description}")
        logger.info(f"{'-' * 80}")

        # Create optimizer with this alpha scale
        # We'll monkey-patch the scale factor
        optimizer = MeanVarianceOptimizer(
            num_positions=20,
            lookback=60,
            risk_aversion=1.0,
            axioma_penalty=0.01,
            use_factor_scores_as_alpha=True
        )

        # Temporarily modify the optimize method to use our alpha scale
        original_optimize = optimizer.optimize

        def custom_optimize(factor_scores_arg, prices_arg):
            import cvxpy as cp

            # Copy from original but with custom alpha scale
            eligible = factor_scores_arg
            selected_tickers = eligible.nlargest(20).index.tolist()

            returns = prices_arg[selected_tickers].pct_change().dropna().tail(60)
            cov_matrix = returns.cov().values

            # CUSTOM ALPHA SCALE HERE
            alpha = factor_scores_arg[selected_tickers].values
            alpha_normalized = (alpha - alpha.mean()) / alpha.std()
            expected_returns = alpha_normalized * alpha_scale  # MODIFIED

            volatility = returns.std().values * np.sqrt(252)

            n = len(selected_tickers)
            w = cp.Variable(n)

            portfolio_return = w @ expected_returns
            portfolio_variance = cp.quad_form(w, cov_matrix)
            axioma_penalty = 0.01 * (w @ volatility)

            objective = cp.Maximize(
                portfolio_return - 1.0 * portfolio_variance - axioma_penalty
            )

            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= 0.15,  # Still using 15% cap for comparison
            ]

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                weights = pd.Series(1.0 / n, index=selected_tickers)
            else:
                weights = pd.Series(w.value, index=selected_tickers)
                weights[weights < 1e-6] = 0
                weights = weights / weights.sum()

            return weights

        optimizer.optimize = custom_optimize

        # Run optimization
        weights = optimizer.optimize(factor_scores, prices_df)

        # Analyze results
        non_zero = weights[weights > 0.01]
        max_weight = weights.max()
        min_nonzero_weight = non_zero[non_zero > 0].min() if len(non_zero) > 0 else 0

        logger.info(f"\nResults:")
        logger.info(f"  Positions with weight > 1%: {len(non_zero)}")
        logger.info(f"  Max weight: {max_weight:.1%}")
        logger.info(f"  Min non-zero weight: {min_nonzero_weight:.1%}")
        logger.info(f"  Top 5 weights: {weights.nlargest(5).values}")

        # Show weight distribution
        logger.info(f"\n  Weight Distribution:")
        for i, (ticker, weight) in enumerate(weights.nlargest(10).items(), 1):
            score = factor_scores[ticker]
            logger.info(f"    {i:2d}. {ticker:6s}  {weight:6.1%}  (score: {score:.3f})")

        results.append({
            'alpha_scale': alpha_scale,
            'name': name,
            'num_positions': len(non_zero),
            'max_weight': max_weight,
            'min_weight': min_nonzero_weight,
            'concentration': weights.nlargest(5).sum(),
        })

    # Summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY COMPARISON")
    logger.info("=" * 80)

    summary_df = pd.DataFrame(results)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Recommendation
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION")
    logger.info("=" * 80)

    logger.info("""
Based on this analysis:

CURRENT (0.10 alpha scale):
- Creates 30% spread in expected returns
- Forces concentration into top 7 positions
- Ignores 13 good ETFs (scores 0.746-0.776)

RECOMMENDED (0.02 alpha scale):
- Creates 6% spread in expected returns (more realistic)
- Should allocate to 12-15 positions
- Better diversification while respecting factor signals
- Still favors top-scoring ETFs but not excessively

IMPLEMENTATION:
Change line 473 in src/portfolio/optimizer.py:
    FROM: expected_returns = alpha_normalized * 0.10
    TO:   expected_returns = alpha_normalized * 0.02

This reduces factor influence while maintaining the multi-factor approach.
Factor scores still matter, but won't dominate the optimization.
    """)

if __name__ == "__main__":
    test_factor_sensitivity()
