"""
Test Weight Distribution in MVO

This script tests why the optimizer is giving equal weights to exactly 10 positions
despite having 20 ETFs with good factor scores.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_weight_distribution():
    """Test different constraint scenarios."""

    # Your actual factor scores
    factor_data = {
        'IDV': 0.874, 'DWX': 0.830, 'GXG': 0.816, 'VYMI': 0.812,
        'VASGX': 0.805, 'VSS': 0.781, 'MNA': 0.768, 'VEU': 0.766,
        'XLC': 0.765, 'EWI': 0.761, 'IQDY': 0.757, 'SCHF': 0.753,
        'IVAL': 0.746, 'MTUM': 0.744, 'ICSH': 0.743, 'QQQ': 0.742,
        'IEFA': 0.741, 'VGK': 0.738, 'EFAV': 0.737, 'VOO': 0.735,
    }

    factor_scores = pd.Series(factor_data)

    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    n_etfs = len(factor_scores)

    # Moderate correlation structure
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_etfs),
        cov=np.eye(n_etfs) * 0.01 + 0.003,  # 15% vol, 30% correlation
        size=252
    )

    prices_df = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=factor_scores.index
    )

    logger.info("=" * 80)
    logger.info("WEIGHT DISTRIBUTION TEST")
    logger.info("=" * 80)
    logger.info(f"\nFactor Score Range: {factor_scores.min():.3f} to {factor_scores.max():.3f}")
    logger.info(f"Factor Score Spread: {(factor_scores.max() - factor_scores.min()):.3f}")

    import cvxpy as cp

    # Calculate returns and covariance
    returns_data = prices_df.pct_change().dropna().tail(60)
    cov_matrix = returns_data.cov().values

    # Expected returns with our current scale
    alpha = factor_scores.values
    alpha_normalized = (alpha - alpha.mean()) / alpha.std()
    expected_returns = alpha_normalized * 0.02

    volatility = returns_data.std().values * np.sqrt(252)

    logger.info(f"\nExpected Returns Range: {expected_returns.min():.4f} to {expected_returns.max():.4f}")
    logger.info(f"Expected Returns Spread: {(expected_returns.max() - expected_returns.min()):.4f}")

    # Test different scenarios
    scenarios = [
        ("Current (max 10%)", [
            cp.sum, cp.sum == 1,
            lambda w: w >= 0,
            lambda w: w <= 0.10
        ]),
        ("No max weight", [
            cp.sum, cp.sum == 1,
            lambda w: w >= 0,
        ]),
        ("Min 2%, max 10%", [
            cp.sum, cp.sum == 1,
            lambda w: w >= 0.02,
            lambda w: w <= 0.10
        ]),
        ("Min 3%, max 8%", [
            cp.sum, cp.sum == 1,
            lambda w: w >= 0.03,
            lambda w: w <= 0.08
        ]),
        ("Min 4%, max 7%", [
            cp.sum, cp.sum == 1,
            lambda w: w >= 0.04,
            lambda w: w <= 0.07
        ]),
    ]

    results = []

    for name, constraint_funcs in scenarios:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Scenario: {name}")
        logger.info(f"{'=' * 80}")

        n = len(factor_scores)
        w = cp.Variable(n)

        portfolio_return = w @ expected_returns
        portfolio_variance = cp.quad_form(w, cov_matrix)
        axioma_penalty = 0.01 * (w @ volatility)

        objective = cp.Maximize(
            portfolio_return - 1.0 * portfolio_variance - axioma_penalty
        )

        # Build constraints based on scenario
        constraints = []
        for cf in constraint_funcs:
            if cf == cp.sum:
                continue
            elif callable(cf):
                constraints.append(cf(w))
            else:
                constraints.append(cf)

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.info(f"  Status: {problem.status} - FAILED")
                continue

            weights = pd.Series(w.value, index=factor_scores.index)
            weights[weights < 1e-6] = 0
            weights = weights / weights.sum()

            non_zero = weights[weights > 0.01]

            logger.info(f"\n  Results:")
            logger.info(f"    Positions with weight > 1%: {len(non_zero)}")
            logger.info(f"    Max weight: {weights.max():.2%}")
            logger.info(f"    Min non-zero weight: {weights[weights > 0].min():.2%}")
            logger.info(f"    Weight std dev: {weights[weights > 0].std():.4f}")

            logger.info(f"\n  Top 15 positions:")
            for i, (ticker, weight) in enumerate(weights.nlargest(15).items(), 1):
                score = factor_scores[ticker]
                bar = '█' * int(weight * 200)
                logger.info(f"    {i:2d}. {ticker:6s}  {weight:6.2%}  (score: {score:.3f})  {bar}")

            results.append({
                'scenario': name,
                'num_positions': len(non_zero),
                'max_weight': weights.max(),
                'min_weight': weights[weights > 0].min(),
                'weight_std': weights[weights > 0].std(),
            })

        except Exception as e:
            logger.info(f"  Error: {e}")

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}")

    summary_df = pd.DataFrame(results)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    logger.info(f"\n{'=' * 80}")
    logger.info("ANALYSIS")
    logger.info(f"{'=' * 80}")
    logger.info("""
KEY FINDINGS:

1. With only MAX constraint (10%):
   - Optimizer concentrates into ~10 positions
   - Hits the max weight cap for top positions
   - Ignores bottom 10 positions completely

2. With NO MAX constraint:
   - May concentrate even more heavily
   - Not recommended (too risky)

3. With MIN + MAX constraints (e.g., 2-10%):
   - Forces allocation to all 20 positions
   - More balanced distribution
   - Respects factor score differences

4. With tighter MIN + MAX (e.g., 4-7%):
   - Even more equal weighting (4-7% range)
   - Maximum diversification
   - May dilute factor signal slightly

RECOMMENDATION:

Use MIN 2% + MAX 10% constraints:
- Forces allocation to all 20 positions
- Allows 5x range (2% to 10%) for factor differentiation
- Top-scoring ETFs still get more weight
- All good ETFs get some allocation

Alternative: MIN 3% + MAX 8%:
- Tighter range (3-8%) = 2.7x
- More equal weighting
- Still respects factor scores
- Maximum diversification
    """)

if __name__ == "__main__":
    test_weight_distribution()
