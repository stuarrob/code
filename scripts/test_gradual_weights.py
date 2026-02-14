"""
Test different alpha scaling values to find configuration that produces gradual weights.

This script tests alpha scaling from 0.02 to 0.15 to see if we can get more gradual
weight distribution while maintaining full 20-position allocation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cvxpy as cp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.factors import (
    FactorIntegrator,
    MomentumFactor,
    QualityFactor,
    SimplifiedValueFactor,
    VolatilityFactor,
)

DATA_DIR = PROJECT_ROOT / "data"

print("=" * 80)
print("TESTING ALPHA SCALING FOR GRADUAL WEIGHTS")
print("=" * 80)
print()

# Load prices
print("Loading data...")
prices = pd.read_parquet(DATA_DIR / "processed" / "etf_prices_filtered.parquet")

# Calculate factors
momentum = MomentumFactor(lookback=252, skip_recent=21)
quality = QualityFactor(lookback=252)
value = SimplifiedValueFactor()
volatility = VolatilityFactor(lookback=60)

integrator = FactorIntegrator(
    factor_weights={
        "momentum": 0.25,
        "quality": 0.25,
        "value": 0.25,
        "volatility": 0.25,
    }
)

# Calculate scores
momentum_scores = momentum.calculate(prices)
quality_scores = quality.calculate(prices)
expense_ratios = pd.Series(
    np.random.uniform(0.0005, 0.01, len(prices.columns)),
    index=prices.columns,
)
value_scores = value.calculate(prices, expense_ratios)
volatility_scores = volatility.calculate(prices)

factor_scores_df = pd.DataFrame({
    'momentum': momentum_scores,
    'quality': quality_scores,
    'value': value_scores,
    'volatility': volatility_scores
})

factor_scores = integrator.integrate(factor_scores_df)

# Select top 20
num_positions = 20
lookback = 60
selected_tickers = factor_scores.nlargest(num_positions).index.tolist()
selected_scores = factor_scores[selected_tickers]

# Calculate returns and covariance
returns = prices[selected_tickers].pct_change(fill_method=None).dropna().tail(lookback)
cov_matrix = returns.cov().values
volatility_vec = returns.std().values * np.sqrt(252)

# Base alpha (normalized factor scores)
alpha = selected_scores.values
alpha_normalized = (alpha - alpha.mean()) / alpha.std()

def solve_mvo_with_alpha(alpha_scaling):
    """Solve MVO with given alpha scaling."""
    expected_returns = alpha_normalized * alpha_scaling

    n = len(expected_returns)
    w = cp.Variable(n)

    portfolio_return = w @ expected_returns
    portfolio_variance = cp.quad_form(w, cov_matrix)
    axioma_term = 0.01 * (w @ volatility_vec)

    objective = cp.Maximize(
        portfolio_return - 1.0 * portfolio_variance - axioma_term
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0.03,  # 3% min
        w <= 0.08,  # 8% max
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if problem.status not in ['optimal', 'optimal_inaccurate']:
        return None

    weights = w.value
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    return weights

# Test range of alpha values
alpha_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

results = []

for alpha_scale in alpha_values:
    weights = solve_mvo_with_alpha(alpha_scale)

    if weights is None:
        continue

    # Count unique weight levels
    unique_weights = len(np.unique(np.round(weights * 1000) / 1000))

    # Count positions at extremes
    at_min = (weights < 0.031).sum()
    at_max = (weights > 0.079).sum()
    in_middle = num_positions - at_min - at_max

    # Calculate metrics
    weight_std = weights.std()
    weight_range = weights.max() - weights.min()

    results.append({
        'alpha': alpha_scale,
        'unique_levels': unique_weights,
        'at_min_3pct': at_min,
        'at_max_8pct': at_max,
        'in_middle': in_middle,
        'weight_std': weight_std,
        'weight_range': weight_range,
        'weights': weights
    })

# Display results
print("=" * 80)
print("RESULTS: Alpha Scaling vs Weight Distribution")
print("=" * 80)
print()
print(f"{'Alpha':>6s} | {'Unique':>7s} | {'@3%':>4s} | {'@8%':>4s} | {'Middle':>6s} | {'Std':>6s} | {'Range':>6s} | Grade")
print("-" * 80)

for r in results:
    # Grade: How gradual is the distribution?
    if r['unique_levels'] >= 10 and r['in_middle'] >= 10:
        grade = "A (Excellent)"
    elif r['unique_levels'] >= 7 and r['in_middle'] >= 7:
        grade = "B (Good)"
    elif r['unique_levels'] >= 5 and r['in_middle'] >= 5:
        grade = "C (Fair)"
    elif r['unique_levels'] >= 3 and r['in_middle'] >= 2:
        grade = "D (Poor)"
    else:
        grade = "F (Fail)"

    print(f"{r['alpha']:6.2f} | {r['unique_levels']:7d} | {r['at_min_3pct']:4d} | {r['at_max_8pct']:4d} | "
          f"{r['in_middle']:6d} | {r['weight_std']:5.1%} | {r['weight_range']:5.1%} | {grade}")

print()

# Find best configuration
best = max(results, key=lambda x: x['unique_levels'] * x['in_middle'])

print("=" * 80)
print("BEST CONFIGURATION")
print("=" * 80)
print(f"Alpha scaling: {best['alpha']:.2f}")
print(f"Unique weight levels: {best['unique_levels']}")
print(f"Positions at 3%: {best['at_min_3pct']}")
print(f"Positions at 8%: {best['at_max_8pct']}")
print(f"Positions in middle: {best['in_middle']}")
print()
print("Weight distribution:")
weights_sorted = sorted(best['weights'], reverse=True)
for i, w in enumerate(weights_sorted, 1):
    print(f"  Position {i:2d}: {w:5.1%}")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if best['in_middle'] >= 10:
    print(f"✅ SUCCESS: Alpha scaling of {best['alpha']:.2f} produces gradual distribution")
    print(f"   - {best['unique_levels']} different weight levels")
    print(f"   - {best['in_middle']} positions with intermediate weights")
    print()
    print("To implement this:")
    print(f"  1. Edit src/portfolio/optimizer.py line 483")
    print(f"  2. Change: expected_returns = alpha_normalized * {best['alpha']:.2f}")
    print(f"  3. Regenerate portfolio and validate")
else:
    print(f"❌ NO GOOD SOLUTION FOUND")
    print()
    print("Even with alpha scaling up to 0.15, we still get mostly extreme weights.")
    print("This confirms the diagnosis: the return signal is too weak relative to risk.")
    print()
    print("Options:")
    print("  1. Accept current behavior (8 @ 8%, 12 @ 3%)")
    print("  2. Switch to RankBased optimizer (gradual by design)")
    print("  3. Implement custom regularization (non-standard)")

print()
