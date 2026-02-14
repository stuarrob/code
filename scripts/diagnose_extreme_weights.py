"""
Diagnostic script to understand why optimizer picks only extreme weights (8% or 3%).

This script will:
1. Load the current portfolio data
2. Reproduce the optimization problem
3. Analyze the optimization landscape
4. Test different configurations to find gradual weight distribution
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_filtered_prices
from src.factors import MomentumFactor, QualityFactor, ValueFactor, VolatilityFactor
from src.factors.factor_integrator import FactorIntegrator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_optimization_problem(factor_scores, prices, num_positions=20, lookback=60):
    """Analyze why optimizer picks extremes."""

    print("=" * 80)
    print("DIAGNOSING EXTREME WEIGHT ALLOCATION")
    print("=" * 80)
    print()

    # Select top 20
    selected_tickers = factor_scores.nlargest(num_positions).index.tolist()
    selected_scores = factor_scores[selected_tickers]

    print(f"Selected {len(selected_tickers)} ETFs")
    print(f"Factor score range: {selected_scores.min():.3f} to {selected_scores.max():.3f}")
    print(f"Factor score spread: {selected_scores.max() - selected_scores.min():.3f}")
    print()

    # Calculate returns and covariance
    returns = prices[selected_tickers].pct_change().dropna().tail(lookback)
    cov_matrix = returns.cov().values

    # Expected returns (current approach)
    alpha = selected_scores.values
    alpha_normalized = (alpha - alpha.mean()) / alpha.std()
    expected_returns = alpha_normalized * 0.02  # Current scaling

    print("Expected Returns (annualized):")
    print(f"  Min: {expected_returns.min():.4f} ({expected_returns.min()*100:.2f}%)")
    print(f"  Max: {expected_returns.max():.4f} ({expected_returns.max()*100:.2f}%)")
    print(f"  Spread: {(expected_returns.max() - expected_returns.min())*100:.2f}%")
    print()

    # Individual volatilities
    volatility = returns.std().values * np.sqrt(252)
    print("Individual Volatilities (annualized):")
    print(f"  Min: {volatility.min():.2%}")
    print(f"  Max: {volatility.max():.2%}")
    print(f"  Mean: {volatility.mean():.2%}")
    print()

    # Test current configuration
    print("=" * 80)
    print("TEST 1: Current Configuration (min=3%, max=8%)")
    print("=" * 80)

    weights = solve_mvo(
        expected_returns, cov_matrix, volatility,
        min_weight=0.03, max_weight=0.08,
        risk_aversion=1.0, axioma_penalty=0.01
    )

    analyze_weight_distribution(weights, selected_tickers, selected_scores)

    # Test without min constraint
    print()
    print("=" * 80)
    print("TEST 2: No Minimum Constraint (min=0%, max=8%)")
    print("=" * 80)

    weights = solve_mvo(
        expected_returns, cov_matrix, volatility,
        min_weight=0.0, max_weight=0.08,
        risk_aversion=1.0, axioma_penalty=0.01
    )

    analyze_weight_distribution(weights, selected_tickers, selected_scores)

    # Test with wider range
    print()
    print("=" * 80)
    print("TEST 3: Wider Range (min=2%, max=10%)")
    print("=" * 80)

    weights = solve_mvo(
        expected_returns, cov_matrix, volatility,
        min_weight=0.02, max_weight=0.10,
        risk_aversion=1.0, axioma_penalty=0.01
    )

    analyze_weight_distribution(weights, selected_tickers, selected_scores)

    # Test with higher risk aversion
    print()
    print("=" * 80)
    print("TEST 4: Higher Risk Aversion (lambda=2.0, min=3%, max=8%)")
    print("=" * 80)

    weights = solve_mvo(
        expected_returns, cov_matrix, volatility,
        min_weight=0.03, max_weight=0.08,
        risk_aversion=2.0, axioma_penalty=0.01
    )

    analyze_weight_distribution(weights, selected_tickers, selected_scores)

    # Test with lower risk aversion
    print()
    print("=" * 80)
    print("TEST 5: Lower Risk Aversion (lambda=0.5, min=3%, max=8%)")
    print("=" * 80)

    weights = solve_mvo(
        expected_returns, cov_matrix, volatility,
        min_weight=0.03, max_weight=0.08,
        risk_aversion=0.5, axioma_penalty=0.01
    )

    analyze_weight_distribution(weights, selected_tickers, selected_scores)

    # Test with stronger Axioma penalty
    print()
    print("=" * 80)
    print("TEST 6: Stronger Axioma Penalty (gamma=0.05, min=3%, max=8%)")
    print("=" * 80)

    weights = solve_mvo(
        expected_returns, cov_matrix, volatility,
        min_weight=0.03, max_weight=0.08,
        risk_aversion=1.0, axioma_penalty=0.05
    )

    analyze_weight_distribution(weights, selected_tickers, selected_scores)

    # Test with NO box constraints (just analyze the unconstrained solution)
    print()
    print("=" * 80)
    print("TEST 7: Unconstrained Solution (no min/max)")
    print("=" * 80)

    weights = solve_mvo(
        expected_returns, cov_matrix, volatility,
        min_weight=None, max_weight=None,
        risk_aversion=1.0, axioma_penalty=0.01
    )

    analyze_weight_distribution(weights, selected_tickers, selected_scores)

    print()
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    print("The extreme weight behavior (all 8% or all 3%) suggests:")
    print("1. The min/max constraints are BINDING (optimizer hits limits)")
    print("2. The optimization objective doesn't differentiate much between positions")
    print("3. The expected return spread may be too small relative to risk penalties")
    print()
    print("To achieve gradual weights, we need to either:")
    print("A. Remove or relax the min/max constraints")
    print("B. Increase the expected return signal strength (alpha scaling)")
    print("C. Reduce the risk penalties (risk_aversion or axioma_penalty)")
    print("D. Use a different objective function (e.g., regularization)")


def solve_mvo(expected_returns, cov_matrix, volatility,
              min_weight, max_weight, risk_aversion, axioma_penalty):
    """Solve mean-variance optimization."""

    n = len(expected_returns)
    w = cp.Variable(n)

    portfolio_return = w @ expected_returns
    portfolio_variance = cp.quad_form(w, cov_matrix)
    axioma_term = axioma_penalty * (w @ volatility)

    objective = cp.Maximize(
        portfolio_return - risk_aversion * portfolio_variance - axioma_term
    )

    constraints = [cp.sum(w) == 1]

    if min_weight is not None:
        constraints.append(w >= min_weight)
    else:
        constraints.append(w >= 0)

    if max_weight is not None:
        constraints.append(w <= max_weight)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if problem.status not in ['optimal', 'optimal_inaccurate']:
        print(f"WARNING: Optimization status: {problem.status}")
        return None

    return w.value


def analyze_weight_distribution(weights, tickers, scores):
    """Analyze the weight distribution."""

    if weights is None:
        print("Optimization failed!")
        return

    # Clean up
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    # Statistics
    print(f"Weight Distribution:")
    print(f"  Min: {weights.min():.2%}")
    print(f"  Max: {weights.max():.2%}")
    print(f"  Mean: {weights.mean():.2%}")
    print(f"  Std: {weights.std():.2%}")
    print(f"  Median: {np.median(weights):.2%}")
    print()

    # Count unique weights (rounded to 0.1%)
    unique_weights = np.unique(np.round(weights * 1000) / 1000)
    print(f"Unique weight levels: {len(unique_weights)}")

    if len(unique_weights) <= 5:
        print("  Weight levels:", [f"{w:.1%}" for w in unique_weights])
    print()

    # Show distribution
    print("Weight histogram:")
    bins = [0, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 1.0]
    bin_labels = ["0-2.5%", "2.5-3.5%", "3.5-4.5%", "4.5-5.5%", "5.5-6.5%", "6.5-7.5%", "7.5-8.5%", ">8.5%"]

    for i in range(len(bins)-1):
        count = ((weights >= bins[i]) & (weights < bins[i+1])).sum()
        if count > 0:
            print(f"  {bin_labels[i]:12s}: {count:2d} positions")
    print()

    # Check if constraints are binding
    if weights.min() > 0.029 and weights.min() < 0.031:
        print("⚠️  MINIMUM CONSTRAINT IS BINDING (positions hitting 3% floor)")

    if weights.max() > 0.079 and weights.max() < 0.081:
        print("⚠️  MAXIMUM CONSTRAINT IS BINDING (positions hitting 8% cap)")

    # Top 10 positions
    print()
    print("Top 10 positions:")
    df = pd.DataFrame({
        'ticker': tickers,
        'weight': weights,
        'score': scores.values
    })
    df_sorted = df.sort_values('weight', ascending=False).head(10)

    for _, row in df_sorted.iterrows():
        print(f"  {row['ticker']:8s} {row['weight']:5.1%}  (score: {row['score']:.3f})")


if __name__ == "__main__":
    print("Loading data...")

    # Load prices
    prices = load_filtered_prices()

    # Calculate factors
    momentum = MomentumFactor(lookback=252, skip_recent=21)
    quality = QualityFactor(lookback=252)
    value = ValueFactor()
    volatility = VolatilityFactor(lookback=60)

    integrator = FactorIntegrator(
        momentum_weight=0.25,
        quality_weight=0.25,
        value_weight=0.25,
        volatility_weight=0.25
    )

    # Calculate scores
    momentum_scores = momentum.calculate(prices)
    quality_scores = quality.calculate(prices)
    value_scores = value.calculate(prices)
    volatility_scores = volatility.calculate(prices)

    factor_scores = integrator.integrate(
        momentum_scores,
        quality_scores,
        value_scores,
        volatility_scores
    )

    print(f"Loaded {len(factor_scores)} ETFs with factor scores")
    print()

    # Run analysis
    analyze_optimization_problem(factor_scores, prices)
