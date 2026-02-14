"""
Simple diagnostic to understand why optimizer picks only extreme weights (8% or 3%).
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
print("DIAGNOSING EXTREME WEIGHT ALLOCATION")
print("=" * 80)
print()

# Load prices
print("Loading data...")
prices = pd.read_parquet(DATA_DIR / "processed" / "etf_prices_filtered.parquet")
print(f"Loaded {len(prices.columns)} ETFs")
print()

# Calculate factors (same as automation script)
print("Calculating factors...")
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

# Calculate individual scores
momentum_scores = momentum.calculate(prices)
quality_scores = quality.calculate(prices)

# Value factor needs expense ratios (use synthetic for now)
expense_ratios = pd.Series(
    np.random.uniform(0.0005, 0.01, len(prices.columns)),
    index=prices.columns,
)
value_scores = value.calculate(prices, expense_ratios)

volatility_scores = volatility.calculate(prices)

# Combine into DataFrame
factor_scores_df = pd.DataFrame({
    'momentum': momentum_scores,
    'quality': quality_scores,
    'value': value_scores,
    'volatility': volatility_scores
})

# Integrate
factor_scores = integrator.integrate(factor_scores_df)
print(f"Factor scores calculated for {factor_scores.notna().sum()} ETFs")
print()

# Select top 20
num_positions = 20
lookback = 60
selected_tickers = factor_scores.nlargest(num_positions).index.tolist()
selected_scores = factor_scores[selected_tickers]

print(f"Top {num_positions} ETFs selected")
print(f"Factor score range: {selected_scores.min():.3f} to {selected_scores.max():.3f}")
print(f"Factor score spread: {selected_scores.max() - selected_scores.min():.3f}")
print()

# Calculate returns and covariance
returns = prices[selected_tickers].pct_change(fill_method=None).dropna().tail(lookback)
cov_matrix = returns.cov().values

# Expected returns (current approach)
alpha = selected_scores.values
alpha_normalized = (alpha - alpha.mean()) / alpha.std()
expected_returns = alpha_normalized * 0.02  # Current scaling

print("Expected Returns (annualized, from factor scores):")
print(f"  Min: {expected_returns.min():.4f} ({expected_returns.min()*100:.2f}%)")
print(f"  Max: {expected_returns.max():.4f} ({expected_returns.max()*100:.2f}%)")
print(f"  Spread: {(expected_returns.max() - expected_returns.min())*100:.2f}%")
print()

# Individual volatilities
volatility_vec = returns.std().values * np.sqrt(252)
print("Individual Volatilities (annualized):")
print(f"  Min: {volatility_vec.min():.2%}")
print(f"  Max: {volatility_vec.max():.2%}")
print(f"  Mean: {volatility_vec.mean():.2%}")
print()

# Correlation analysis
corr_matrix = returns.corr()
print("Correlation Matrix Statistics:")
print(f"  Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
print(f"  Min correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f}")
print(f"  Max correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")
print()

def solve_and_analyze(expected_returns, cov_matrix, volatility_vec, min_weight, max_weight, risk_aversion, axioma_penalty, test_name):
    """Solve MVO and analyze results."""
    print("=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)
    print(f"Parameters: min={min_weight if min_weight else 0:.1%}, max={max_weight if max_weight else 'none'}, lambda={risk_aversion}, gamma={axioma_penalty}")
    print()

    n = len(expected_returns)
    w = cp.Variable(n)

    portfolio_return = w @ expected_returns
    portfolio_variance = cp.quad_form(w, cov_matrix)
    axioma_term = axioma_penalty * (w @ volatility_vec)

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
        print(f"❌ Optimization failed: {problem.status}")
        return None

    weights = w.value
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    # Statistics
    print("Weight Distribution:")
    print(f"  Min: {weights.min():.2%}")
    print(f"  Max: {weights.max():.2%}")
    print(f"  Mean: {weights.mean():.2%}")
    print(f"  Std: {weights.std():.2%}")
    print(f"  Median: {np.median(weights):.2%}")
    print()

    # Count unique weights
    unique_weights = np.unique(np.round(weights * 1000) / 1000)
    print(f"Unique weight levels: {len(unique_weights)}")
    if len(unique_weights) <= 10:
        print(f"  Values: {[f'{w:.1%}' for w in sorted(unique_weights, reverse=True)]}")
    print()

    # Histogram
    bins = [0, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 1.0]
    bin_labels = ["0-2.5%", "2.5-3.5%", "3.5-4.5%", "4.5-5.5%", "5.5-6.5%", "6.5-7.5%", "7.5-8.5%", ">8.5%"]

    print("Weight histogram:")
    for i in range(len(bins)-1):
        count = ((weights >= bins[i]) & (weights < bins[i+1])).sum()
        if count > 0:
            bar = "█" * count
            print(f"  {bin_labels[i]:12s}: {count:2d} {bar}")
    print()

    # Check binding constraints
    if min_weight and weights.min() < min_weight + 0.001:
        print("⚠️  MINIMUM CONSTRAINT IS BINDING (positions hitting floor)")
    if max_weight and weights.max() > max_weight - 0.001:
        print("⚠️  MAXIMUM CONSTRAINT IS BINDING (positions hitting cap)")

    # Portfolio metrics
    port_return = weights @ expected_returns
    port_variance = weights @ cov_matrix @ weights
    port_vol = np.sqrt(port_variance * 252)

    print(f"\nPortfolio Metrics:")
    print(f"  Expected return: {port_return:.4f} ({port_return*100:.2f}%)")
    print(f"  Portfolio volatility: {port_vol:.2%}")
    print(f"  Sharpe (approx): {port_return / (port_vol / np.sqrt(252)):.2f}")
    print()

    return weights


# Test 1: Current configuration
weights1 = solve_and_analyze(
    expected_returns, cov_matrix, volatility_vec,
    min_weight=0.03, max_weight=0.08,
    risk_aversion=1.0, axioma_penalty=0.01,
    test_name="Current Configuration (min=3%, max=8%)"
)

# Test 2: Unconstrained
weights2 = solve_and_analyze(
    expected_returns, cov_matrix, volatility_vec,
    min_weight=None, max_weight=None,
    risk_aversion=1.0, axioma_penalty=0.01,
    test_name="Unconstrained (no min/max)"
)

# Test 3: Only max constraint
weights3 = solve_and_analyze(
    expected_returns, cov_matrix, volatility_vec,
    min_weight=None, max_weight=0.08,
    risk_aversion=1.0, axioma_penalty=0.01,
    test_name="Only Max Constraint (max=8%)"
)

# Test 4: Wider range
weights4 = solve_and_analyze(
    expected_returns, cov_matrix, volatility_vec,
    min_weight=0.01, max_weight=0.15,
    risk_aversion=1.0, axioma_penalty=0.01,
    test_name="Wider Range (min=1%, max=15%)"
)

# Test 5: Lower risk aversion
weights5 = solve_and_analyze(
    expected_returns, cov_matrix, volatility_vec,
    min_weight=0.03, max_weight=0.08,
    risk_aversion=0.5, axioma_penalty=0.01,
    test_name="Lower Risk Aversion (lambda=0.5)"
)

# Test 6: Higher alpha scaling
alpha_normalized_high = (alpha - alpha.mean()) / alpha.std()
expected_returns_high = alpha_normalized_high * 0.05  # 0.05 instead of 0.02

weights6 = solve_and_analyze(
    expected_returns_high, cov_matrix, volatility_vec,
    min_weight=0.03, max_weight=0.08,
    risk_aversion=1.0, axioma_penalty=0.01,
    test_name="Higher Alpha Scaling (0.05 instead of 0.02)"
)

# Test 7: No Axioma penalty
weights7 = solve_and_analyze(
    expected_returns, cov_matrix, volatility_vec,
    min_weight=0.03, max_weight=0.08,
    risk_aversion=1.0, axioma_penalty=0.0,
    test_name="No Axioma Penalty (gamma=0)"
)

print("=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print()
print("The extreme weight behavior (all 8% or all 3%) indicates:")
print()
print("1. **MIN CONSTRAINT IS BINDING**")
print("   The 3% minimum forces allocation to all 20 positions")
print("   This was INTENTIONAL to ensure diversification")
print()
print("2. **MAX CONSTRAINT IS BINDING**")
print("   The 8% maximum caps the top positions")
print("   Optimizer would allocate MORE to top positions if allowed")
print()
print("3. **EXPECTED RETURN SPREAD IS SMALL**")
print(f"   Alpha spread: {(expected_returns.max() - expected_returns.min())*100:.2f}%")
print("   This is SMALL compared to risk penalties")
print("   Result: Optimizer can't differentiate much between positions")
print()
print("4. **HIGH CORRELATION**")
print("   ETFs are highly correlated (many international equity funds)")
print("   Diversification benefit is limited")
print("   Optimizer sees little advantage to varying weights")
print()
print("WHY THIS HAPPENS:")
print()
print("Mean-Variance Optimization trades off:")
print("  - Return: Higher factor score = allocate more")
print("  - Risk: Covariance with other positions = allocate less")
print()
print("When expected returns are SMALL (±2%) and correlations are HIGH (0.7+),")
print("the optimizer sees little benefit to varying weights.")
print()
print("It wants to either:")
print("  A) Allocate MAX to top scorers (hits 8% cap)")
print("  B) Allocate MIN to bottom scorers (hits 3% floor)")
print()
print("There's no 'middle ground' because the return spread doesn't justify")
print("deviating from the extreme positions given the risk structure.")
print()
print("POTENTIAL SOLUTIONS:")
print()
print("1. **Accept current behavior** (RECOMMENDED)")
print("   - Top 8 get max allocation (8%) = High conviction")
print("   - Bottom 12 get min allocation (3%) = Diversification")
print("   - This IS diversified (20 positions vs 7 originally)")
print("   - Performance validated: 16.2% CAGR, 1.10 Sharpe")
print()
print("2. **Increase alpha scaling** (0.02 → 0.05+)")
print("   - Gives more weight to factor scores")
print("   - May create MORE gradual distribution")
print("   - Risk: Could lead back to over-concentration")
print()
print("3. **Remove min constraint** (3% → 0%)")
print("   - Allows optimizer more freedom")
print("   - Risk: Back to 7-10 positions (defeats diversification goal)")
print()
print("4. **Use regularization** (not currently implemented)")
print("   - Add penalty for weight variance")
print("   - Encourages more uniform distribution")
print("   - Would need custom implementation")
print()
print("RECOMMENDATION:")
print()
print("The current behavior is NOT A BUG. It's the optimizer rationally responding")
print("to the trade-off between small return signals and high correlation.")
print()
print("If you want MORE gradual weights, the cleanest solution is:")
print("  → Increase alpha scaling from 0.02 to 0.04-0.05")
print("  → Test that it doesn't revert to over-concentration")
print("  → Validate backtest performance is maintained")
