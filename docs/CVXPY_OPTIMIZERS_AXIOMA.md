# CVXPY Optimizers with Axioma Adjustment

## Overview

We have implemented two sophisticated portfolio optimizers using CVXPY with **Axioma risk adjustment** for robustness under uncertain expected returns.

The Axioma adjustment adds a penalty term `w'σ` to the objective function, where `w` are portfolio weights and `σ` is the volatility vector. This makes the optimal portfolio more stable when expected returns are uncertain.

**Key Insight**: The Axioma adjustment ensures that the portfolio remains optimal even if expected returns change, by penalizing positions that contribute high individual risk.

---

## 1. MinVarianceOptimizer

### Purpose
Constructs a minimum variance portfolio from top-scoring ETFs, with Axioma adjustment for robustness.

### Objective Function
```
minimize: w'Σw + γ * w'σ

where:
  Σ = covariance matrix
  σ = volatility vector (annualized standard deviations)
  γ = Axioma risk penalty parameter (default: 0.01)
  w = portfolio weights
```

### Parameters
- `num_positions` (int): Number of ETFs to select (default: 20)
- `min_score` (float): Minimum factor score threshold
- `lookback` (int): Days of returns for covariance estimation (default: 60)
- `target_return` (float): Optional target portfolio return constraint
- `risk_penalty` (float): Axioma risk penalty γ (default: 0.01)

### Usage
```python
from src.portfolio import MinVarianceOptimizer

optimizer = MinVarianceOptimizer(
    num_positions=20,
    lookback=60,
    risk_penalty=0.01
)

weights = optimizer.optimize(factor_scores, prices)
```

### How It Works
1. Filter ETFs by minimum factor score (if specified)
2. Select top N ETFs by factor score
3. Calculate covariance matrix from historical returns
4. Solve convex optimization problem:
   - Minimize: variance + Axioma penalty
   - Subject to: weights sum to 1, long-only
5. Return optimal weights

### When to Use
- You want a **low-volatility** portfolio
- You have factor scores but don't want to use them as return predictions
- You prefer **risk minimization** over return maximization
- You want robustness to changes in market conditions

---

## 2. MeanVarianceOptimizer

### Purpose
Classic Markowitz mean-variance optimization with Axioma adjustment, using factor scores as expected return signal.

### Objective Function
```
maximize: μ'w - λ * w'Σw - γ * w'σ

where:
  μ = expected returns (from factor scores or historical data)
  Σ = covariance matrix
  σ = volatility vector
  λ = risk aversion parameter (default: 1.0)
  γ = Axioma penalty (default: 0.01)
  w = portfolio weights
```

### Parameters
- `num_positions` (int): Number of ETFs to select (default: 20)
- `min_score` (float): Minimum factor score threshold
- `lookback` (int): Days of returns for covariance estimation (default: 60)
- `risk_aversion` (float): Risk aversion λ (default: 1.0)
  - Higher = more conservative
  - Lower = more aggressive
- `axioma_penalty` (float): Axioma penalty γ (default: 0.01)
- `use_factor_scores_as_alpha` (bool): Use factor scores as expected returns (default: True)

### Usage
```python
from src.portfolio import MeanVarianceOptimizer

optimizer = MeanVarianceOptimizer(
    num_positions=20,
    lookback=60,
    risk_aversion=1.0,
    axioma_penalty=0.01,
    use_factor_scores_as_alpha=True
)

weights = optimizer.optimize(factor_scores, prices)
```

### How It Works
1. Filter ETFs by minimum factor score
2. Select top N ETFs by factor score
3. Calculate covariance matrix from historical returns
4. Derive expected returns:
   - If `use_factor_scores_as_alpha=True`: Normalize factor scores to ±10% alpha
   - If `False`: Use historical mean returns
5. Solve convex optimization problem:
   - Maximize: expected return - λ × variance - γ × Axioma penalty
   - Subject to: weights sum to 1, long-only, max 15% per position
6. Return optimal weights with diagnostics

### When to Use
- You want to **balance return and risk**
- You have confidence in your factor scores as return predictors
- You want control over **risk aversion** (λ parameter)
- You need **concentration limits** (15% max per position)
- You want robustness to estimation error in expected returns

---

## Axioma Adjustment Deep Dive

### What is the Axioma Adjustment?

The Axioma adjustment adds the term `γ * w'σ` to the objective function, where:
- `w` = portfolio weights
- `σ` = individual asset volatilities
- `γ` = penalty parameter

This is equivalent to penalizing the **weighted sum of individual volatilities**.

### Why Does It Work?

Traditional mean-variance optimization can be unstable when expected returns are uncertain. Small changes in expected returns can lead to large changes in optimal weights.

The Axioma adjustment makes the portfolio **robust** by:

1. **Penalizing concentrated positions in high-volatility assets**
   - Even if they have high expected returns
   - Reduces sensitivity to return estimation errors

2. **Encouraging diversification**
   - Spreads risk across multiple assets
   - Reduces exposure to any single volatile position

3. **Stabilizing the optimization**
   - Less turnover when re-optimizing
   - More stable weights over time

### Mathematical Intuition

Consider two portfolios with same variance but different compositions:
- **Portfolio A**: Concentrated in a few low-correlation, high-volatility assets
- **Portfolio B**: Diversified across many moderate-volatility assets

Both have the same total variance (w'Σw), but Portfolio A has higher `w'σ` because it's concentrated in high-volatility assets.

The Axioma penalty prefers Portfolio B because it's more robust to:
- Correlation changes
- Estimation error in expected returns
- Regime changes

### Choosing the Penalty Parameter γ

**Default: γ = 0.01**

- **γ = 0**: No Axioma adjustment (standard mean-variance)
- **γ = 0.01**: Light adjustment (recommended starting point)
- **γ = 0.05**: Moderate adjustment (more diversification)
- **γ = 0.10**: Strong adjustment (heavy diversification)

Higher γ → more diversified, more stable, potentially lower returns
Lower γ → more concentrated, less stable, potentially higher returns

### References

1. **Axioma Portfolio Optimization**
   - Axioma Inc., "Portfolio Construction and Risk Management"
   - Industry standard for robust portfolio optimization

2. **Black-Litterman with Robust Optimization**
   - Meucci, A. (2005), "Risk and Asset Allocation"
   - Discusses robustness under uncertain views

3. **Academic Foundation**
   - Michaud, R. (1989), "The Markowitz Optimization Enigma"
   - Describes instability of mean-variance optimization

---

## Comparison: MinVar vs MVO

| Feature | MinVarianceOptimizer | MeanVarianceOptimizer |
|---------|----------------------|------------------------|
| **Objective** | Minimize risk | Balance return vs risk |
| **Uses factor scores** | For selection only | As expected returns |
| **Risk aversion** | Implicit (∞) | Configurable (λ) |
| **Concentration** | No limit | 15% max per position |
| **Best for** | Low-volatility portfolio | Return-seeking portfolio |
| **Robustness** | Very high | High (with Axioma) |

---

## Test Results

All optimizers pass comprehensive tests:

### MinVarianceOptimizer Tests
- ✅ Basic optimization (20 positions, valid weights)
- ✅ Axioma penalty affects results (different HHI)
- ✅ Min score filtering works correctly
- ✅ Handles insufficient data gracefully
- ✅ Target return constraint works
- ✅ Fallback on optimization failure

### MeanVarianceOptimizer Tests
- ✅ Basic optimization with concentration limits
- ✅ Factor scores influence weights correctly
- ✅ Risk aversion affects diversification
- ✅ Axioma penalty changes weights
- ✅ Concentration limit enforced (≤15%)
- ✅ Historical returns mode works
- ✅ Min score filtering works
- ✅ Handles insufficient eligible ETFs

### Integration Tests
- ✅ All optimizers produce valid weights (sum=1, ≥0)
- ✅ MinVar and MVO produce different results
- ✅ Both achieve reasonable diversification (HHI < 0.3)

**Test Command:**
```bash
python scripts/test_cvxpy_optimizers.py
```

---

## Example: Comparing Optimizers

```python
import pandas as pd
from src.portfolio import SimpleOptimizer, MinVarianceOptimizer, MeanVarianceOptimizer

# Assume we have factor_scores and prices

# 1. Simple equal-weight
simple = SimpleOptimizer(num_positions=20)
weights_simple = simple.optimize(factor_scores)

# 2. Minimum variance with Axioma
minvar = MinVarianceOptimizer(
    num_positions=20,
    lookback=60,
    risk_penalty=0.01
)
weights_minvar = minvar.optimize(factor_scores, prices)

# 3. Mean-variance with Axioma
mvo = MeanVarianceOptimizer(
    num_positions=20,
    lookback=60,
    risk_aversion=1.0,
    axioma_penalty=0.01,
    use_factor_scores_as_alpha=True
)
weights_mvo = mvo.optimize(factor_scores, prices)

# Compare
comparison = pd.DataFrame({
    'Simple': weights_simple,
    'MinVar': weights_minvar,
    'MVO': weights_mvo
}).fillna(0)

print(comparison.head(10))
print(f"\nConcentration (HHI):")
print(f"Simple: {(weights_simple**2).sum():.4f}")
print(f"MinVar: {(weights_minvar**2).sum():.4f}")
print(f"MVO: {(weights_mvo**2).sum():.4f}")
```

---

## Recommendations

### For Most Users
Use **MeanVarianceOptimizer** with default parameters:
```python
optimizer = MeanVarianceOptimizer(
    num_positions=20,
    lookback=60,
    risk_aversion=1.0,
    axioma_penalty=0.01
)
```

**Why?**
- Balances return and risk
- Uses factor scores as alpha signal
- Concentration limits prevent over-concentration
- Axioma adjustment provides robustness

### For Conservative Investors
Use **MinVarianceOptimizer**:
```python
optimizer = MinVarianceOptimizer(
    num_positions=20,
    lookback=60,
    risk_penalty=0.02  # Higher for more diversification
)
```

**Why?**
- Minimizes volatility
- Good for risk-averse investors
- Stable in volatile markets

### For Aggressive Investors
Use **MeanVarianceOptimizer** with low risk aversion:
```python
optimizer = MeanVarianceOptimizer(
    num_positions=20,
    lookback=60,
    risk_aversion=0.5,  # Lower = more aggressive
    axioma_penalty=0.01
)
```

**Why?**
- Tilts toward high expected returns
- Accepts higher volatility
- Still has concentration limits for safety

### For Simple Approach
Use **SimpleOptimizer** (no CVXPY required):
```python
optimizer = SimpleOptimizer(num_positions=20)
```

**Why?**
- No dependencies (CVXPY not needed)
- Fast and transparent
- Equal-weight among top-scoring ETFs
- Good baseline for comparison

---

## Implementation Notes

### Dependencies
Both optimizers require:
- `cvxpy` - Convex optimization library
- `numpy` - Numerical operations
- `pandas` - Data structures

Install with:
```bash
pip install cvxpy numpy pandas
```

### Performance
- **MinVarianceOptimizer**: ~0.1-0.5 seconds per optimization
- **MeanVarianceOptimizer**: ~0.1-0.5 seconds per optimization

Performance scales with:
- Number of assets (N)
- Lookback period (affects covariance calculation)
- Solver choice (OSQP is fast for these problems)

### Solver
Both use **OSQP** (Operator Splitting QP solver):
- Fast for medium-scale problems
- Handles quadratic objectives well
- Robust convergence

Alternative solvers (if OSQP fails):
- ECOS
- SCS
- CVXOPT

---

## Future Enhancements

Potential additions:
1. **Transaction cost awareness** - Penalize turnover in optimization
2. **Tracking error constraints** - Stay close to benchmark
3. **Sector constraints** - Max allocation per sector
4. **Turnover constraints** - Limit changes from current portfolio
5. **Factor risk budgeting** - Control exposure to specific factors
6. **Regime-dependent parameters** - Adjust γ and λ based on market regime

---

## Conclusion

The CVXPY optimizers with Axioma adjustment provide:

✅ **Robustness** - Stable under uncertain expected returns
✅ **Flexibility** - Configurable risk aversion and penalties
✅ **Performance** - Fast optimization with OSQP solver
✅ **Validation** - Comprehensive test coverage
✅ **Practicality** - Concentration limits and constraints

Use them when you want sophisticated portfolio construction that goes beyond simple equal-weighting while maintaining robustness and stability.
