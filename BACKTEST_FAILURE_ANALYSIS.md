# Backtest Failure Analysis & Recommendations

## Current Performance (2-Year Backtest)

```
Initial Value:    $1,000,000
Final Value:      $1,053,919  (+5.5%)
CAGR:             2.75%
Sharpe Ratio:     -0.10  ❌ NEGATIVE!
Max Drawdown:     -12.34%
Avg Turnover:     175%  ❌ EXTREME CHURN!
Transaction Costs: $72,073
Stop-Loss Triggers: 45  ❌ EXCESSIVE!
Stop-Loss Losses: $-325,510
```

**Verdict**: Strategy is WORSE than holding cash (4% risk-free rate).

---

## Root Causes

### 1. **Signal Quality is Terrible**
**Problem**: Composite signals generate random noise, not real alpha
- RSI/MACD/Bollinger Bands on ETF prices are lagging indicators
- No fundamental or sector rotation logic
- ETFs are already diversified instruments - technical analysis on them is questionable
- Signals optimized on 1-day snapshot don't persist over time (overfitting)

**Evidence**:
- Sharpe -0.10 means signals are anti-correlated with actual performance
- 53% win rate (coin flip)
- High turnover suggests signals flip-flop constantly

### 2. **Turnover Penalty is Too Weak**
**Problem**: `turnover_penalty=0.1` is WAY too low

Current objective:
```python
objective = mu @ w - risk_aversion * portfolio_var - 0.1 * turnover
```

When expected returns are 20-30% and turnover penalty is 0.1:
- **Benefit of 10% return difference**: +10.0
- **Cost of 100% turnover**: -0.1 × 1.0 = -0.1
- **Net**: +9.9 (turnover is irrelevant!)

**Result**: Optimizer ignores turnover completely, churns 175% monthly.

### 3. **Stop-Loss is Counterproductive**
**Problem**: 10% stop-loss triggers 45 times, loses $325K

- ETFs naturally have 10-20% volatility
- Stop-loss sells temporary dips, locks in losses
- Then re-buys higher (whipsaw)
- Academic research: stop-losses REDUCE returns for diversified portfolios

### 4. **No Market Timing / Cash Allocation**
**Problem**: Always 100% invested, even in downturns

- Portfolio can't move to cash during bear markets
- No momentum/trend filters on broad market (SPY)
- Gets crushed in drawdowns with no defense

### 5. **Retrospective Overfitting**
**Problem**: Static optimization (1 snapshot) != rolling backtest

- CVXPY optimizes on 252-day lookback window
- Uses Sharpe 3-4 based on PAST data
- Future performance regresses to mean
- Classic overfitting: great in-sample, terrible out-of-sample

---

## Recommendations

### Immediate Fixes (High Priority)

#### 1. **Increase Turnover Penalty 50-100x**

```python
# Current (broken)
turnover_penalty = 0.1

# Fixed (actually penalizes churn)
turnover_penalty = 5.0 to 10.0  # Now cost of turnover matters!
```

This will force optimizer to only change positions when truly beneficial.

**Target**: <50% monthly turnover = 10 ETF changes/month with 20 positions.

#### 2. **Disable Stop-Loss (or use 20-25%)**

Stop-loss is hurting, not helping. Options:
- **Remove entirely** (let optimization handle risk)
- **Increase to 20-25%** (only catastrophic failures)
- **Use trailing stop** (not fixed from purchase price)

#### 3. **Add Market Regime Filter**

Only invest when market is in uptrend:
```python
# Simple SMA crossover on SPY
spy_sma_50 = spy.rolling(50).mean()
spy_sma_200 = spy.rolling(200).mean()

if spy_sma_50 > spy_sma_200:
    # Bull market: invest in ETFs
    allocate_to_portfolio()
else:
    # Bear market: move to cash/bonds
    allocate_to_cash(weight=0.50)  # 50% cash in downturns
```

This simple filter avoids 2008/2020-style crashes.

#### 4. **Simplify Signal Generation**

Current signals are too complex and noisy. Better approaches:

**Option A: Momentum + Value (Simple)**
```python
# 6-month momentum
momentum_6m = prices.pct_change(126)

# Relative strength vs SPY
rel_strength = (etf_return - spy_return) / spy_return

# Combine
signal = 0.7 * momentum_6m + 0.3 * rel_strength
```

**Option B: Sector Rotation**
```python
# Identify top-performing sectors
# Allocate to best sector ETFs
# Much simpler than individual ETF selection
```

**Option C: Factor-Based**
```python
# Use factor ETFs (QUAL, MTUM, USMV, SIZE)
# These already have built-in smart-beta
# Just do tactical allocation across factors
```

### Medium-Term Improvements

#### 5. **Walk-Forward Optimization**

Current approach: Train on all data, test on same data (overfitting)

Better approach:
```
Year 1: Train on 2020-2021 data → Test on 2022
Year 2: Train on 2021-2022 data → Test on 2023
Year 3: Train on 2022-2023 data → Test on 2024
```

This gives true out-of-sample performance.

#### 6. **Ensemble Signals**

Instead of one composite signal:
- Run multiple uncorrelated strategies
- Equal-weight or meta-optimize allocation
- Reduces overfitting to any single approach

#### 7. **Add Fundamental Filters**

ETF-level fundamentals:
- **AUM > $100M** (liquidity)
- **Expense ratio < 0.75%** (cost)
- **Age > 3 years** (track record)
- **Volume > 100K daily** (tradability)

### Long-Term Strategy

#### 8. **Rethink the Approach**

**Current**: Pick 20 best ETFs from 277 using technical signals
**Problem**: ETFs are already diversified - this is "diworsification"

**Better Alternatives**:

**A. Strategic Core + Tactical Satellite**
```
Core (70%): SPY, AGG, IEF  (buy & hold)
Satellite (30%): Rotate among 5-10 sector/factor ETFs
```
- Lower turnover
- Lower costs
- More robust

**B. Factor Timing**
```
Allocate across:
- Momentum (MTUM)
- Quality (QUAL)
- Low Volatility (USMV)
- Size (IJH, IWM)
- Value (VLUE)
```
- Tilt based on market regime
- Well-researched factors
- Lower complexity

**C. Global Asset Allocation**
```
Allocate across asset classes:
- US Stocks (SPY, QQQ)
- Int'l Stocks (EFA, EEM)
- Bonds (AGG, TLT)
- Commodities (GLD, DBC)
- Real Estate (VNQ)
```
- Diversification that actually works
- Simpler signals (macro trends)
- Lower turnover

---

## Proposed Grid Search Parameters

Given the above analysis, here's what to actually test:

```python
GRID_PARAMS = {
    # Core strategy type
    'strategy_type': [
        'tactical_etf',      # Current approach (20 ETFs from universe)
        'factor_rotation',   # 5-7 factor ETFs, rotate weights
        'sector_rotation',   # 11 sector ETFs, rotate weights
        'core_satellite'     # 70% SPY/AGG, 30% tactical
    ],

    # Signal approach
    'signal_type': [
        'momentum_6m',       # Simple 6-month momentum
        'momentum_3m',       # 3-month momentum
        'momentum_dual',     # 3m + 6m combined
        'rel_strength',      # Relative to SPY
        'composite_simple'   # Simplified version of current
    ],

    # Market regime filter
    'market_filter': [
        None,                # Always invested
        'sma_50_200',        # SMA crossover
        'sma_trend',         # SPY > 200 SMA
        'momentum'           # SPY 3-month momentum > 0
    ],

    # Cash allocation in bear markets
    'bear_cash_pct': [0.0, 0.30, 0.50],

    # Turnover penalty (MUCH HIGHER)
    'turnover_penalty': [1.0, 5.0, 10.0, 20.0],

    # Rebalance frequency
    'rebalance_frequency': ['monthly', 'quarterly'],

    # Stop-loss
    'stop_loss': [None, 0.20, 0.25],  # Higher or disabled

    # Number of positions
    'max_positions': [5, 10, 15, 20],
}
```

**Estimated experiments**: 3 × 5 × 4 × 3 × 4 × 2 × 3 × 4 = 17,280
**Time**: ~144 hours (6 days) @ 30s each
**Solution**: Test high-priority subsets first

### Priority Subset 1 (Quick Test - 2 hours)

```python
{
    'strategy_type': ['factor_rotation', 'sector_rotation'],
    'signal_type': ['momentum_6m', 'momentum_dual'],
    'market_filter': [None, 'sma_50_200'],
    'bear_cash_pct': [0.0, 0.50],
    'turnover_penalty': [5.0, 10.0],
    'rebalance_frequency': ['quarterly'],
    'stop_loss': [None],
    'max_positions': [10]
}
```
= 2 × 2 × 2 × 2 × 2 × 1 × 1 × 1 = **64 experiments** × 120s = 2.1 hours

### Priority Subset 2 (Deeper Dive - 8 hours)

Expand best performers from Subset 1 with more variations.

---

## Expected Realistic Performance

With proper implementation:

```
CAGR:          8-12%  (vs 2.75% currently)
Sharpe:        0.8-1.2  (vs -0.10 currently)
Max Drawdown:  -15% to -25%  (vs -12%, but with positive returns!)
Turnover:      30-60%  (vs 175% currently)
Win Rate:      55-60%  (vs 53% currently)
```

This is REALISTIC for quantitative ETF strategies, not magic but profitable.

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 days)
1. ✅ Increase turnover penalty to 10.0
2. ✅ Disable stop-loss
3. ✅ Add SMA 50/200 market filter
4. ✅ Test on same 2-year period
5. ✅ Verify improvement

### Phase 2: Signal Redesign (2-3 days)
1. Implement momentum-based signals
2. Add relative strength vs SPY
3. Test sector rotation variant
4. Compare all signal types

### Phase 3: Grid Search (3-5 days)
1. Run Priority Subset 1 (64 experiments)
2. Analyze results
3. Run Priority Subset 2 on best performers
4. Identify optimal parameters

### Phase 4: Walk-Forward Validation (2-3 days)
1. Implement walk-forward backtesting
2. Test best parameters out-of-sample
3. Verify robustness
4. Final parameter selection

**Total: 8-13 days for robust strategy**

---

## Next Steps

**IMMEDIATE**: Do you want me to:

1. **Quick fix**: Implement Phase 1 (turnover penalty + market filter) and rerun backtest to see improvement?

2. **Full grid search**: Implement the complete grid search framework with all strategy types?

3. **Focused approach**: Pick ONE strategy type (e.g., sector rotation) and optimize it thoroughly first?

4. **Something else**: Different approach entirely?

What's your preference?
