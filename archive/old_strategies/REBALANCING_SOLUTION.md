# Portfolio Rebalancing Investigation & Solutions

**Date:** October 31, 2025
**Analysis Period:** March 3 - October 13, 2025 (33 weeks)

## Executive Summary

Paper trading backtest showed **GOOD risk-adjusted returns** (Sharpe 1.29, +7.81% total return) but **EXCESSIVE turnover** (75.2% weekly average). Investigation revealed root cause: **highly volatile factor scores** causing complete portfolio churn every week.

**Critical Finding:** ZERO ETFs were held for >75% of the backtest period - indicating severe overfitting and instability.

---

## Diagnostic Results

###  1. Turnover Analysis

| Metric | Value | Status |
|--------|-------|--------|
| Average Weekly Turnover | **75.2%** | ❌ EXCESSIVE |
| Target Turnover | <25% | ❌ FAILED |
| Weeks with >50% Turnover | 30/32 (94%) | ❌ CRITICAL |
| Max Weekly Turnover | 95% | ❌ CRITICAL |

### 2. Factor Stability Analysis

**Week-over-week factor score volatility (for portfolio holdings):**

| Factor | Avg Change | Std Dev | Assessment |
|--------|-----------|---------|------------|
| **Value** | 0.26 | 0.18 | ❌ VERY VOLATILE - Primary culprit |
| **Momentum** | 0.18 | 0.15 | ⚠️ HIGH - Secondary issue |
| **Volatility** | 0.17 | 0.15 | ⚠️ HIGH |
| **Quality** | 0.06 | 0.06 | ✓ STABLE |

**Key Finding:** Value and momentum factors are changing dramatically week-to-week, causing optimizer to select completely different portfolios.

### 3. Holding Persistence Analysis

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Unique ETFs Held | 214 | Too many (10x target) |
| ETFs held all 33 weeks | **0** | ❌ CRITICAL |
| ETFs held >75% of time | **0** | ❌ NO CORE PORTFOLIO |
| ETFs held >50% of time | **0** | ❌ COMPLETE INSTABILITY |
| ETFs held only once | 74 (35%) | ❌ EXCESSIVE CHURN |

**Most Persistent Holdings:**
- IDV: 14/33 weeks (42%) - Best
- VYMI, DWX: 11/33 weeks (33%)
- GLD, DIM: 10/33 weeks (30%)

**Interpretation:** Even the "most stable" holdings are only present ~40% of the time. This indicates the strategy has NO stable core and is essentially a different portfolio every week.

### 4. Weight Change Analysis

| Metric | Value |
|--------|-------|
| Avg weight change (continuing positions) | 2.17% |
| Positions with >2% weight change | 43% |

**Note:** Weight changes are actually reasonable - the problem is position turnover, not weight drift.

---

## Root Cause Analysis

### Primary Cause: Factor Overfitting

The factors (especially Value and Momentum) are calculated with NO SMOOTHING and react immediately to price changes. This causes:

1. **Value Factor:** Uses only expense ratios (synthetic in backtest) which change randomly, creating noise
2. **Momentum Factor:** 252-day lookback but recalculated weekly - very sensitive to recent price action
3. **No Persistence Mechanism:** Optimizer sees "new best" ETFs every week with no memory of previous holdings

### Secondary Cause: Optimizer Sensitivity

Mean-Variance Optimizer with no turnover constraints will always find the "mathematically optimal" portfolio, even if it requires 100% turnover. Small changes in factor scores → large changes in optimal weights.

### Tertiary Cause: No Rebalancing Logic

Current implementation rebalances EVERY week regardless of need. Missing:
- Drift threshold checking BEFORE optimization
- Transaction cost consideration
- Turnover penalties

---

## Recommended Solutions (Priority Order)

### 🔴 CRITICAL - Implement Immediately

#### 1. **Increase Rebalancing Threshold**
```python
# Current: Rebalances every week automatically
# Recommended: Only rebalance if drift > threshold

DRIFT_THRESHOLD = 0.15  # 15% drift required to trigger rebalance
```

**Expected Impact:** Reduce rebalances from 33 → ~10-15 per period

**Implementation:** Check portfolio drift before calling optimizer. Skip optimization if drift < threshold.

#### 2. **Factor Score Smoothing (Exponential Moving Average)**
```python
# Apply EMA to factor scores to reduce volatility
# EMA formula: smoothed_t = α × current_t + (1-α) × smoothed_{t-1}

FACTOR_EMA_ALPHA = 0.3  # Lower = more smoothing

# Example:
smoothed_momentum = 0.3 * current_momentum + 0.7 * previous_momentum
```

**Expected Impact:** Reduce factor volatility by 50-70%, leading to more stable portfolio selections

**Implementation:** Track previous week's factor scores and blend with current week before optimization.

#### 3. **Fix Value Factor**
```python
# Current: Using random synthetic expense ratios (creates noise)
# Recommended: Either:
#   a) Fetch real expense ratio data from yfinance
#   b) Remove value factor temporarily
#   c) Use P/E or P/B ratios instead

# Option A (best):
def get_real_expense_ratios(tickers):
    expense_ratios = {}
    for ticker in tickers:
        try:
            etf = yf.Ticker(ticker)
            expense_ratios[ticker] = etf.info.get('expenseRatio', 0.005)
        except:
            expense_ratios[ticker] = 0.005
    return pd.Series(expense_ratios)
```

**Expected Impact:** Remove largest source of noise (Value factor changing 0.26/week)

###  ⚠️ HIGH PRIORITY - Implement Within 1 Week

#### 4. **Add Turnover Penalty to Optimizer**

Modify optimizer to penalize deviations from current portfolio:

```python
# Pseudo-code for turnover-aware optimization
def optimize_with_turnover_penalty(
    factor_scores,
    prices,
    current_portfolio,  # NEW
    turnover_penalty=0.02  # NEW
):
    # Standard mean-variance optimization
    weights = base_optimize(factor_scores, prices)

    # Apply penalty to positions not in current portfolio
    if current_portfolio is not None:
        current_tickers = set(current_portfolio['ticker'])
        for ticker in weights.index:
            if ticker not in current_tickers:
                # Reduce weight of new positions
                weights[ticker] *= (1 - turnover_penalty)

        # Renormalize
        weights /= weights.sum()

    return weights
```

**Expected Impact:** Bias toward keeping existing positions unless significantly better alternatives exist

#### 5. **Extend Factor Lookback Windows**

```python
# Current lookback periods:
momentum_lookback = 252  # 1 year - KEEP
quality_lookback = 252   # 1 year - KEEP
volatility_lookback = 60  # 3 months - TOO SHORT

# Recommended:
volatility_lookback = 126  # 6 months (more stable)
```

**Expected Impact:** Reduce volatility factor changes by ~30%

### ℹ️ MEDIUM PRIORITY - Implement Within 2-4 Weeks

#### 6. **Change Rebalancing Frequency**

```python
# Current: Weekly
# Options:
# - Bi-weekly (every 2 weeks)
# - Monthly
# - Adaptive (rebalance when drift > threshold, check weekly)

# Recommended: Adaptive with weekly checks
rebalance_freq = "adaptive"
check_freq = "weekly"
drift_threshold = 0.15
```

**Expected Impact:** Fewer rebalances, lower costs, similar performance

#### 7. **Implement Transaction Cost Model**

```python
# Add explicit transaction costs to optimization
TRANSACTION_COST_BPS = 10  # 10 basis points (0.1%)

# Penalize turnover in objective function
def adjusted_return(expected_return, turnover):
    return expected_return - (turnover * TRANSACTION_COST_BPS / 10000)
```

**Expected Impact:** Make optimizer internalize rebalancing costs

---

## Stop-Loss Framework

To protect against severe drawdowns while maintaining weekly monitoring:

### Position-Level Stop-Losses

```python
POSITION_STOP_LOSS = -0.12  # -12% from entry price
TRAILING_STOP_PCT = 0.10    # 10% from peak

def check_position_stops(portfolio, current_prices):
    """Check each position for stop-loss triggers"""
    positions_to_close = []

    for position in portfolio:
        # Calculate return from entry
        pnl = (current_prices[position.ticker] - position.entry_price) / position.entry_price

        # Hard stop-loss
        if pnl <= POSITION_STOP_LOSS:
            positions_to_close.append(position.ticker)
            log(f"STOP-LOSS: {position.ticker} at {pnl:.2%}")

        # Trailing stop (for winners)
        elif position.peak_price is not None:
            drawdown_from_peak = (current_prices[position.ticker] - position.peak_price) / position.peak_price
            if drawdown_from_peak <= -TRAILING_STOP_PCT:
                positions_to_close.append(position.ticker)
                log(f"TRAILING STOP: {position.ticker}, -{TRAILING_STOP_PCT:.0%} from peak")

    return positions_to_close
```

### Portfolio-Level Circuit Breaker

```python
PORTFOLIO_DAILY_STOP = -0.03  # -3% daily drawdown triggers defensive mode
PORTFOLIO_PEAK_DRAWDOWN = -0.15  # -15% from peak → reduce positions

def check_portfolio_stops(portfolio_value, prev_value, peak_value):
    """Check for portfolio-level risk events"""

    # Daily circuit breaker
    daily_return = (portfolio_value - prev_value) / prev_value
    if daily_return <= PORTFOLIO_DAILY_STOP:
        return "CIRCUIT_BREAKER"  # Halt trading, review positions

    # Drawdown from peak
    drawdown = (portfolio_value - peak_value) / peak_value
    if drawdown <= PORTFOLIO_PEAK_DRAWDOWN:
        return "REDUCE_RISK"  # Cut positions to 50%, move to cash

    return "NORMAL"
```

### Volatility-Adjusted Stops

```python
def calculate_dynamic_stop(position, recent_volatility):
    """Adjust stop-loss based on instrument volatility"""
    BASE_STOP = -0.12

    # Widen stops for volatile instruments
    if recent_volatility > 0.20:  # >20% annualized vol
        stop = BASE_STOP * 1.5  # -18% for volatile ETFs
    elif recent_volatility < 0.10:  # <10% vol
        stop = BASE_STOP * 0.75  # -9% for stable ETFs
    else:
        stop = BASE_STOP

    return stop
```

---

## Proposed Implementation Plan

### Phase 1: Quick Wins (This Week)
1. **Add drift threshold check** before optimization (30 min)
2. **Implement factor score EMA** (2 hours)
3. **Fix value factor** to use real expense ratios (1 hour)
4. **Run new backtest** with these changes

**Expected Outcome:** Turnover drops from 75% → 25-35%

### Phase 2: Optimizer Improvements (Week 2)
1. **Add turnover penalty** to optimization (4 hours)
2. **Extend volatility lookback** to 126 days (5 min)
3. **Run comparative backtest**

**Expected Outcome:** Turnover drops to 15-25%, core portfolio emerges

### Phase 3: Stop-Loss Framework (Weeks 3-4)
1. **Implement position tracking** with entry prices (2 hours)
2. **Add stop-loss checking** to weekly workflow (2 hours)
3. **Add portfolio-level circuit breakers** (1 hour)
4. **Test with historical scenarios**

**Expected Outcome:** Downside protection without sacrificing returns

### Phase 4: Production Readiness (Month 2)
1. **Add transaction cost modeling**
2. **Implement adaptive rebalancing**
3. **Build monitoring dashboard**
4. **3-month forward paper trading**

---

## Recommended Configuration for Next Backtest

```python
# File: scripts/backtest_paper_trading.py (modified)

# Stability parameters
FACTOR_EMA_ALPHA = 0.3          # 30% current, 70% previous
DRIFT_THRESHOLD = 0.15          # Require 15% drift to rebalance
TURNOVER_PENALTY = 0.03         # 3% penalty for new positions

# Factor parameters
MOMENTUM_LOOKBACK = 252         # 1 year (unchanged)
QUALITY_LOOKBACK = 252          # 1 year (unchanged)
VOLATILITY_LOOKBACK = 126       # 6 months (increased from 60)

# Stop-loss parameters
POSITION_STOP_LOSS = -0.12      # -12% hard stop
TRAILING_STOP_PCT = 0.10        # 10% trailing stop
PORTFOLIO_DAILY_STOP = -0.03    # -3% daily circuit breaker
```

**Run command:**
```bash
python scripts/backtest_paper_trading_v2.py \
    --start 2025-03-01 \
    --end 2025-10-31 \
    --capital 100000 \
    --monthly-add 5000 \
    --config balanced \
    --output results/stable_backtest_balanced.xlsx
```

---

## Expected Results After Fixes

Based on the diagnostic analysis, implementing Phase 1 & 2 changes should result in:

| Metric | Current | Expected | Target |
|--------|---------|----------|--------|
| Weekly Turnover | 75.2% | **20-30%** | <25% |
| Core Holdings (>75% time) | 0 | **8-12** | >10 |
| Total Unique ETFs | 214 | **40-60** | <100 |
| Sharpe Ratio | 1.29 | **1.2-1.4** | >1.0 |
| Max Drawdown | -5.7% | **-8 to -12%** | <-15% |
| Total Return | +7.81% | **+6% to +9%** | >5% |

**Key Trade-off:** Slightly lower returns in exchange for dramatically lower turnover and more stable portfolio.

---

## Next Steps

1. **Review this document** and confirm approach
2. **Prioritize Phase 1 changes** (can be done in <4 hours)
3. **Run new backtest** with stability parameters
4. **Compare results** to current backtest
5. **If successful:** Deploy Phase 2 and begin weekly paper trading
6. **If insufficient:** Adjust parameters and iterate

---

## Appendix: Files Generated

- `results/paper_trading_backtest_20251031_191034.xlsx` - Original backtest with high turnover
- `results/rebalancing_diagnosis_20251031_192833.txt` - Detailed diagnostic report
- `scripts/diagnose_rebalancing.py` - Diagnostic script for future analysis
- `scripts/backtest_with_stability.py` - Skeleton for stable backtest (needs completion)

---

**Prepared by:** Claude (AI Assistant)
**Review Required:** Human approval before implementation
