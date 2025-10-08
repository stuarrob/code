# Summary: Momentum Iteration System

## What I've Built

A complete system for iteratively finding "run with winners, sell losers" parameters that work out-of-sample.

### 1. Simple Momentum Signal Generator
**File**: `src/signals/momentum_signal.py`

- **6-month price momentum** (most robust according to research)
- **Relative strength vs SPY** (outperformers only)
- **Trend filter** (price > 200-day SMA)
- **Minimal parameters** (reduces overfitting)

### 2. Weekly Backtest with 90% Stop-Loss
**File**: `scripts/backtest_weekly_momentum.py`

- Weekly rebalancing (agile)
- 90% stop-loss (defensive - catches losing trades)
- High turnover penalty (force low churn)
- Comprehensive output (metrics, charts, logs)

**Run with**:
```bash
python scripts/backtest_weekly_momentum.py
```

**Customize**:
```bash
python scripts/backtest_weekly_momentum.py --momentum-period 252  # 1 year
python scripts/backtest_weekly_momentum.py --turnover-penalty 100.0
python scripts/backtest_weekly_momentum.py --no-rel-strength
```

### 3. Comparison Tool
**File**: `scripts/compare_momentum_tests.py`

Analyzes ALL backtest runs side-by-side:
- Best Sharpe Ratio
- Lowest turnover
- Parameter impact analysis
- Actionable recommendations

**Run with**:
```bash
python scripts/compare_momentum_tests.py
```

### 4. Batch Iteration Script
**File**: `scripts/iterate_momentum_params.sh`

Tests 9 parameter combinations automatically:
- 3 momentum periods (63, 126, 252 days)
- 3 turnover penalties (20, 50, 100)
- With/without relative strength
- With/without trend filter

**Run overnight**:
```bash
chmod +x scripts/iterate_momentum_params.sh
nohup ./scripts/iterate_momentum_params.sh > /tmp/momentum_iteration.log 2>&1 &
```

### 5. Comprehensive Documentation
- [BACKTEST_FAILURE_ANALYSIS.md](BACKTEST_FAILURE_ANALYSIS.md) - Why current approach failed
- [MOMENTUM_ITERATION_GUIDE.md](MOMENTUM_ITERATION_GUIDE.md) - How to iterate parameters
- [GRID_SEARCH_README.md](GRID_SEARCH_README.md) - Large-scale parameter search

---

## How This Addresses Your Concerns

### ✅ "Ensure backtests run well with few signals"
- Simple momentum signals (3 components vs 10+ before)
- Minimal parameters to tune
- Less prone to overfitting

### ✅ "Clearly overfitting with current optimizer"
- Separation of signal generation from optimization
- Out-of-sample testing via backtest
- Walk-forward validation possible
- Can compare in-sample predictions vs actual results

### ✅ "Continuously review parameters"
- Easy to run with different parameters
- Comparison tool shows what works
- Iterate until you find good signal

### ✅ "Appropriate technical signals out-of-sample"
- 6/12-month momentum (most robust factor academically)
- Can test different periods easily
- Relative strength (vs SPY) optional
- Trend filter optional

### ✅ "Run with winners, sell losers"
- Momentum naturally favors winners
- 90% stop-loss catches losers
- High turnover penalty reduces churn
- Only changes positions when truly beneficial

---

## Current Status

### Baseline Test Running:
- **Parameters**: 126-day momentum, turnover penalty 50, weekly rebalancing
- **Log**: `/tmp/momentum_backtest_baseline.log`
- **Results**: `results/momentum_backtests/test_TIMESTAMP/`

**Early observations** (from logs):
- Turnover still 120-190% (high - optimizer still churning)
- In-sample Sharpe predictions: 1.8-3.7 (excellent on historical data)
- Need final out-of-sample Sharpe to judge success

### Quick Fix Test (Completed Earlier):
- **CAGR**: 2.94%
- **Sharpe**: -0.07 ❌
- **Turnover**: 177% ❌

This confirmed the complex signals (RSI/MACD/Bollinger) don't work.

---

## What To Do Next

### Option 1: Wait for Baseline to Finish
Check `/tmp/momentum_backtest_baseline.log` for results.

If **Sharpe > 0.5**: Good! Fine-tune parameters
If **Sharpe 0-0.5**: Mediocre. Try simpler (pure 12-month momentum)
If **Sharpe < 0**: Signals still broken. Try different time period or strategy

### Option 2: Run Batch Iteration Overnight
```bash
nohup ./scripts/iterate_momentum_params.sh > /tmp/momentum_iteration.log 2>&1 &
```

This tests 9 combinations. Tomorrow morning:
```bash
python scripts/compare_momentum_tests.py
```

See which parameters work best.

### Option 3: Modify Parameters Yourself
Edit `scripts/backtest_weekly_momentum.py`:
```python
PARAMS = {
    'momentum_period': 252,        # Change to 1 year
    'turnover_penalty': 200.0,     # Even higher
    'rel_strength_enabled': False, # Disable for simplicity
    ...
}
```

Then run:
```bash
python scripts/backtest_weekly_momentum.py
```

---

## Why Turnover Is Still High

Even with 50x higher turnover penalty (0.2 → 10.0), turnover stayed ~175%.

**Root cause**: The signals are too noisy. Every week the optimizer sees "better" ETFs because:
1. Signals fluctuate randomly week-to-week
2. Optimizer thinks new positions will have Sharpe 2-3
3. Reality: they perform like old positions (Sharpe -0.07)
4. Classic overfitting

**Solutions to try**:
1. **Even higher turnover penalty** (100, 200, 500)
2. **Monthly rebalancing** instead of weekly
3. **Simpler signals** (pure 12-month momentum only, nothing else)
4. **Minimum holding period** (force 4+ week holds)

---

## Realistic Expectations

If you find good parameters:
- **CAGR**: 8-12% (not 20%+)
- **Sharpe**: 0.6-1.2 (not 2+)
- **Turnover**: 30-60% (not <20%)
- **Max Drawdown**: 15-25%

This is **realistic for momentum strategies**. Higher numbers = overfitting.

---

## When Signals Actually Work

Momentum works best in:
- **Trending markets** (2017-2020)
- **Low inflation** periods
- **Stable interest rates**

Momentum fails in:
- **High inflation** (2021-2022)
- **Rising rates** rapidly
- **Mean-reverting markets**

Your 2023-2024 test period may be challenging for momentum. Try 2017-2020 to verify signals work in better environment.

---

## Files for You to Run

### Single Test
```bash
python scripts/backtest_weekly_momentum.py --momentum-period 252
```

### Compare All Tests
```bash
python scripts/compare_momentum_tests.py
```

### Batch Overnight
```bash
nohup ./scripts/iterate_momentum_params.sh > /tmp/momentum_iteration.log 2>&1 &
```

### Check Status
```bash
tail -f /tmp/momentum_backtest_baseline.log
```

---

## Next Steps

1. **Wait for baseline** to finish (check `/tmp/momentum_backtest_baseline.log`)
2. **Review results** in `results/momentum_backtests/test_TIMESTAMP/`
3. **If good**: Fine-tune and validate
4. **If bad**: Run batch iteration or try simpler signals
5. **Compare** all tests with `compare_momentum_tests.py`
6. **Iterate** until Sharpe > 0.5 and turnover < 80%

---

## Summary

I've built you a **complete iterative backtesting system** that lets you:
- Test different momentum parameters quickly
- See which ones work out-of-sample
- Reduce overfitting via simple signals
- Continuously refine "run with winners, sell losers"

The system is **ready to use** - just run the scripts and iterate until you find parameters that work in your target period.

**My assessment**: The 2023-2024 period may be fundamentally difficult for momentum. Consider testing 2017-2020 or accepting that buy-and-hold SPY might be best for recent years.
