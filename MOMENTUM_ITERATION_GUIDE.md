# Momentum Strategy Iteration Guide

## Goal
Find parameters for "run with winners, sell losers" that work out-of-sample with:
- **Positive Sharpe Ratio** (>0.5 target)
- **Low Turnover** (<50% weekly target)
- **Few ETF changes** (<10 per week)
- **Good CAGR** (8-12% realistic)

## Current Status

**Baseline test running**: 6-month momentum, turnover penalty 50, weekly rebalancing, 90% stop-loss

**Early signs** (from logs):
- Turnover still 120-190% (HIGH - optimizer ignoring penalty again!)
- In-sample Sharpe predictions: 1.8-3.7 (excellent)
- Need to see out-of-sample actual Sharpe from backtest

## How to Iterate

### Step 1: Run Baseline Test
```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
python scripts/backtest_weekly_momentum.py
```

**Check results**:
- Log: `/tmp/momentum_backtest_baseline.log`
- Results: `results/momentum_backtests/test_TIMESTAMP/`
- Graphics: `results/momentum_backtests/test_TIMESTAMP/*.png`

### Step 2: Try Different Parameters

**Test one variable at a time**:

```bash
# Longer momentum (1 year instead of 6 months)
python scripts/backtest_weekly_momentum.py --momentum-period 252

# Shorter momentum (3 months)
python scripts/backtest_weekly_momentum.py --momentum-period 63

# Higher turnover penalty
python scripts/backtest_weekly_momentum.py --turnover-penalty 100.0

# Even higher
python scripts/backtest_weekly_momentum.py --turnover-penalty 200.0

# Disable relative strength (simpler signal)
python scripts/backtest_weekly_momentum.py --no-rel-strength

# Disable trend filter
python scripts/backtest_weekly_momentum.py --no-trend-filter

# Fewer positions (force concentration)
python scripts/backtest_weekly_momentum.py --max-positions 10

# No stop-loss
python scripts/backtest_weekly_momentum.py --no-stop-loss
```

### Step 3: Compare All Tests

```bash
python scripts/compare_momentum_tests.py
```

This shows:
- All tests side-by-side
- Which parameters gave best Sharpe
- Which had lowest turnover
- Parameter impact analysis

### Step 4: Run Batch Overnight

```bash
nohup ./scripts/iterate_momentum_params.sh > /tmp/momentum_iteration.log 2>&1 &
```

This runs 9 different parameter combinations automatically.

---

## Understanding Results

### Good Signs
✅ **Positive Sharpe** (>0.0)
✅ **Low turnover** (<80%)
✅ **Stable positions** (<15 ETF changes/week)
✅ **Few stop-losses** (<20 per year)
✅ **Win rate** (>52%)

### Warning Signs
⚠️ **Negative Sharpe** - Signals don't work
⚠️ **High turnover** (>150%) - Ignoring turnover penalty
⚠️ **Many stops** (>40/year) - Stop-loss counterproductive
⚠️ **Low win rate** (<50%) - Random picks

### Critical Issues
❌ **Sharpe < -0.5** - Strategy is broken
❌ **Turnover > 200%** - Complete portfolio replacement weekly
❌ **Stops > 60/year** - Sell winners too early

---

## Parameter Recommendations

### If Results Are Good (Sharpe > 0.5)
1. **Fine-tune around winning values**
   - Try momentum ±20 days
   - Try turnover penalty ±20

2. **Test longer period**
   - Extend backtest to 3-5 years
   - Check different market regimes

3. **Walk-forward validation**
   - Train on 2020-2022
   - Test on 2023-2024
   - Verify out-of-sample

### If Results Are Mediocre (Sharpe 0-0.5)
1. **Try simpler signals**
   - Pure 12-month momentum only
   - No relative strength
   - No trend filter

2. **Increase turnover penalty**
   - 100, 200, even 500
   - Force lower churn

3. **Change rebalancing**
   - Try monthly instead of weekly
   - Reduce transaction costs

### If Results Are Bad (Sharpe < 0)
1. **Signals are fundamentally broken**
   - Try 12-month momentum ONLY (academic standard)
   - Consider sector rotation instead
   - Test on different time periods

2. **Check market conditions**
   - Momentum doesn't work in all regimes
   - 2022 was terrible for momentum (inflation/rates)
   - Try 2017-2020 period

3. **Alternative approaches**
   - Buy-and-hold SPY + tactical cash
   - Sector rotation (11 sectors only)
   - Factor investing (MTUM, QUAL, USMV)

---

## Common Issues & Solutions

### Issue: Turnover Still 150%+ Despite High Penalty

**Cause**: Signals are too noisy, optimizer sees "better" ETFs every week

**Solutions**:
1. Increase penalty to 200-500
2. Use monthly rebalancing (not weekly)
3. Simplify signals (remove RSI/MACD, use pure momentum)
4. Add minimum holding period constraint

### Issue: Stop-Loss Triggers Too Much

**Cause**: 90% stop is too tight for normal ETF volatility

**Solutions**:
1. Widen stop to 85% (15% loss)
2. Use trailing stop instead of fixed
3. Disable stop-loss entirely
4. Only use stops on new positions (not rebalances)

### Issue: Negative Sharpe But High In-Sample Predictions

**Cause**: Classic overfitting - optimization sees patterns that don't persist

**Solutions**:
1. Simpler signals with fewer parameters
2. Longer momentum periods (less noise)
3. Walk-forward testing (never optimize on test data)
4. Accept that current period may be unfavorable

---

## File Locations

### Results
- `results/momentum_backtests/test_TIMESTAMP/`
  - `results.json` - All metrics
  - `params.json` - Parameters used
  - `portfolio_value.png` - Equity curve
  - `drawdown.png` - Drawdown chart
  - `rolling_metrics.png` - Rolling Sharpe/volatility
  - `rebalancing_impact.png` - Turnover and costs

### Logs
- `/tmp/momentum_backtest_baseline.log` - Full backtest log
- `/tmp/momentum_iteration.log` - Batch iteration log

### Code
- `src/signals/momentum_signal.py` - Signal generator
- `scripts/backtest_weekly_momentum.py` - Main backtest script
- `scripts/compare_momentum_tests.py` - Comparison tool
- `scripts/iterate_momentum_params.sh` - Batch runner

---

## Next Steps After Finding Good Parameters

### 1. Validate Out-of-Sample
```python
# Modify backtest dates in script
start_date = "2017-01-01"  # Train period
test_start = "2022-01-01"  # Test period
```

### 2. Calculate Transaction Costs Accurately
- Check actual broker rates
- Include slippage estimates
- Model market impact

### 3. Implement Walk-Forward
- Rolling 2-year train, 6-month test
- Optimize parameters on train only
- Test on out-of-sample data
- More realistic performance

### 4. Add Risk Management
- Max drawdown circuit breaker
- Volatility scaling
- Cash allocation in downtrends

### 5. Production Deployment
- Connect to Interactive Brokers API
- Automate weekly rebalancing
- Monitor performance
- Alert on anomalies

---

## Expected Realistic Performance

**If you find good parameters**:
- CAGR: 8-15%
- Sharpe: 0.6-1.2
- Max Drawdown: 15-30%
- Turnover: 30-80%
- Win Rate: 52-58%

**This is realistic** for quantitative momentum strategies. Don't expect 20%+ CAGR with Sharpe >2 - that's overfitting.

---

## When to Give Up on Current Approach

If after testing 20+ parameter combinations:
- All have negative Sharpe
- All have >150% turnover
- All worse than buy-and-hold SPY

Then **the signals don't work in this period**. Try:
1. Different time period (2017-2020 instead of 2022-2024)
2. Simpler strategy (12-month momentum only, no optimization)
3. Different approach (sector rotation, factor investing)
4. Accept buy-and-hold may be best

**Remember**: Not all periods favor momentum. 2022 was particularly bad (inflation + rising rates). Test on 2017-2020 for better momentum environment.
