# Signal Exploration Framework - Ready to Run

## Status: ✅ COMPLETE - Ready for Execution

The comprehensive technical signal exploration framework has been built and is ready to run.

## What Was Built

### 1. Signal Exploration Grid Search (`scripts/signal_exploration_grid_search.py`)
- **767 lines** of comprehensive framework
- Tests all major technical indicator families:
  - Momentum (multiple periods: 20, 63, 126, 252 days)
  - RSI (multiple periods)
  - MACD
  - Bollinger Bands
  - Stochastic Oscillator
  - ADX (trend strength)
  - Volume Ratio
  - Price vs SMA
  - Acceleration (momentum derivatives)

- Tests **9 signal combinations** across different strategies
- Tests **3 weighting strategies** (equal, momentum_heavy, balanced)
- Tests **3 combination methods** (weighted, voting, ensemble)
- Tests **4 portfolio parameter sets** (varying turnover penalties and rebalance frequencies)
- **Total: ~324 experiments**

- **CRITICAL INNOVATION**: Measures **signal stability** (week-to-week correlation)
  - This is the key metric to understand why turnover is high
  - Identifies which signals are stable vs. noisy

### 2. Analysis Tool (`scripts/analyze_signal_exploration.py`)
- **284 lines** of post-hoc analysis
- Identifies:
  - Best performers by Sharpe Ratio
  - Correlation between signal stability and performance
  - Which indicator families work best
  - Optimal weighting strategies
  - Turnover penalty effectiveness
  - Actionable recommendations

## How to Run

### Step 1: Start Grid Search (Long Running - 5-8 hours estimated)
```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
nohup python scripts/signal_exploration_grid_search.py > /tmp/signal_exploration.log 2>&1 &
```

Monitor progress:
```bash
tail -f /tmp/signal_exploration.log
```

### Step 2: After Completion, Analyze Results
```bash
source venv/bin/activate
python scripts/analyze_signal_exploration.py
```

## Output Structure

The grid search creates a timestamped directory in `results/signal_exploration/YYYYMMDD_HHMMSS/`:

```
results/signal_exploration/20251007_123456/
├── exp_0001.json          # Individual experiment results
├── exp_0002.json
├── ...
├── exp_0324.json
├── interim_results_0010.json  # Checkpoints every 10 experiments
├── interim_results_0020.json
├── ...
├── all_results.json       # Final summary (JSON)
├── all_results.csv        # Final summary (CSV)
└── master.log             # Execution log
```

## What Each Experiment Contains

Each experiment JSON includes:
- **Performance Metrics**: Sharpe, CAGR, Max Drawdown, Sortino, Calmar
- **Turnover Metrics**: Average turnover %, number of rebalances
- **Signal Quality**: Signal stability (week-to-week correlation) - **KEY METRIC**
- **Configuration**: Which indicators, weighting, method, parameters
- **Holdings**: Final portfolio composition

## Why This Matters

The current problem:
- Backtest shows 2.75% CAGR, Sharpe -0.10 (worse than cash)
- High turnover (175%) despite high penalties
- Signals appear "great" in optimization but "abysmal" in reality

**Root cause hypothesis**: Signals are pure noise - they change completely week-to-week

**This framework will identify**:
1. Which technical signals are **stable** (don't change week-to-week)
2. Which signals are **profitable** (positive Sharpe in backtest)
3. The correlation between stability and profitability
4. Whether technical indicators work at all in this market period (2022-2024)

## Expected Outcomes

After analysis, you'll know:

### Scenario A: Profitable + Stable Signals Found ✅
- Fine-tune around winning parameters
- Validate on out-of-sample period
- Implement in production

### Scenario B: Weak Performance ⚠️
- Signals marginally profitable but not great
- Options:
  - Add market regime filter
  - Combine with other alpha sources
  - Fine-tune best configuration

### Scenario C: No Profitable Signals ❌
- Technical indicators don't work in this period
- Options:
  - Test on different time period (2017-2020)
  - Use simpler buy-and-hold with tactical cash
  - Try alternative data sources (fundamentals, sentiment)

## Next Steps After Analysis

1. Review analysis output from `analyze_signal_exploration.py`
2. Identify top 3-5 configurations worth investigating
3. Run deeper validation on those specific configs
4. Iterate based on findings

---

**Ready to run whenever you are!** 🚀

The framework is designed to run autonomously and generate comprehensive output files for analysis.
