# Grid Search Experiment - How to Run

## Quick Start

### Run overnight grid search:
```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
nohup python scripts/autonomous_grid_search.py > /tmp/grid_search_master.log 2>&1 &
```

This will:
- Test **72 parameter combinations** (2-3 hours)
- Create comprehensive output files in `results/grid_search/YYYYMMDD_HHMMSS/`
- Generate analysis automatically

## What Gets Created

### Directory Structure:
```
results/grid_search/20251007_213045/
├── grid_search_master.log          # Master log file
├── experiment_0001.log              # Individual experiment logs
├── experiment_0001_results.json     # Detailed results per experiment
├── experiment_0002.log
├── experiment_0002_results.json
├── ...
├── all_results.csv                  # ALL results in one CSV (easy to analyze)
├── ANALYSIS_SUMMARY.md              # Top performers, parameter importance
├── interim_results_0005.json        # Checkpoints every 5 experiments
├── interim_results_0010.json
├── ...
└── final_results.json               # Complete results JSON
```

### Key Files to Review:

1. **`ANALYSIS_SUMMARY.md`** - START HERE
   - Top 10 by Sharpe Ratio
   - Top 10 by CAGR
   - Top 10 by Efficiency (low turnover + high Sharpe)
   - Parameter importance analysis

2. **`all_results.csv`** - Open in Excel/Python
   - All experiments in one table
   - Easy to sort, filter, create charts
   - Columns: experiment_id, cagr, sharpe_ratio, turnover, etf_changes_per_rebalance, etc.

3. **`experiment_XXXX.log`** - Detailed logs for each test
   - Full backtest output
   - Rebalancing events
   - Stop-loss triggers
   - Error details if failed

## Current Test Parameters

```python
GRID_PARAMS = {
    'variant': ['balanced'],
    'rebalance_frequency': ['monthly', 'quarterly'],
    'turnover_penalty': [5.0, 10.0, 20.0, 50.0],
    'concentration_penalty': [0.5, 1.0, 2.0],
    'stop_loss_pct': [None],  # Disabled
    'max_positions': [10, 15, 20],
    'signal_lookback': [63, 126, 252],  # 3m, 6m, 1y
}
```

**Total combinations**: 1 × 2 × 4 × 3 × 1 × 3 × 3 = **72 experiments**
**Estimated time**: ~2-3 hours

## How to Analyze Results

### Method 1: Read the Summary
```bash
cat results/grid_search/LATEST_TIMESTAMP/ANALYSIS_SUMMARY.md
```

This shows:
- Best Sharpe Ratios with their parameters
- Best CAGRs
- Most efficient (high return, low turnover)
- Which parameters matter most

### Method 2: Analyze CSV in Python
```python
import pandas as pd

# Load results
df = pd.read_csv('results/grid_search/LATEST/all_results.csv')

# Find experiments with:
# - Sharpe > 0.5
# - Turnover < 100%
# - ETF changes < 10 per rebalance
good_strategies = df[
    (df['sharpe_ratio'] > 0.5) &
    (df['avg_turnover_pct'] < 100) &
    (df['etf_changes_per_rebalance'] < 10)
]

print(good_strategies.sort_values('sharpe_ratio', ascending=False))
```

### Method 3: Check Individual Logs
```bash
# Find best experiment
grep "Sharpe" results/grid_search/LATEST/ANALYSIS_SUMMARY.md | head -1

# Read its log
cat results/grid_search/LATEST/experiment_0042.log
```

## Monitoring Progress

```bash
# Watch master log
tail -f /tmp/grid_search_master.log

# Check how many done
ls results/grid_search/LATEST/experiment_*.log | wc -l

# See latest results
tail -20 /tmp/grid_search_master.log
```

## Customizing the Search

Edit `scripts/autonomous_grid_search.py` and modify `GRID_PARAMS`:

```python
# Example: Test more turnover penalties
'turnover_penalty': [1.0, 5.0, 10.0, 20.0, 50.0, 100.0],

# Example: Test different rebalance frequencies
'rebalance_frequency': ['weekly', 'monthly', 'quarterly'],

# Example: Add back stop-loss testing
'stop_loss_pct': [None, 0.15, 0.20, 0.25],
```

Then run again - it creates a NEW timestamped directory so previous results aren't lost.

## Expected Results

Based on the analysis, realistic expectations:

**Good Strategy**:
- Sharpe: 0.5 - 1.2
- CAGR: 6% - 12%
- Turnover: 30% - 80%
- ETF Changes: 5-15 per rebalance

**Red Flags**:
- Negative Sharpe = Strategy is broken
- Turnover > 150% = Excessive churn
- Changes > 20 per rebalance = Signal is noise

## What We're Testing

The grid search will reveal:

1. **Is higher turnover penalty better?**
   - Compare 5.0 vs 10.0 vs 20.0 vs 50.0
   - Find sweet spot between flexibility and churn

2. **Does rebalance frequency matter?**
   - Monthly vs Quarterly
   - Which reduces whipsaw?

3. **What's the optimal number of positions?**
   - 10 vs 15 vs 20
   - Does more diversification help?

4. **Does signal lookback period matter?**
   - 3 months vs 6 months vs 1 year
   - Longer = more stable but less responsive

5. **Why is signal "great" but backtest terrible?**
   - Compare in-sample Sharpe (from optimization) vs out-of-sample Sharpe (from backtest)
   - This is tracked in experiment logs
   - Shows the overfitting gap

## Troubleshooting

### "All experiments failing"
Check `grid_search_master.log` for error messages. Common issues:
- Missing data files
- Import errors
- Parameter conflicts

### "Grid search taking forever"
- Default is 72 experiments × 2 min = 2.4 hours
- Reduce parameters or use smaller ETF universe (edit `n_etfs=200` → `100`)

### "Results look wrong"
- Check individual experiment logs
- Verify data quality in price files
- Make sure asset class map loaded correctly

## Next Steps After Grid Search

1. **Review `ANALYSIS_SUMMARY.md`** - identify top performers

2. **Validate best parameters** - run longer backtest (5+ years) with winning params

3. **Walk-forward test** - test on truly out-of-sample data

4. **If still poor results** - signals are fundamentally broken, need new approach
   - Consider simpler strategies (momentum only)
   - Try sector rotation instead of individual ETF picking
   - Use factor ETFs instead of technical signals

---

**Remember**: Grid search finds the best *within current framework*. If all results are bad, the framework itself may be flawed.
