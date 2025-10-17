# Automated Portfolio Management System

**Created**: 2025-10-17
**Status**: Production Ready ✅

## Overview

This automated system streamlines the entire weekly portfolio management workflow from the OPERATIONS_MANUAL.md into:
- **One command** to run all steps
- **Excel-based tracking** for positions, trades, and performance
- **Jupyter notebook** for visualization and monitoring

### What It Does

The automation replaces the manual 6-step weekly workflow with a single script that:
1. Updates price data
2. Validates data quality
3. Calculates factor scores
4. Generates portfolio recommendations
5. Checks if rebalancing is needed
6. Updates Excel tracking workbook with all results

---

## Quick Start

### First Time Setup

```bash
# Install required package
source venv/bin/activate
pip install openpyxl

# Run the automation (creates initial portfolio)
python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20
```

### Weekly Workflow

```bash
# Every Monday morning (or your chosen day):
python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20

# Open monitoring notebook to review results
jupyter notebook notebooks/06_portfolio_monitoring.ipynb
```

That's it! The entire workflow is automated.

---

## System Components

### 1. Master Automation Script

**Location**: [`scripts/run_weekly_portfolio.py`](scripts/run_weekly_portfolio.py)

**Usage**:
```bash
python scripts/run_weekly_portfolio.py [OPTIONS]

Options:
  --capital CAPITAL              Portfolio capital (default: 1000000)
  --positions POSITIONS          Number of positions (default: 20)
  --optimizer {mvo,minvar,rankbased}  Optimizer to use (default: mvo)
  --drift-threshold THRESHOLD    Rebalancing threshold (default: 0.05)
  --force-rebalance              Force rebalancing regardless of drift
```

**Examples**:
```bash
# Standard weekly run
python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20

# Conservative portfolio with more positions
python scripts/run_weekly_portfolio.py --capital 500000 --positions 30 --optimizer minvar

# Force rebalancing (even if within drift threshold)
python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20 --force-rebalance

# Rank-based optimizer with higher drift threshold
python scripts/run_weekly_portfolio.py --optimizer rankbased --drift-threshold 0.075
```

**What It Does**:
- ✅ Downloads latest ETF prices (623 ETFs)
- ✅ Validates data quality
- ✅ Calculates 4 factor scores (momentum, quality, value, volatility)
- ✅ Integrates factors into composite scores
- ✅ Optimizes portfolio using selected optimizer
- ✅ Checks if rebalancing needed (drift threshold)
- ✅ Generates trade recommendations
- ✅ Updates Excel tracking workbook
- ✅ Logs everything to timestamped log file

**Output Files**:
- `results/portfolio_tracking.xlsx` - Excel workbook (updated)
- `logs/weekly_automation_YYYYMMDD_HHMMSS.log` - Detailed log
- `data/factor_scores_latest.parquet` - Latest factor scores

### 2. Excel Tracking Workbook

**Location**: [`results/portfolio_tracking.xlsx`](results/portfolio_tracking.xlsx)

The workbook contains **4 sheets** that track everything:

#### Sheet 1: Positions
Complete history of all portfolio positions over time.

| Column | Description |
|--------|-------------|
| timestamp | When this portfolio was generated |
| ticker | ETF ticker symbol |
| weight | Portfolio weight (e.g., 0.05 = 5%) |
| value | Dollar value of position |
| shares | Number of shares to hold |
| price | Current price per share |
| factor_score | Integrated factor score |

#### Sheet 2: Trades
All trade recommendations with execution tracking.

| Column | Description |
|--------|-------------|
| timestamp | When trade was recommended |
| ticker | ETF ticker symbol |
| action | BUY or SELL |
| shares | Number of shares to trade |
| price | Price per share |
| value | Total trade value |
| executed | TRUE/FALSE - mark as TRUE after executing |

**After executing trades**:
1. Open the Excel file
2. Go to "Trades" sheet
3. Find today's trades
4. Mark the "executed" column as TRUE for completed trades
5. Save the file

#### Sheet 3: Performance
Portfolio performance metrics over time.

| Column | Description |
|--------|-------------|
| timestamp | Date of portfolio |
| total_value | Total portfolio value |
| num_positions | Number of positions |
| expected_return | Expected annual return (from factor scores) |
| expected_volatility | Expected annual volatility |
| expected_sharpe | Expected Sharpe ratio |

#### Sheet 4: Metadata
Run history and configuration.

| Column | Description |
|--------|-------------|
| timestamp | When script was run |
| rebalanced | TRUE if rebalancing occurred |
| optimizer | Optimizer used (mvo/minvar/rankbased) |
| drift_threshold | Drift threshold setting |
| capital | Portfolio capital |
| num_positions | Number of positions |

### 3. Monitoring Notebook

**Location**: [`notebooks/06_portfolio_monitoring.ipynb`](notebooks/06_portfolio_monitoring.ipynb)

**Purpose**: Comprehensive visualization and analysis of your portfolio.

**To Use**:
```bash
# Start Jupyter
jupyter notebook notebooks/06_portfolio_monitoring.ipynb

# Or use the start script
./start_jupyter.sh
```

**Features**:

#### Current Portfolio Analysis
- Portfolio allocation pie chart (top 10 + other)
- Position weights bar chart
- Factor score distribution
- Concentration metrics (HHI index)
- Top/bottom positions by weight

#### Performance Analysis
- Cumulative returns chart (vs SPY benchmark)
- Period returns bar chart
- Drawdown visualization
- Performance metrics (total return, Sharpe, max DD)
- Expected vs realized performance comparison

#### Trade History
- Trade activity by rebalance date
- Trade size distribution
- Most frequently traded ETFs
- Buy/sell breakdown

#### Risk Metrics
- Current VIX level and market regime
- Recommended stop-loss levels
- Concentration risk analysis
- Top N position exposure

#### Execution Tracking
- Pending trades to execute
- Executed trades history
- Total pending value

#### Export Summary Report
- Text summary of portfolio status
- Saved to `results/portfolio_summary_*.txt`

---

## Complete Weekly Workflow

### Monday Morning Routine (5-10 minutes)

#### Step 1: Run Automation (2 minutes)
```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20
```

**Expected Output**:
```
================================================================================
WEEKLY AUTOMATION COMPLETED SUCCESSFULLY
================================================================================
Rebalancing needed: True/False
Trades to execute: X
  BUY orders: Y
  SELL orders: Z
Tracking workbook: results/portfolio_tracking.xlsx
Log file: logs/weekly_automation_20251017_182248.log
```

#### Step 2: Review Results in Notebook (2 minutes)
```bash
jupyter notebook notebooks/06_portfolio_monitoring.ipynb
# Run all cells: Kernel → Restart & Run All
```

**Review**:
- Current portfolio allocation
- Performance vs benchmark
- Any pending trades
- Risk metrics and VIX level

#### Step 3: Execute Trades (If Needed) (3-5 minutes)

If rebalancing is needed (script says "Rebalancing needed: True"):

1. **Open Excel file**: `results/portfolio_tracking.xlsx`
2. **Go to "Trades" sheet**
3. **Filter for today's date** (latest timestamp)
4. **Execute trades in your broker**:
   - For each SELL: Sell the specified shares
   - For each BUY: Buy the specified shares
5. **Mark trades as executed**:
   - Update "executed" column to TRUE
   - Save the Excel file

#### Step 4: Done! ✅

Your portfolio is now updated and tracked.

---

## Understanding Rebalancing

The system automatically checks if rebalancing is needed by comparing:
- **Current portfolio** (from Excel Positions sheet, latest entry)
- **Target portfolio** (newly optimized based on latest data)

### Drift Threshold

**Default**: 5% for MVO, 7.5% for MinVar

If the total drift (sum of absolute weight differences) exceeds the threshold, rebalancing is triggered.

**Example**:
- Current: ABC=5%, XYZ=5%, DEF=5%
- Target: ABC=7%, XYZ=3%, DEF=6%
- Drift: |7-5| + |3-5| + |6-5| = 4% → No rebalancing

**Adjust threshold**:
```bash
# More frequent rebalancing (3% threshold)
python scripts/run_weekly_portfolio.py --drift-threshold 0.03

# Less frequent rebalancing (10% threshold)
python scripts/run_weekly_portfolio.py --drift-threshold 0.10
```

### Force Rebalancing

To rebalance regardless of drift:
```bash
python scripts/run_weekly_portfolio.py --force-rebalance
```

Use this when:
- Major market event occurred
- You want to update factor exposures
- Quarterly rebalancing schedule

---

## Optimizer Comparison

### MVO (Mean-Variance Optimizer) ✅ RECOMMENDED
```bash
python scripts/run_weekly_portfolio.py --optimizer mvo
```

**Characteristics**:
- 17.0% CAGR, 1.07 Sharpe (validated)
- Balanced risk/return
- Moderate turnover
- **Best for**: Most users

### MinVar (Minimum Variance)
```bash
python scripts/run_weekly_portfolio.py --optimizer minvar --drift-threshold 0.075
```

**Characteristics**:
- 9.3% CAGR, 0.80 Sharpe (validated)
- Lower volatility
- Higher turnover (use 7.5% drift threshold)
- **Best for**: Conservative investors

### RankBased
```bash
python scripts/run_weekly_portfolio.py --optimizer rankbased
```

**Characteristics**:
- 17.1% CAGR, 1.06 Sharpe (validated)
- Similar to MVO
- No covariance estimation
- **Best for**: Alternative to MVO

---

## File Locations

### Input Files
| File | Description |
|------|-------------|
| `data/processed/etf_prices_filtered.parquet` | Filtered ETF prices (623 ETFs) |
| `data/raw/prices/*.csv` | Raw price data |

### Output Files
| File | Description |
|------|-------------|
| `results/portfolio_tracking.xlsx` | **Main tracking workbook** |
| `results/portfolio_summary_*.txt` | Text summary reports |
| `data/factor_scores_latest.parquet` | Latest factor scores |
| `logs/weekly_automation_*.log` | Detailed logs |

### Notebooks
| Notebook | Description |
|----------|-------------|
| `notebooks/06_portfolio_monitoring.ipynb` | **Monitoring dashboard** |
| `notebooks/04_real_data_validation_results.ipynb` | Validation results |

---

## Troubleshooting

### Issue: "No current portfolio found"

**Cause**: First run, no Excel file exists yet.

**Solution**: This is normal! The script creates initial portfolio. Review and execute the recommended trades.

### Issue: "Data collection failed"

**Cause**: Yahoo Finance API issue or network problem.

**Solution**:
```bash
# Wait 30 minutes and retry
python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20
```

### Issue: "Tracking workbook corrupted"

**Cause**: Excel file was modified incorrectly.

**Solution**:
```bash
# Backup current file
cp results/portfolio_tracking.xlsx results/portfolio_tracking_backup.xlsx

# The next run will append to it, or delete it to start fresh
rm results/portfolio_tracking.xlsx
python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20
```

### Issue: High expected volatility or unrealistic Sharpe

**Cause**: Limited data window or market volatility.

**Solution**: This is expected with only recent data. The metrics are estimates. Focus on realized performance in the notebook.

### Issue: "Optimization failed"

**Cause**: All factor scores negative or covariance matrix issues.

**Solution**:
```bash
# Try rank-based optimizer (doesn't use covariance)
python scripts/run_weekly_portfolio.py --optimizer rankbased

# Or check data quality
python scripts/validate_real_data.py
```

---

## Advanced Usage

### Custom Parameters

Edit [`scripts/run_weekly_portfolio.py`](scripts/run_weekly_portfolio.py) to customize:

**Factor Weights** (line 91-96):
```python
self.integrator = FactorIntegrator(factor_weights={
    'momentum': 0.30,      # Increase momentum
    'quality': 0.30,       # Increase quality
    'value': 0.20,         # Decrease value
    'volatility': 0.20     # Decrease volatility
})
```

**Risk Aversion** (line 107):
```python
self.optimizer = MeanVarianceOptimizer(
    risk_aversion=1.5  # Higher = more conservative
)
```

**Lookback Periods** (line 88-92):
```python
self.factor_calculators = {
    'momentum': MomentumFactor(lookback=252, skip_recent=21),  # 1 year
    'quality': QualityFactor(lookback=252),
    'value': SimplifiedValueFactor(),
    'volatility': VolatilityFactor(lookback=90)  # Change to 90 days
}
```

### Scheduling (Linux/Mac)

**Option 1: Cron Job (Weekly)**
```bash
# Edit crontab
crontab -e

# Add line (runs every Monday at 9:30 AM)
30 9 * * 1 cd /home/stuar/code/ETFTrader && source venv/bin/activate && python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20
```

**Option 2: Systemd Timer (More Control)**
```bash
# Create timer unit
sudo nano /etc/systemd/system/etf-portfolio.timer

# Add configuration for weekly execution
```

---

## Performance Tracking

### Key Metrics to Monitor (in Notebook)

1. **Cumulative Return**: Track vs SPY benchmark
2. **Sharpe Ratio**: Risk-adjusted returns (target > 1.0)
3. **Max Drawdown**: Worst peak-to-trough decline (target < 25%)
4. **Win Rate**: Percentage of positive periods
5. **Turnover**: How often positions change

### When to Adjust

**Consider adjusting if**:
- Underperforming SPY by >5% for 6+ months
- Sharpe ratio < 0.5 for 3+ months
- Max drawdown > 30%
- Excessive turnover (>50% monthly)

**Adjustment Options**:
1. Change optimizer
2. Adjust factor weights
3. Increase/decrease number of positions
4. Modify drift threshold
5. Change rebalancing frequency

---

## Comparison: Old vs New Workflow

### Old Manual Workflow (OPERATIONS_MANUAL.md)
- ❌ 6 separate steps to run
- ❌ Manual tracking in separate files
- ❌ No integrated visualization
- ❌ Easy to miss steps
- ❌ ~30 minutes per week

### New Automated Workflow
- ✅ Single command
- ✅ Automatic Excel tracking
- ✅ Integrated visualization
- ✅ Complete audit trail
- ✅ ~5 minutes per week

---

## Summary

### What You Get

1. **Automation**: One command runs entire workflow
2. **Tracking**: Excel workbook maintains complete history
3. **Visualization**: Jupyter notebook for analysis
4. **Flexibility**: Multiple optimizers and parameters
5. **Reliability**: Comprehensive logging and error handling

### Files Created

- `scripts/run_weekly_portfolio.py` - Master automation script
- `results/portfolio_tracking.xlsx` - Excel tracking workbook
- `notebooks/06_portfolio_monitoring.ipynb` - Monitoring dashboard

### Weekly Workflow

1. Run: `python scripts/run_weekly_portfolio.py --capital 1000000 --positions 20`
2. Review: Open `notebooks/06_portfolio_monitoring.ipynb`
3. Execute: Trade recommended positions (if rebalancing)
4. Track: Mark trades as executed in Excel

### Next Steps

1. **Initial Setup**: Run automation script to create first portfolio
2. **Execute Trades**: Implement initial positions in your broker
3. **Weekly Routine**: Run script every Monday, review, execute trades
4. **Monthly Review**: Analyze performance trends in notebook
5. **Quarterly Validation**: Re-run validation backtest

---

## Support

**Documentation**:
- This file: Automated workflow
- [`OPERATIONS_MANUAL.md`](OPERATIONS_MANUAL.md): Detailed manual operations
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md): Overall project status

**Log Files**:
- Check `logs/weekly_automation_*.log` for detailed execution logs
- Review notebook output for any warnings or errors

**Questions**:
- Review OPERATIONS_MANUAL.md for in-depth explanations
- Check validation results in `notebooks/04_real_data_validation_results.ipynb`
- Examine factor calculations in `src/factors/`

---

**Version**: 1.0
**Last Updated**: 2025-10-17
**Status**: Production Ready ✅
