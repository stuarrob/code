# AQR Multi-Factor ETF Strategy - Operations Manual

**Version**: 1.0
**Last Updated**: 2025-10-10
**Status**: Production Ready

---

## Table of Contents

1. [Quick Start Guide](#1-quick-start-guide)
2. [System Overview](#2-system-overview)
3. [Installation and Setup](#3-installation-and-setup)
4. [Daily/Weekly Operations](#4-dailyweekly-operations)
5. [Monthly/Quarterly Maintenance](#5-monthlyquarterly-maintenance)
6. [Monitoring and Alerts](#6-monitoring-and-alerts)
7. [Troubleshooting](#7-troubleshooting)
8. [Parameter Tuning](#8-parameter-tuning)
9. [Backup and Recovery](#9-backup-and-recovery)
10. [Security and Compliance](#10-security-and-compliance)

---

## 1. Quick Start Guide

###  For the Impatient: Generate a Portfolio Right Now

```bash
cd /home/stuar/code/ETFTrader

# Activate virtual environment
source venv/bin/activate

# Generate portfolio with MVO optimizer (recommended)
python scripts/07_run_live_portfolio.py --optimizer mvo --positions 20 --capital 1000000

# Review the output and portfolio recommendations
```

**Output Files**:
- `results/live_portfolio/target_portfolio_latest.csv` - Your recommended portfolio
- `results/live_portfolio/trade_recommendations_latest.csv` - What to buy/sell

### Expected Output

You should see:
- Factor scores calculated for 623 ETFs
- Portfolio of 20 positions generated
- Estimated portfolio statistics (expected return, volatility, Sharpe)
- Trade recommendations (if rebalancing)

**Next Steps**: Review the recommendations, execute trades through your broker, and save your executed portfolio for tracking.

---

## 2. System Overview

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
│  scripts/collect_etf_universe.py → data/raw/prices/*.csv    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   DATA VALIDATION                            │
│  scripts/validate_real_data.py → data/processed/filtered    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                 FACTOR CALCULATION                           │
│  Momentum + Quality + Value + Volatility → Combined Scores  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              PORTFOLIO OPTIMIZATION                          │
│  MVO with Axioma adjustment → Target weights                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              RISK MANAGEMENT                                 │
│  VIX-based stop-loss + Threshold rebalancing                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│               EXECUTION & MONITORING                         │
│  Trade recommendations + Performance tracking                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| Factor Library | Calculate momentum, quality, value, volatility | `src/factors/` |
| Portfolio Optimization | MVO with Axioma adjustment | `src/portfolio/optimizer.py` |
| Risk Management | VIX-based stop-loss, rebalancing | `src/portfolio/risk_manager.py` |
| Backtest Engine | Event-driven validation | `src/backtesting/` |
| Live Portfolio Script | Production portfolio generation | `scripts/07_run_live_portfolio.py` |
| Data Collection | ETF universe and price download | `scripts/collect_etf_universe.py` |

### 2.3 Data Flow

1. **Weekly** (or as needed): Update ETF prices
2. **Weekly** (or as needed): Recalculate factor scores
3. **Weekly**: Check if rebalancing is needed (5% drift threshold)
4. **If rebalancing**: Generate new portfolio, execute trades
5. **Daily**: Monitor positions, check stop-loss distances
6. **Monthly**: Review performance, update validation

---

## 3. Installation and Setup

### 3.1 System Requirements

**Minimum**:
- Python 3.8+
- 4 GB RAM
- 2 GB disk space
- Internet connection (for data download)

**Recommended**:
- Python 3.10+
- 8 GB RAM
- 5 GB disk space
- High-speed internet

### 3.2 Python Environment Setup

```bash
cd /home/stuar/code/ETFTrader

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3.3 Required Python Packages

Core packages (see `requirements.txt`):
```
pandas>=1.5.0
numpy>=1.24.0
yfinance>=0.2.0
cvxpy>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
```

### 3.4 Directory Structure Verification

```bash
# Check that key directories exist
ls -la data/raw/prices        # Price data
ls -la data/processed          # Filtered data
ls -la results/live_portfolio  # Portfolio outputs
ls -la logs                    # Log files
```

If any are missing:
```bash
mkdir -p data/raw/prices data/processed results/live_portfolio logs
```

### 3.5 Initial Data Download

```bash
# Download ETF universe and prices (takes 10-15 minutes)
python scripts/collect_etf_universe.py

# Validate and filter data (takes 1-2 minutes)
python scripts/validate_real_data.py
```

**Expected Output**:
- `data/raw/prices/` contains ~750 CSV files
- `data/processed/etf_prices_filtered.parquet` created
- Console shows ~623 eligible ETFs

### 3.6 VIX Data Download (Optional but Recommended)

For dynamic stop-loss functionality:

```bash
# Option 1: Add ^VIX to ETF universe
# Edit scripts/collect_etf_universe.py and add '^VIX' to tickers list
# Then re-run collection

# Option 2: Manual download
python -c "
import yfinance as yf
import pandas as pd
vix = yf.Ticker('^VIX')
data = vix.history(period='max')
data.to_csv('data/raw/prices/^VIX.csv')
print('VIX data downloaded')
"
```

---

## 4. Daily/Weekly Operations

### 4.1 Weekly Workflow (Recommended)

**Every Monday Morning** (or your chosen day):

#### Step 1: Update Price Data (5-10 minutes)

```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate

# Download latest prices
python scripts/collect_etf_universe.py
```

**Check**: Look for any download errors. If many fail, may be API issue - try again in 30 minutes.

#### Step 2: Validate Data Quality (1-2 minutes)

```bash
# Re-filter for quality
python scripts/validate_real_data.py
```

**Check**: Number of eligible ETFs should be ~620-630. If drops significantly (< 600), investigate.

#### Step 3: Generate Current Portfolio (30 seconds)

```bash
# If you have a current portfolio to compare against:
python scripts/07_run_live_portfolio.py \
    --optimizer mvo \
    --positions 20 \
    --capital 1000000 \
    --current-portfolio results/live_portfolio/executed_portfolio.csv

# If starting fresh (no current portfolio):
python scripts/07_run_live_portfolio.py \
    --optimizer mvo \
    --positions 20 \
    --capital 1000000
```

**Check**: Review the trade recommendations. If no trades recommended, you're done (portfolio within drift threshold).

#### Step 4: Review Recommendations

Open the output files:
```bash
# View target portfolio
cat results/live_portfolio/target_portfolio_latest.csv

# View trade recommendations
cat results/live_portfolio/trade_recommendations_latest.csv
```

**Decision Points**:
- Are the recommended trades reasonable?
- Do any positions have stop-loss alerts?
- Is the turnover excessive (> 30%)?

#### Step 5: Execute Trades (If Needed)

**Manual Execution** (current setup):
1. Log into your broker (Interactive Brokers, TD Ameritrade, etc.)
2. For each SELL order:
   - Enter ticker symbol
   - Enter shares to sell (from recommendations)
   - Execute market order (or limit if preferred)
3. For each BUY order:
   - Enter ticker symbol
   - Enter shares to buy (from recommendations)
   - Execute market order (or limit if preferred)

**Record Execution**:
```bash
# Copy the executed portfolio for next time
cp results/live_portfolio/target_portfolio_latest.csv \
   results/live_portfolio/executed_portfolio_YYYYMMDD.csv
```

#### Step 6: Update Tracking (Optional)

If maintaining a tracking spreadsheet:
1. Record executed trades (date, ticker, shares, price)
2. Update portfolio value
3. Calculate returns since last rebalance

### 4.2 Daily Monitoring (5 minutes)

**Every Trading Day** (optional, for active monitoring):

#### Check Stop-Loss Distances

```bash
# If you have current positions, check risk
python scripts/check_stop_losses.py \
    --portfolio results/live_portfolio/executed_portfolio.csv \
    --current-prices data/raw/prices/*.csv
```

**Action Items**:
- If any position near stop-loss (< 2% away), consider manual intervention
- If VIX spikes > 30, review all positions

#### Market Regime Check

Check current VIX level:
```bash
python -c "
import yfinance as yf
vix = yf.Ticker('^VIX')
latest = vix.history(period='1d')['Close'].iloc[-1]
print(f'Current VIX: {latest:.1f}')
if latest < 15:
    print('Regime: Low Volatility (15% stop-loss)')
elif latest <= 25:
    print('Regime: Normal Volatility (12% stop-loss)')
else:
    print('Regime: High Volatility (10% stop-loss)')
"
```

---

## 5. Monthly/Quarterly Maintenance

### 5.1 Monthly Performance Review

**First Monday of Each Month**:

#### Calculate Returns

```bash
# If you have a tracking file
python scripts/calculate_performance.py \
    --portfolio-file results/tracking/portfolio_history.csv \
    --start-date 2025-09-01 \
    --end-date 2025-09-30
```

**Manual Calculation**:
```
Monthly Return = (End Value - Start Value) / Start Value
```

Compare against:
- S&P 500 (SPY) return
- Target return (17% annual ≈ 1.3% monthly)
- Your expectations

#### Review Top/Bottom Performers

```bash
# List all positions with returns
python -c "
import pandas as pd
portfolio = pd.read_csv('results/live_portfolio/executed_portfolio.csv')
# Add your return calculation logic here
"
```

**Action Items**:
- If a position has loss > 10%, review factor scores to see if it's deteriorating
- If overall portfolio return < 0% for 3 consecutive months, consider parameter adjustment

### 5.2 Quarterly Validation

**Every 3 Months** (Jan, Apr, Jul, Oct):

#### Re-run Validation Backtest

```bash
# Re-run full validation on updated data
python scripts/08_backtest_real_data_3periods.py

# Open the validation notebook
jupyter notebook notebooks/04_real_data_validation_results.ipynb
# Then: Kernel → Restart & Run All
```

**Check**:
- Are recent results consistent with historical validation?
- Has the strategy degraded (lower Sharpe, higher DD)?
- Do parameters need adjustment?

#### Update Period Definitions

If we're now far from 2020:
1. Edit `scripts/08_backtest_real_data_3periods.py`
2. Update period date ranges to recent history
3. Re-run validation
4. Update documentation if results change significantly

### 5.3 Annual Comprehensive Review

**Once Per Year** (e.g., every January):

#### Full System Audit

```bash
# Run all tests
pytest tests/ -v

# Re-validate entire system
python scripts/validate_real_data.py
python scripts/08_backtest_real_data_3periods.py

# Check data quality
python -c "
import pandas as pd
prices = pd.read_parquet('data/processed/etf_prices_filtered.parquet')
missing_pct = prices.isna().sum() / len(prices) * 100
print(f'Missing data: {missing_pct.mean():.2f}% (should be <5%)')
"
```

#### Performance vs Target

Calculate annual metrics:
- CAGR: Target > 12%, Actual?
- Sharpe: Target > 0.8, Actual?
- Max Drawdown: Target < 25%, Actual?
- Turnover: Target < 30%/month, Actual?

#### Documentation Updates

- Update `PROJECT_STATUS.md` with latest results
- Update `OPERATIONS_MANUAL.md` (this file) with learnings
- Refresh LaTeX document if methodology changed
- Archive old validation reports

#### Tax Preparation

```bash
# Export all trades for the year
python scripts/export_trades_for_tax.py --year 2025
```

---

## 6. Monitoring and Alerts

### 6.1 Key Metrics to Monitor

| Metric | Frequency | Warning Threshold | Critical Threshold |
|--------|-----------|-------------------|-------------------|
| VIX | Daily | > 25 | > 35 |
| Portfolio Return | Daily | < -5% (day) | < -10% (day) |
| Position Loss | Daily | < -10% | < -12% (stop-loss) |
| Factor Score Drift | Weekly | > 10% change | > 20% change |
| Data Missing | Weekly | > 5% | > 10% |

### 6.2 Automated Alerts (Future Enhancement)

**To Implement** (when building web app):

```python
# Pseudocode for alert system
if portfolio_loss_today > 5%:
    send_email_alert("Portfolio down 5% today")

if any_position_near_stop_loss(threshold=0.02):  # Within 2%
    send_email_alert("Position near stop-loss")

if vix > 30:
    send_email_alert("VIX elevated - review positions")
```

### 6.3 Manual Monitoring Checklist

**Weekly Checklist**:
- [ ] Price data updated successfully
- [ ] No errors in factor calculation
- [ ] Portfolio generation completed
- [ ] Trade recommendations reviewed
- [ ] All positions > 5% from stop-loss
- [ ] VIX level noted
- [ ] Drift < 5% (no rebalancing needed) OR trades executed

**Monthly Checklist**:
- [ ] Performance calculated
- [ ] Returns vs benchmark compared
- [ ] Top/bottom performers reviewed
- [ ] Portfolio value updated in tracking
- [ ] Any underperforming positions flagged

**Quarterly Checklist**:
- [ ] Full validation backtest run
- [ ] Notebook results reviewed
- [ ] Parameter sensitivity checked
- [ ] Documentation updated

---

## 7. Troubleshooting

### 7.1 Common Issues and Solutions

#### Issue: Data Download Fails

**Symptoms**: Many ETFs fail to download, errors like "No data found"

**Causes**:
- Yahoo Finance API temporarily down
- Rate limiting
- Ticker delisted

**Solutions**:
```bash
# Wait 30 minutes and retry
sleep 1800
python scripts/collect_etf_universe.py

# If specific tickers consistently fail, check if delisted:
python -c "
import yfinance as yf
ticker = yf.Ticker('PROBLEM_TICKER')
print(ticker.info)
"

# Remove delisted tickers from universe
```

#### Issue: Factor Calculation Produces Many NaNs

**Symptoms**: Warnings like "Found NaN values in 50 ETFs"

**Causes**:
- Insufficient price history (< 252 days for momentum)
- All prices identical (divide by zero in volatility)
- Missing data

**Solutions**:
```bash
# Check specific ETF history
python -c "
import pandas as pd
prices = pd.read_parquet('data/processed/etf_prices_filtered.parquet')
etf = 'PROBLEM_TICKER'
print(f'Days of data: {prices[etf].notna().sum()}')
print(f'Missing pct: {prices[etf].isna().mean()*100:.1f}%')
"

# If < 252 days, ETF will be excluded from momentum
# This is expected for recently launched ETFs
# Will be automatically filtered out by factor calculation
```

#### Issue: Optimization Fails

**Symptoms**: Error like "Problem infeasible" or "No solution found"

**Causes**:
- All factor scores negative
- Covariance matrix singular
- Constraints too restrictive

**Solutions**:
```bash
# Check factor score distribution
python -c "
import pandas as pd
# Load latest factor scores
# Check if any are valid
"

# If all negative, may need to adjust factor calculation
# Or relax minimum score constraint
```

#### Issue: Excessive Turnover

**Symptoms**: Recommendations include 15+ trades weekly

**Causes**:
- Drift threshold too low (< 5%)
- Factor scores too volatile
- Using MinVar optimizer

**Solutions**:
```bash
# If using MinVar, increase drift threshold
python scripts/07_run_live_portfolio.py \
    --optimizer minvar \
    --drift-threshold 0.075  # 7.5% instead of 5%

# Or switch to MVO/RankBased which have lower turnover
python scripts/07_run_live_portfolio.py \
    --optimizer mvo  # Recommended
```

#### Issue: Poor Performance (Returns < 0% for 3+ months)

**Symptoms**: Portfolio consistently underperforming

**Possible Causes**:
- Market regime changed significantly
- Factor effectiveness degraded
- Poor execution timing
- Data quality issues

**Diagnostic Steps**:
```bash
# 1. Re-run validation on recent data
python scripts/08_backtest_real_data_3periods.py

# 2. Check if recent period is volatile/unusual
python -c "
import pandas as pd
import yfinance as yf
spy = yf.Ticker('SPY')
recent = spy.history(period='3mo')
returns = recent['Close'].pct_change()
vol = returns.std() * (252**0.5)
print(f'Recent 3-month SPY volatility: {vol:.1%}')
# If > 25%, market is volatile - underperformance may be temporary
"

# 3. Review factor scores for current holdings
python scripts/review_holdings.py --portfolio results/live_portfolio/executed_portfolio.csv
```

**Actions**:
- If market volatile (VIX > 30): Consider reducing position sizes temporarily
- If factor scores deteriorating: May need to rebalance despite being within drift threshold
- If SPY also down significantly: Strategy is correctly tracking market decline

### 7.2 Data Quality Checks

```bash
# Check for stale data
python -c "
import pandas as pd
from datetime import datetime, timedelta
prices = pd.read_parquet('data/processed/etf_prices_filtered.parquet')
latest_date = prices.index[-1]
days_old = (datetime.now() - latest_date).days
print(f'Data age: {days_old} days')
if days_old > 7:
    print('WARNING: Data is stale, run collection script')
"

# Check for missing data
python -c "
import pandas as pd
prices = pd.read_parquet('data/processed/etf_prices_filtered.parquet')
missing_pct = prices.isna().sum() / len(prices) * 100
worst = missing_pct.nlargest(10)
print('ETFs with most missing data:')
print(worst)
if (missing_pct > 10).any():
    print('WARNING: Some ETFs have >10% missing data')
"
```

### 7.3 Getting Help

**Error Messages**:
1. Check log files: `logs/*.log`
2. Search error message in project documentation
3. Check GitHub issues (if applicable)

**Performance Questions**:
1. Review `results/real_data_validation/REAL_DATA_ANALYSIS.md`
2. Compare your results to validation benchmarks
3. Check if market conditions are unusual

**Technical Issues**:
1. Verify Python environment: `pip list`
2. Check Python version: `python --version` (should be 3.8+)
3. Re-run tests: `pytest tests/ -v`

---

## 8. Parameter Tuning

### 8.1 When to Adjust Parameters

**DO adjust if**:
- Consistent underperformance for 6+ months
- Validation shows strategy degradation
- Market conditions permanently changed
- Better parameter values found through research

**DO NOT adjust if**:
- Short-term underperformance (< 3 months)
- Single bad trade
- Emotionally uncomfortable with volatility
- Trying to "fix" a loss

### 8.2 Key Parameters and Their Effects

#### Number of Positions (default: 20)

```bash
# Test different portfolio sizes
python scripts/07_run_live_portfolio.py --optimizer mvo --positions 10  # More concentrated
python scripts/07_run_live_portfolio.py --optimizer mvo --positions 30  # More diversified
```

**Effects**:
- Fewer positions (10-15): Higher returns, higher risk
- More positions (25-30): Lower returns, lower risk
- **Recommendation**: Keep at 20 (validated optimal)

#### Drift Threshold (default: 5% for MVO, 7.5% for MinVar)

```bash
# Lower threshold = more frequent rebalancing
python scripts/07_run_live_portfolio.py --optimizer mvo --drift-threshold 0.03  # 3%

# Higher threshold = less frequent rebalancing
python scripts/07_run_live_portfolio.py --optimizer mvo --drift-threshold 0.07  # 7%
```

**Effects**:
- Lower threshold: More rebalancing, higher costs, potentially better risk control
- Higher threshold: Less rebalancing, lower costs, potentially higher drift
- **Recommendation**: 5% for MVO/RankBased, 7.5% for MinVar

#### Risk Aversion (default: 1.0)

**To change**: Edit `scripts/07_run_live_portfolio.py`, line ~250:
```python
optimizer = MeanVarianceOptimizer(
    risk_aversion=2.0  # Increase from 1.0
)
```

**Effects**:
- Lower (0.5): Higher risk, higher expected returns
- Higher (2.0): Lower risk, lower expected returns
- **Recommendation**: 1.0 (balanced)

#### Factor Weights (default: 25% each)

**To change**: Edit `scripts/07_run_live_portfolio.py`, line ~115:
```python
integrator = FactorIntegrator(weights={
    'momentum': 0.30,
    'quality': 0.30,
    'value': 0.20,
    'volatility': 0.20
})
```

**Effects**: Emphasizes certain factors over others
- **Recommendation**: Equal weighting (25% each) unless specific research suggests otherwise

### 8.3 Testing Parameter Changes

**Before deploying changes to live portfolio**:

```bash
# 1. Run backtest with new parameters
#    Edit scripts/08_backtest_real_data_3periods.py with new values
python scripts/08_backtest_real_data_3periods.py

# 2. Compare results to baseline
#    Check if Sharpe improved, drawdown controlled, etc.

# 3. Run validation notebook
jupyter notebook notebooks/04_real_data_validation_results.ipynb

# 4. If results better: Deploy to live
# 5. If results worse: Revert changes
```

**Documentation**: Always document parameter changes in `PARAMETER_CHANGES_LOG.md`:
```markdown
## 2025-10-15: Increased Risk Aversion to 1.5

**Reason**: Recent volatility (VIX > 30) suggested more defensive posture

**Expected Effect**: Lower returns but also lower drawdowns

**Actual Results**: (Fill in after 1-3 months)
```

---

## 9. Backup and Recovery

### 9.1 What to Backup

**Critical** (backup weekly):
- `data/processed/etf_prices_filtered.parquet` - Your clean ETF data
- `results/live_portfolio/executed_portfolio_*.csv` - Your portfolio history
- `logs/*.log` - System logs

**Important** (backup monthly):
- `data/raw/prices/*.csv` - Raw price data (can be re-downloaded if lost)
- `results/real_data_validation/*.csv` - Validation results

**Configuration** (backup after changes):
- `scripts/07_run_live_portfolio.py` - If you've customized parameters
- `requirements.txt` - If you've added packages

### 9.2 Backup Commands

```bash
# Create backup directory
mkdir -p ~/ETFTrader_backups

# Backup critical data
DATE=$(date +%Y%m%d)
tar -czf ~/ETFTrader_backups/etftrader_backup_$DATE.tar.gz \
    /home/stuar/code/ETFTrader/data/processed \
    /home/stuar/code/ETFTrader/results/live_portfolio \
    /home/stuar/code/ETFTrader/logs

# Check backup size
ls -lh ~/ETFTrader_backups/etftrader_backup_$DATE.tar.gz
```

### 9.3 Cloud Backup (Recommended)

**Option 1: Google Drive**
```bash
# Install rclone
# Configure: rclone config

# Sync backups
rclone sync ~/ETFTrader_backups remote:ETFTrader_backups
```

**Option 2: AWS S3**
```bash
# Install AWS CLI
pip install awscli

# Configure
aws configure

# Upload
aws s3 sync ~/ETFTrader_backups s3://your-bucket/etftrader-backups/
```

### 9.4 Recovery Procedures

#### Recover from Backup

```bash
# Extract backup
cd /home/stuar/code/ETFTrader
tar -xzf ~/ETFTrader_backups/etftrader_backup_YYYYMMDD.tar.gz
```

#### Rebuild from Scratch

If complete data loss:
```bash
# 1. Clone repository (if using git)
git clone <repo-url> /home/stuar/code/ETFTrader

# 2. Setup environment
cd /home/stuar/code/ETFTrader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Download fresh data
python scripts/collect_etf_universe.py
python scripts/validate_real_data.py

# 4. Generate portfolio
python scripts/07_run_live_portfolio.py --optimizer mvo --positions 20 --capital 1000000
```

**Portfolio History Loss**: If you lost portfolio history, you'll need to:
1. Manually reconstruct from broker statements
2. Or start fresh with new baseline

---

## 10. Security and Compliance

### 10.1 Data Security

**Sensitive Data**:
- Portfolio holdings (could reveal strategy if leaked)
- Performance results (proprietary)
- Trade history (personal financial information)

**Protection Measures**:
```bash
# Restrict file permissions
chmod 700 /home/stuar/code/ETFTrader
chmod 600 /home/stuar/code/ETFTrader/results/live_portfolio/*.csv

# Encrypt backups
gpg --symmetric ~/ETFTrader_backups/etftrader_backup_YYYYMMDD.tar.gz
# Use strong passphrase
```

### 10.2 API Security

**Yahoo Finance API**:
- No authentication required
- Rate limiting: ~2000 requests/hour
- No sensitive data transmitted

**Best Practices**:
- Don't share API keys (if you add premium data sources)
- Use HTTPS for all downloads
- Validate data integrity

### 10.3 Trade Execution Security

**When executing trades**:
- Always verify ticker symbols (avoid typos)
- Double-check share quantities
- Use limit orders if possible (avoid slippage on large orders)
- Enable two-factor authentication on broker account
- Log out after executing trades

### 10.4 Compliance Considerations

**Regulatory**:
- This is a personal investment strategy (not investment advice)
- No licensing required for personal use
- If managing money for others: Consult legal/compliance advisors

**Tax Reporting**:
- Keep records of all trades
- Calculate capital gains/losses annually
- Consult tax advisor for specifics

**Record Keeping**:
- Retain trade confirmations: 7 years
- Retain annual tax documents: Permanently
- Retain strategy documentation: Until strategy deprecated

---

## Appendix A: Command Reference

### Quick Command Cheat Sheet

```bash
# Data Operations
python scripts/collect_etf_universe.py          # Download prices
python scripts/validate_real_data.py             # Filter quality

# Portfolio Generation
python scripts/07_run_live_portfolio.py --optimizer mvo --positions 20 --capital 1000000

# Validation & Testing
python scripts/08_backtest_real_data_3periods.py  # Full backtest
pytest tests/ -v                                   # Run all tests
jupyter notebook notebooks/04_real_data_validation_results.ipynb  # View results

# Monitoring
ls -lht results/live_portfolio/                  # Latest portfolios
tail -f logs/portfolio.log                        # Watch logs

# Maintenance
pip list --outdated                               # Check for updates
git status                                        # Check for changes
```

---

## Appendix B: File Locations

| Purpose | Path |
|---------|------|
| Raw price data | `data/raw/prices/*.csv` |
| Filtered data | `data/processed/etf_prices_filtered.parquet` |
| Live portfolios | `results/live_portfolio/` |
| Validation results | `results/real_data_validation/` |
| Logs | `logs/*.log` |
| Notebooks | `notebooks/*.ipynb` |
| Documentation | `docs/*.{md,tex,pdf}` |
| Tests | `tests/` |

---

## Appendix C: Contact and Support

**Project Location**: `/home/stuar/code/ETFTrader`

**Documentation**:
- Technical: `docs/TECHNICAL_INVESTMENT_DOCUMENT.pdf`
- Project Plan: `AQR_MULTIFACTOR_PROJECT_PLAN.md`
- Operations: `OPERATIONS_MANUAL.md` (this file)

**Version History**:
- v1.0 (2025-10-10): Initial production release
- Real data validated: Oct 2020 - Oct 2025
- 623 ETFs, 17.0% CAGR, 1.07 Sharpe

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| **CAGR** | Compound Annual Growth Rate - annualized return |
| **Sharpe Ratio** | Risk-adjusted return metric (return / volatility) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **VIX** | CBOE Volatility Index - market fear gauge |
| **MVO** | Mean-Variance Optimization (Markowitz) |
| **Axioma Adjustment** | Risk penalty for robustness under uncertain returns |
| **Drift Threshold** | Maximum allowed deviation before rebalancing |
| **Factor Score** | Normalized measure of ETF quality on each factor |
| **Geometric Mean** | Multiplicative average, penalizes inconsistency |
| **Turnover** | Percentage of portfolio traded in a period |

---

**End of Operations Manual**

*Last Updated*: 2025-10-10
*Version*: 1.0
*Status*: Production Ready
