# ETFTrader WebApp - Quick Setup Guide

## 🚨 Before Using the WebApp

The webapp needs **factor scores** to generate portfolio recommendations using your AQR strategy.

### Step 1: Generate Factor Scores

```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
python scripts/02_calculate_factors.py
```

**What this does:**
- Calculates momentum, quality, value, and volatility scores for all 623 ETFs
- Creates files in `data/signals/`:
  - `momentum_scores.parquet`
  - `quality_scores.parquet`
  - `value_scores.parquet`
  - `volatility_scores.parquet`

**Time:** ~2-3 minutes

### Step 2: Start the WebApp

Now the webapp will work correctly!

```bash
# Backend (should already be running)
docker-compose up -d

# Frontend
cd frontend
npm start
```

### Step 3: Create a Portfolio

1. Go to **Portfolios** page
2. Click **"Create Portfolio"**
3. Fill in:
   - Name: "My AQR Portfolio"
   - Optimizer: **MVO** (recommended - 17% CAGR in backtests)
   - Positions: 20
   - Capital: $1,000,000

4. Click **Create**

The system will:
- ✅ Calculate composite factor scores using geometric mean
- ✅ Run the MVO optimizer
- ✅ Generate top 20 ETF recommendations
- ✅ Automatically create initial positions
- ✅ Show you the portfolio!

### Step 4: Monitor & Rebalance

**View your portfolio:**
- Click "View" to see positions
- Check "Rebalancing Status"
  - 🟢 Green "OK" = No action needed (drift < 5%)
  - 🟠 Orange "NEEDS REBALANCING" = Time to rebalance (drift > 5%)

**Weekly workflow:**
1. Update price data: `python scripts/collect_etf_universe.py`
2. Regenerate factors: `python scripts/02_calculate_factors.py`
3. Check webapp for rebalancing alerts
4. Execute recommended trades if needed

---

## Troubleshooting

### "No factor recommendations available"

**Problem:** Factor scores haven't been generated
**Solution:** Run `python scripts/02_calculate_factors.py`

### "Close" button doesn't work

This is a known issue - just click outside the modal or press Escape.

### Backend not responding

```bash
# Check if backend is running
docker-compose ps

# Restart if needed
docker-compose restart backend

# View logs
docker-compose logs backend
```

### Factor calculation fails

```bash
# Make sure you have price data
ls data/processed/etf_prices_filtered.parquet

# If missing, collect data first
python scripts/collect_etf_universe.py
```

---

## What the WebApp Does

### Automatic Portfolio Generation
- Uses your validated AQR multi-factor strategy
- MVO optimizer (17.0% CAGR, 1.07 Sharpe in backtests)
- Geometric mean factor integration
- Top 20 ETFs by composite score

### Rebalancing Alerts
- Monitors position drift
- Alerts when drift > 5% threshold
- Shows which positions need adjustment

### Paper Trading
- Simulates realistic costs (slippage, commissions)
- Tracks P&L
- No real money at risk

---

## Next Steps

After you're comfortable with the webapp:

1. **Week 9**: Add real-time charts, advanced analytics
2. **Week 10**: Connect to Interactive Brokers for live data
3. **Future**: Automate the entire process

---

**Ready to use!** Just generate the factor scores and you're all set. 🚀
