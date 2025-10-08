#!/bin/bash
# Iterate through momentum parameter combinations
# Run overnight to find what works

cd /home/stuar/code/ETFTrader
source venv/bin/activate

echo "================================"
echo "MOMENTUM PARAMETER ITERATION"
echo "================================"
echo ""
echo "Testing multiple parameter combinations..."
echo "Results will be saved to: results/momentum_backtests/"
echo ""

# Test 1: Baseline (126-day momentum, turnover penalty 50)
echo "[1/9] Testing baseline: 126-day momentum, turnover 50..."
python scripts/backtest_weekly_momentum.py

# Test 2: Higher turnover penalty
echo "[2/9] Testing higher turnover penalty: 100..."
python scripts/backtest_weekly_momentum.py --turnover-penalty 100.0

# Test 3: Lower turnover penalty
echo "[3/9] Testing lower turnover penalty: 20..."
python scripts/backtest_weekly_momentum.py --turnover-penalty 20.0

# Test 4: Longer momentum period (1 year)
echo "[4/9] Testing 1-year momentum..."
python scripts/backtest_weekly_momentum.py --momentum-period 252

# Test 5: Shorter momentum period (3 months)
echo "[5/9] Testing 3-month momentum..."
python scripts/backtest_weekly_momentum.py --momentum-period 63

# Test 6: No relative strength
echo "[6/9] Testing without relative strength..."
python scripts/backtest_weekly_momentum.py --no-rel-strength

# Test 7: No trend filter
echo "[7/9] Testing without trend filter..."
python scripts/backtest_weekly_momentum.py --no-trend-filter

# Test 8: Combination: 1-year momentum + high turnover penalty
echo "[8/9] Testing 1-year momentum + high turnover penalty..."
python scripts/backtest_weekly_momentum.py --momentum-period 252 --turnover-penalty 100.0

# Test 9: Combination: 3-month momentum + low turnover penalty
echo "[9/9] Testing 3-month momentum + low turnover penalty..."
python scripts/backtest_weekly_momentum.py --momentum-period 63 --turnover-penalty 20.0

echo ""
echo "================================"
echo "ALL TESTS COMPLETE"
echo "================================"
echo ""
echo "Analyzing results..."
python scripts/compare_momentum_tests.py

echo ""
echo "Done! Review the comparison above to see which parameters work best."
echo ""
