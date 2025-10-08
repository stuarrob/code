"""
Analyze Signal Exploration Results

Post-hoc analysis of grid search to identify:
1. Which indicator families work
2. Which combinations are stable (low signal churn)
3. Which weighting strategies are best
4. Correlation between signal stability and backtest performance

Run AFTER signal_exploration_grid_search.py completes.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Find latest results directory
results_base = project_root / "results" / "signal_exploration"
latest_dir = max(results_base.glob("*"), key=lambda p: p.stat().st_mtime)

print(f"\nAnalyzing results from: {latest_dir}\n")

# Load all results
csv_file = latest_dir / "all_results.csv"
if not csv_file.exists():
    print(f"ERROR: No results file found at {csv_file}")
    sys.exit(1)

df = pd.read_csv(csv_file)

print("="*100)
print("SIGNAL EXPLORATION ANALYSIS")
print("="*100)

print(f"\nTotal experiments: {len(df)}")
print(f"Date range: {latest_dir.name}")

# ============================================================================
# 1. BEST OVERALL PERFORMERS
# ============================================================================

print("\n" + "="*100)
print("TOP 10 BEST PERFORMERS (by Sharpe Ratio)")
print("="*100 + "\n")

top_10 = df.nlargest(10, 'sharpe')
for idx, row in top_10.iterrows():
    print(f"#{idx+1} - Sharpe: {row['sharpe']:.2f}")
    print(f"   Combo: {row['combo_name']}")
    print(f"   Weighting: {row['weighting']}, Method: {row['method']}")
    print(f"   CAGR: {row['cagr']*100:.1f}%, Turnover: {row['avg_turnover_pct']:.0f}%")
    print(f"   Signal Stability: {row['stability']:.2f}")
    print(f"   Turnover Penalty: {row['turnover_penalty']}, Rebalance: {row['rebalance']}")
    print()

# ============================================================================
# 2. SIGNAL STABILITY VS PERFORMANCE
# ============================================================================

print("="*100)
print("SIGNAL STABILITY ANALYSIS")
print("="*100 + "\n")

print("Key Insight: High signal stability → Low turnover → Better performance\n")

# Correlation
corr_stability_sharpe = df['stability'].corr(df['sharpe'])
corr_stability_turnover = df['stability'].corr(df['avg_turnover_pct'])
corr_turnover_sharpe = df['avg_turnover_pct'].corr(df['sharpe'])

print(f"Signal Stability ←→ Sharpe Ratio:  {corr_stability_sharpe:>7.3f}")
print(f"Signal Stability ←→ Turnover:      {corr_stability_turnover:>7.3f}")
print(f"Turnover ←→ Sharpe Ratio:          {corr_turnover_sharpe:>7.3f}")

# Best stable signals
print("\nMost Stable Signals (stability > 0.7):")
stable = df[df['stability'] > 0.7].sort_values('sharpe', ascending=False)
if len(stable) > 0:
    for idx, row in stable.head(5).iterrows():
        print(f"  {row['combo_name']:30s} Stability={row['stability']:.2f}, Sharpe={row['sharpe']:.2f}, Turnover={row['avg_turnover_pct']:.0f}%")
else:
    print("  No signals with stability > 0.7")

# ============================================================================
# 3. INDICATOR FAMILY PERFORMANCE
# ============================================================================

print("\n" + "="*100)
print("INDICATOR FAMILY PERFORMANCE")
print("="*100 + "\n")

# Group by combo name
by_combo = df.groupby('combo_name').agg({
    'sharpe': ['mean', 'std', 'max', 'count'],
    'cagr': 'mean',
    'avg_turnover_pct': 'mean',
    'stability': 'mean'
}).round(3)

by_combo.columns = ['_'.join(col).strip() for col in by_combo.columns.values]
by_combo = by_combo.sort_values('sharpe_mean', ascending=False)

print("Average Performance by Signal Combination:\n")
print(by_combo.to_string())

print("\n\nBest Indicator Families:")
print(f"  1. {by_combo.index[0]:30s}  Avg Sharpe={by_combo.iloc[0]['sharpe_mean']:.2f}")
print(f"  2. {by_combo.index[1]:30s}  Avg Sharpe={by_combo.iloc[1]['sharpe_mean']:.2f}")
print(f"  3. {by_combo.index[2]:30s}  Avg Sharpe={by_combo.iloc[2]['sharpe_mean']:.2f}")

# ============================================================================
# 4. WEIGHTING STRATEGY IMPACT
# ============================================================================

print("\n" + "="*100)
print("WEIGHTING STRATEGY IMPACT")
print("="*100 + "\n")

by_weighting = df.groupby('weighting').agg({
    'sharpe': ['mean', 'std', 'max'],
    'stability': 'mean',
    'avg_turnover_pct': 'mean'
}).round(3)

print(by_weighting.to_string())

print(f"\n\nBest Weighting Strategy: {by_weighting['sharpe']['mean'].idxmax()}")
print(f"  Avg Sharpe: {by_weighting['sharpe']['mean'].max():.2f}")

# ============================================================================
# 5. COMBINATION METHOD IMPACT
# ============================================================================

print("\n" + "="*100)
print("COMBINATION METHOD IMPACT")
print("="*100 + "\n")

by_method = df.groupby('method').agg({
    'sharpe': ['mean', 'std', 'max'],
    'stability': 'mean'
}).round(3)

print(by_method.to_string())

print(f"\n\nBest Combination Method: {by_method['sharpe']['mean'].idxmax()}")

# ============================================================================
# 6. TURNOVER PENALTY EFFECTIVENESS
# ============================================================================

print("\n" + "="*100)
print("TURNOVER PENALTY EFFECTIVENESS")
print("="*100 + "\n")

by_turnover_penalty = df.groupby('turnover_penalty').agg({
    'avg_turnover_pct': 'mean',
    'sharpe': 'mean'
}).round(3)

print(by_turnover_penalty.to_string())

print("\n\nDid higher penalty reduce turnover?")
for penalty in sorted(df['turnover_penalty'].unique()):
    avg_turn = by_turnover_penalty.loc[penalty, 'avg_turnover_pct']
    print(f"  Penalty {penalty:>5.0f}: Avg Turnover = {avg_turn:.0f}%")

# ============================================================================
# 7. REBALANCE FREQUENCY IMPACT
# ============================================================================

print("\n" + "="*100)
print("REBALANCE FREQUENCY IMPACT")
print("="*100 + "\n")

by_rebalance = df.groupby('rebalance').agg({
    'sharpe': 'mean',
    'avg_turnover_pct': 'mean',
    'cagr': 'mean'
}).round(3)

print(by_rebalance.to_string())

# ============================================================================
# 8. RECOMMENDATIONS
# ============================================================================

print("\n" + "="*100)
print("RECOMMENDATIONS")
print("="*100 + "\n")

# Find best overall
best = df.nlargest(1, 'sharpe').iloc[0]

print(f"🏆 BEST OVERALL CONFIGURATION:\n")
print(f"   Signal Combo:        {best['combo_name']}")
print(f"   Weighting:           {best['weighting']}")
print(f"   Combination Method:  {best['method']}")
print(f"   Turnover Penalty:    {best['turnover_penalty']}")
print(f"   Rebalance Frequency: {best['rebalance']}")
print(f"\n   Performance:")
print(f"     Sharpe Ratio:      {best['sharpe']:.2f}")
print(f"     CAGR:              {best['cagr']*100:.1f}%")
print(f"     Max Drawdown:      {best['max_drawdown']*100:.1f}%")
print(f"     Turnover:          {best['avg_turnover_pct']:.0f}%")
print(f"     Signal Stability:  {best['stability']:.2f}")

# Find best low-turnover
low_turnover = df[df['avg_turnover_pct'] < 80].nlargest(1, 'sharpe')
if len(low_turnover) > 0:
    best_low = low_turnover.iloc[0]
    print(f"\n\n⚡ BEST LOW-TURNOVER CONFIGURATION (<80%):\n")
    print(f"   Signal Combo:        {best_low['combo_name']}")
    print(f"   Weighting:           {best_low['weighting']}")
    print(f"   Combination Method:  {best_low['method']}")
    print(f"   Turnover Penalty:    {best_low['turnover_penalty']}")
    print(f"\n   Performance:")
    print(f"     Sharpe Ratio:      {best_low['sharpe']:.2f}")
    print(f"     CAGR:              {best_low['cagr']*100:.1f}%")
    print(f"     Turnover:          {best_low['avg_turnover_pct']:.0f}%")
    print(f"     Signal Stability:  {best_low['stability']:.2f}")

# General insights
print("\n\n📊 KEY INSIGHTS:\n")

if corr_stability_sharpe > 0.3:
    print("   ✅ Signal stability MATTERS - More stable signals → Better Sharpe")
else:
    print("   ⚠️  Signal stability doesn't correlate with performance")

if corr_stability_turnover < -0.3:
    print("   ✅ Stable signals → Lower turnover (as expected)")
else:
    print("   ⚠️  Signal stability doesn't reduce turnover much")

if df['sharpe'].max() > 0.5:
    print(f"   ✅ Found profitable signals (Max Sharpe: {df['sharpe'].max():.2f})")
elif df['sharpe'].max() > 0:
    print(f"   ⚠️  Marginally profitable (Max Sharpe: {df['sharpe'].max():.2f})")
else:
    print(f"   ❌ NO profitable signals found (Max Sharpe: {df['sharpe'].max():.2f})")

if df['avg_turnover_pct'].min() < 80:
    print(f"   ✅ Low turnover achievable ({df['avg_turnover_pct'].min():.0f}% minimum)")
else:
    print(f"   ❌ All strategies have high turnover (>{df['avg_turnover_pct'].min():.0f}%)")

print("\n" + "="*100)
print("NEXT STEPS")
print("="*100 + "\n")

if df['sharpe'].max() < 0:
    print("❌ NO PROFITABLE SIGNALS FOUND\n")
    print("The technical indicators don't work in this market period.")
    print("\nOptions:")
    print("  1. Test on different time period (2017-2020 instead of 2022-2024)")
    print("  2. Use simpler buy-and-hold with tactical cash allocation")
    print("  3. Try alternative data sources (fundamentals, sentiment)")

elif df['sharpe'].max() < 0.5:
    print("⚠️  WEAK PERFORMANCE\n")
    print("Signals are marginally profitable but not great.")
    print("\nOptions:")
    print("  1. Fine-tune best configuration with more parameter variations")
    print("  2. Add market regime filter (only trade in favorable conditions)")
    print("  3. Combine with other alpha sources")

else:
    print("✅ PROFITABLE SIGNALS FOUND\n")
    print("Good signals identified. Next steps:")
    print("  1. Validate best config on out-of-sample period")
    print("  2. Walk-forward test to verify robustness")
    print("  3. Implement in production with monitoring")
    print("  4. Fine-tune around winning parameters")

print(f"\n\nDetailed results: {latest_dir}")
print("="*100 + "\n")
