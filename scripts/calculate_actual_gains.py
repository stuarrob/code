#!/usr/bin/env python3
"""Calculate actual investment gains vs capital contributions."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path.home() / "trading"

# Load the results
xl = pd.ExcelFile(RESULTS_DIR / "trailing_stop_comparison_latest.xlsx")
summary = pd.read_excel(xl, sheet_name="Summary")

# Capital breakdown
initial_capital = 100000
monthly_addition = 5000
months = 11  # Jan to Nov 2025
total_contributions = initial_capital + (monthly_addition * months)

print("=" * 80)
print("INVESTMENT GAIN ANALYSIS")
print("=" * 80)
print()
print("Capital Breakdown:")
print(f"  Initial Capital:      ${initial_capital:,.0f}")
print(f"  Monthly Additions:    ${monthly_addition * months:,.0f} ({months} months x ${monthly_addition:,})")
print(f"  Total Contributed:    ${total_contributions:,.0f}")
print()
print("Strategy Performance (Actual Investment Gains):")
print("-" * 80)

results = []
for _, row in summary.iterrows():
    final_value = row["Final Value"]
    gain_dollars = final_value - total_contributions
    gain_pct = (gain_dollars / total_contributions) * 100

    results.append({
        "Strategy": row["Strategy"],
        "Final Value": final_value,
        "Contributed": total_contributions,
        "Actual Gain ($)": gain_dollars,
        "Actual Gain (%)": gain_pct,
        "Sharpe": row["Sharpe"],
        "Max Drawdown": row["Max Drawdown"],
    })

    name = row["Strategy"][:35]
    print(f"{name:<35}")
    print(f"    Final: ${final_value:,.0f} | Contributed: ${total_contributions:,.0f} | GAIN: ${gain_dollars:,.0f} ({gain_pct:.1f}%)")
    print()

print("-" * 80)

# Create summary dataframe
results_df = pd.DataFrame(results)

# Best performers
best_gain_idx = results_df["Actual Gain ($)"].idxmax()
best_sharpe_idx = results_df["Sharpe"].idxmax()
best_dd_idx = results_df["Max Drawdown"].idxmax()

print()
print("WINNERS:")
print(f"  Best Actual Gain: {results_df.loc[best_gain_idx, 'Strategy']}")
print(f"      -> ${results_df.loc[best_gain_idx, 'Actual Gain ($)']:,.0f} ({results_df.loc[best_gain_idx, 'Actual Gain (%)']:.1f}%)")
print()
print(f"  Best Risk-Adjusted (Sharpe): {results_df.loc[best_sharpe_idx, 'Strategy']}")
print(f"      -> Sharpe: {results_df.loc[best_sharpe_idx, 'Sharpe']:.2f}")
print()
print(f"  Best Drawdown Control: {results_df.loc[best_dd_idx, 'Strategy']}")
print(f"      -> Max DD: {results_df.loc[best_dd_idx, 'Max Drawdown']:.2%}")

# Save updated results
output_file = RESULTS_DIR / "actual_gains_analysis.xlsx"
results_df.to_excel(output_file, index=False)
print()
print(f"Saved: {output_file}")
