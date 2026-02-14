#!/usr/bin/env python3
"""
Visualize actual investment gains (net of cash injections) for all strategies.
Shows the TRUE return on invested capital, not inflated by total return metrics.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

RESULTS_DIR = Path.home() / "trading"

def calculate_actual_gains():
    """Read latest comparison and calculate actual gains from portfolio history sheets."""

    excel_file = RESULTS_DIR / "trailing_stop_comparison_latest.xlsx"

    if not excel_file.exists():
        print(f"Error: {excel_file} not found")
        print("Please run backtest_trailing_stop_comparison.py first")
        return None

    # Read summary for other metrics
    summary_df = pd.read_excel(excel_file, sheet_name="Summary")
    xl = pd.ExcelFile(excel_file)

    print("="*80)
    print("ACTUAL INVESTMENT GAINS ANALYSIS")
    print("="*80)

    # Calculate actual gains from each strategy's portfolio history
    results = []

    # Match sheet names to summary row indices
    sheet_to_summary_idx = {
        "S1_Entry Stop 12pct (Current)": 0,
        "S2_Trail 8pct (from peak)": 1,
        "S3_Trail 10pct (from peak)": 2,
        "S4_Trail 12pct (from peak)": 3,
        "S5_Trail 15pct (from peak)": 4,
        "S6_Hybrid Entry 12pct + Trail 8": 5,
        "S7_Hybrid Entry 12pct + Trail 1": 6,
    }

    for sheet in xl.sheet_names[1:]:  # Skip Summary
        df = pd.read_excel(excel_file, sheet_name=sheet)

        if "total_value" not in df.columns or "contributions" not in df.columns:
            continue

        final_row = df.iloc[-1]
        total_value = final_row["total_value"]
        contributions = final_row["contributions"]
        actual_gain = total_value - contributions
        actual_gain_pct = (actual_gain / contributions) * 100

        # Get summary row by index
        summary_idx = sheet_to_summary_idx.get(sheet, -1)
        if summary_idx >= 0 and summary_idx < len(summary_df):
            summary_row = summary_df.iloc[summary_idx]
            strategy = summary_row["Strategy"]
            max_dd = summary_row["Max Drawdown"]
            sharpe = summary_row["Sharpe"]
            stops = summary_row["Stop-Losses"]
        else:
            # Fallback - get strategy name from sheet name
            strategy = sheet.replace("S1_", "").replace("S2_", "").replace("S3_", "")
            strategy = strategy.replace("S4_", "").replace("S5_", "").replace("S6_", "").replace("S7_", "")
            strategy = strategy.replace("pct", "%").replace("_", " ")
            max_dd = 0
            sharpe = 0
            stops = 0

        results.append({
            "strategy": strategy,
            "final_value": total_value,
            "contributions": contributions,
            "actual_gain": actual_gain,
            "actual_gain_pct": actual_gain_pct,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "stops_triggered": stops
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("actual_gain", ascending=False)

    print(f"\nCapital Structure:")
    print(f"  Total contributions: ${results_df.iloc[0]['contributions']:,.0f}")
    print(f"    - $100,000 initial capital")
    print(f"    - $55,000 monthly additions ($10,000 x 5.5 months)")
    print(f"  Period: Jan 2025 - Dec 2025 (~11 months)")

    print("\n" + "="*80)
    print("RESULTS BY ACTUAL GAIN")
    print("="*80)

    for _, row in results_df.iterrows():
        print(f"\n{row['strategy']}:")
        print(f"  Final Value:     ${row['final_value']:>12,.2f}")
        print(f"  Contributions:   ${row['contributions']:>12,.0f}")
        print(f"  ACTUAL GAIN:     ${row['actual_gain']:>12,.2f} ({row['actual_gain_pct']:+.2f}%)")
        print(f"  Max Drawdown:    {row['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio:    {row['sharpe']:.2f}")
        print(f"  Stops Triggered: {row['stops_triggered']:.0f}")

    return results_df


def create_visualization(results_df):
    """Create visualization of actual gains."""

    if results_df is None:
        return

    # Sort by actual gain for display
    results_df = results_df.sort_values("actual_gain_pct", ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trailing Stop Strategy Comparison - ACTUAL Investment Gains\n(Total Contributions: $155,000 | 2025 YTD)",
                 fontsize=14, fontweight="bold")

    # Color scheme - green for positive, red for negative
    colors = ["#2E7D32" if x > 0 else "#C62828" for x in results_df["actual_gain_pct"]]

    # 1. Actual Gain (%) - Horizontal bar chart
    ax1 = axes[0, 0]
    bars = ax1.barh(results_df["strategy"], results_df["actual_gain_pct"], color=colors, edgecolor="black", alpha=0.8)
    ax1.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax1.set_xlabel("Actual Return (%)")
    ax1.set_title("Actual Investment Return\n(Net of $155,000 Contributions)")

    # Add value labels
    for bar, val in zip(bars, results_df["actual_gain_pct"]):
        x_pos = bar.get_width()
        offset = 0.2 if val >= 0 else -0.2
        ha = "left" if val >= 0 else "right"
        ax1.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}%", va="center", ha=ha, fontsize=10, fontweight="bold")

    # 2. Actual Gain ($) - Horizontal bar chart
    ax2 = axes[0, 1]
    colors_dollar = ["#2E7D32" if x > 0 else "#C62828" for x in results_df["actual_gain"]]
    bars2 = ax2.barh(results_df["strategy"], results_df["actual_gain"], color=colors_dollar, edgecolor="black", alpha=0.8)
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Actual Gain ($)")
    ax2.set_title("Dollar Gain on $155,000 Invested")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Add value labels
    for bar, val in zip(bars2, results_df["actual_gain"]):
        x_pos = bar.get_width()
        offset = 300 if val >= 0 else -300
        ha = "left" if val >= 0 else "right"
        ax2.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f"${val:+,.0f}", va="center", ha=ha, fontsize=10, fontweight="bold")

    # 3. Risk-Return scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df["max_drawdown"] * 100, results_df["actual_gain_pct"],
                         c=results_df["sharpe"], cmap="RdYlGn", s=200, edgecolors="black", linewidth=1.5)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax3.set_xlabel("Maximum Drawdown (%)")
    ax3.set_ylabel("Actual Return (%)")
    ax3.set_title("Risk vs Return (color = Sharpe Ratio)")
    plt.colorbar(scatter, ax=ax3, label="Sharpe Ratio")

    # Label points
    for _, row in results_df.iterrows():
        label = row["strategy"].replace("Entry Stop ", "Entry\n").replace("Trail ", "Trail\n").replace(" (from peak)", "")
        label = label.replace("Hybrid ", "Hybrid\n")
        ax3.annotate(label, (row["max_drawdown"]*100, row["actual_gain_pct"]),
                    textcoords="offset points", xytext=(8, 0), fontsize=8)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary table - sorted by actual gain
    table_data = []
    for _, row in results_df.sort_values("actual_gain", ascending=False).iterrows():
        table_data.append([
            row["strategy"][:28],
            f"${row['final_value']:,.0f}",
            f"${row['actual_gain']:+,.0f}",
            f"{row['actual_gain_pct']:+.2f}%",
            f"{row['max_drawdown']:.1%}",
            f"{row['sharpe']:.2f}"
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=["Strategy", "Final Value", "Gain ($)", "Return %", "Max DD", "Sharpe"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color the header
    for i in range(6):
        table[(0, i)].set_facecolor("#1565C0")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Color rows based on gain (green gradient for better performers)
    max_gain = results_df["actual_gain_pct"].max()
    min_gain = results_df["actual_gain_pct"].min()
    for i, (_, row) in enumerate(results_df.sort_values("actual_gain", ascending=False).iterrows()):
        # Interpolate color from light green to dark green based on relative performance
        norm_val = (row["actual_gain_pct"] - min_gain) / (max_gain - min_gain) if max_gain != min_gain else 0.5
        green_intensity = int(200 + 55 * (1 - norm_val))
        color = f"#{green_intensity:02x}EE{green_intensity:02x}"
        for j in range(6):
            table[(i+1, j)].set_facecolor(color)

    ax4.set_title("Strategy Comparison Summary\n(Sorted by Actual Gain)", fontsize=11, fontweight="bold")

    plt.tight_layout()

    # Save
    output_file = RESULTS_DIR / "actual_gains_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved visualization: {output_file}")

    # Also save to Excel
    excel_output = RESULTS_DIR / "actual_gains_analysis.xlsx"
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        results_df.sort_values("actual_gain", ascending=False).to_excel(
            writer, sheet_name="Actual Gains", index=False
        )
    print(f"Saved Excel: {excel_output}")

    plt.close()

    # Print best strategy
    best = results_df.loc[results_df["actual_gain"].idxmax()]
    print("\n" + "="*80)
    print("BEST STRATEGY BY ACTUAL GAIN:")
    print("="*80)
    print(f"  {best['strategy']}")
    print(f"  Final Value: ${best['final_value']:,.2f}")
    print(f"  Total Contributions: ${best['contributions']:,.0f}")
    print(f"  Actual Gain: ${best['actual_gain']:+,.2f} ({best['actual_gain_pct']:+.2f}%)")
    print(f"  Max Drawdown: {best['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {best['sharpe']:.2f}")


if __name__ == "__main__":
    results_df = calculate_actual_gains()
    create_visualization(results_df)
