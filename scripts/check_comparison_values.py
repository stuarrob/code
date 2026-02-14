#!/usr/bin/env python3
"""Check the actual values in the comparison spreadsheet."""

import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
excel_file = Path.home() / "trading" / "trailing_stop_comparison_latest.xlsx"

xl = pd.ExcelFile(excel_file)
print("Sheet names:", xl.sheet_names)
print()

for sheet in xl.sheet_names[1:]:  # Skip Summary
    df = pd.read_excel(excel_file, sheet_name=sheet)
    if "total_value" in df.columns:
        final_row = df.iloc[-1]
        total_val = final_row["total_value"]
        contrib = final_row["contributions"]
        actual_gain = total_val - contrib
        pct = (actual_gain / contrib) * 100

        print(f"{sheet}:")
        print(f"  Final total_value: ${total_val:,.2f}")
        print(f"  Total contributions: ${contrib:,.0f}")
        print(f"  Actual gain: ${actual_gain:,.2f} ({pct:+.2f}%)")
        print()
