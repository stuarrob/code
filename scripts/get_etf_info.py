#!/usr/bin/env python3
"""Get ETF information for trade sheet."""

import yfinance as yf

tickers = ["IDV", "VSS", "IEFA", "VSGX", "IQDY", "IDEV", "EWU", "DIM",
           "EWO", "DVYE", "FEZ", "FFNOX", "VEA", "EMLC", "EZU", "RWX",
           "DBEU", "DLS", "IEV", "HEDJ"]

print("Ticker|Full Name|Category|Expense Ratio|AUM")
print("-" * 100)

for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get("longName", info.get("shortName", "N/A"))
        category = info.get("category", "N/A")
        expense = info.get("annualReportExpenseRatio", info.get("expenseRatio", None))
        if expense is not None:
            expense = f"{expense*100:.2f}%"
        else:
            expense = "N/A"
        aum = info.get("totalAssets", None)
        if aum is not None:
            if aum > 1e9:
                aum = f"${aum/1e9:.1f}B"
            else:
                aum = f"${aum/1e6:.0f}M"
        else:
            aum = "N/A"
        print(f"{ticker}|{name}|{category}|{expense}|{aum}")
    except Exception as e:
        print(f"{ticker}|Error: {e}|N/A|N/A|N/A")
