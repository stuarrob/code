"""
Create the data validation notebook programmatically
"""

import nbformat as nbf

# Create new notebook
nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (ETFTrader)",
        "language": "python",
        "name": "etftrader",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.0",
    },
}

# Cells
cells = []

# Title
cells.append(
    nbf.v4.new_markdown_cell(
        """# ETF Data Validation Dashboard

This notebook validates the quality and completeness of the ETF data we've collected.

**Note:** Run this notebook from the project root (`/home/stuar/code/ETFTrader`) using `jupyter lab`

## Contents:
1. ETF Universe Overview
2. Price Data Quality Check
3. Missing Data Analysis
4. Data Coverage Visualization
5. Fundamental Data Review
6. Sample Price Data Exploration
7. Summary & Next Steps"""
    )
)

# Cell 1: Imports
cells.append(
    nbf.v4.new_code_cell(
        """# Import libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Find project root by looking for marker files
current_path = Path.cwd().resolve()
project_root = current_path

# Search up the directory tree for project root
while project_root != project_root.parent:
    if (project_root / 'src' / 'data_collection').exists() or (project_root / 'requirements.txt').exists():
        break
    project_root = project_root.parent

# Ensure project root is in path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from src.data_collection.data_validator import DataValidator

# Define data paths (absolute)
DATA_ROOT = project_root / 'data'
UNIVERSE_FILE = DATA_ROOT / 'raw' / 'etf_universe.csv'
PRICES_DIR = DATA_ROOT / 'raw' / 'prices'
FUNDAMENTALS_FILE = DATA_ROOT / 'raw' / 'fundamentals.csv'

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print(f"✅ Libraries imported successfully")
print(f"📂 Project root: {project_root}")
print(f"📂 Current directory: {current_path}")"""
    )
)

# Section 1
cells.append(nbf.v4.new_markdown_cell("## 1. ETF Universe Overview"))

cells.append(
    nbf.v4.new_code_cell(
        """# Load ETF universe
universe = pd.read_csv(UNIVERSE_FILE)

print(f"Total ETFs in Universe: {len(universe)}")
print(f"\\nDataset shape: {universe.shape}")

# Display sample
print("\\nSample ETFs:")
universe.head(10)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Category distribution
plt.figure(figsize=(14, 6))

category_counts = universe['category'].value_counts().head(15)
plt.barh(range(len(category_counts)), category_counts.values)
plt.yticks(range(len(category_counts)), category_counts.index)
plt.xlabel('Number of ETFs')
plt.title('Top 15 ETF Categories in Universe')
plt.tight_layout()
plt.show()

print(f"\\nTotal Categories: {universe['category'].nunique()}")"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# AUM distribution
plt.figure(figsize=(12, 5))

# Filter out NaN values
aum_data = universe['aum'].dropna()

if len(aum_data) > 0:
    plt.subplot(1, 2, 1)
    plt.hist(np.log10(aum_data), bins=30, edgecolor='black')
    plt.xlabel('Log10(AUM)')
    plt.ylabel('Frequency')
    plt.title('AUM Distribution (Log Scale)')

    plt.subplot(1, 2, 2)
    top_10_aum = universe.nlargest(10, 'aum')[['ticker', 'name', 'aum']]
    plt.barh(range(len(top_10_aum)), top_10_aum['aum'].values / 1e9)
    plt.yticks(range(len(top_10_aum)), top_10_aum['ticker'].values)
    plt.xlabel('AUM (Billions USD)')
    plt.title('Top 10 ETFs by AUM')

    plt.tight_layout()
    plt.show()

    print(f"\\nAUM Statistics (in Billions):")
    print(f"Mean: ${aum_data.mean()/1e9:.2f}B")
    print(f"Median: ${aum_data.median()/1e9:.2f}B")
    print(f"Min: ${aum_data.min()/1e9:.2f}B")
    print(f"Max: ${aum_data.max()/1e9:.2f}B")
else:
    print("No AUM data available")"""
    )
)

# Section 2
cells.append(nbf.v4.new_markdown_cell("## 2. Price Data Quality Check"))

cells.append(
    nbf.v4.new_code_cell(
        """# Run data validator with absolute paths
validator = DataValidator(
    prices_dir=str(PRICES_DIR),
    fundamentals_file=str(FUNDAMENTALS_FILE)
)
validation_results = validator.validate_universe(str(UNIVERSE_FILE))

print("\\nValidation Results (first 10):")
validation_results.head(10)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Status distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
status_counts = validation_results['status'].value_counts()
plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Data Validation Status Distribution')

plt.subplot(1, 2, 2)
colors = {'OK': 'green', 'WARNING': 'orange', 'CRITICAL': 'red', 'MISSING': 'gray', 'ERROR': 'purple'}
bars = plt.bar(range(len(status_counts)), status_counts.values,
               color=[colors.get(x, 'blue') for x in status_counts.index])
plt.xticks(range(len(status_counts)), status_counts.index, rotation=45)
plt.ylabel('Count')
plt.title('Validation Status Counts')

plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Calculate quality score
quality_score = validator.get_quality_score(validation_results)

print(f"\\n{'='*60}")
print(f"Overall Data Quality Score: {quality_score}/100")
print(f"{'='*60}")

if quality_score >= 90:
    print("✅ EXCELLENT - Data is ready for production use")
elif quality_score >= 75:
    print("✅ GOOD - Data is acceptable with minor issues")
elif quality_score >= 60:
    print("⚠️  ACCEPTABLE - Some data quality concerns")
else:
    print("❌ POOR - Significant data quality issues need attention")"""
    )
)

# Section 3
cells.append(nbf.v4.new_markdown_cell("## 3. Missing Data Analysis"))

cells.append(
    nbf.v4.new_code_cell(
        """# Filter for successful validations
valid_data = validation_results[validation_results['status'].isin(['OK', 'WARNING'])]

if len(valid_data) > 0 and 'missing_pct' in valid_data.columns:
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.hist(valid_data['missing_pct'], bins=30, edgecolor='black')
    plt.xlabel('Missing Data %')
    plt.ylabel('Number of ETFs')
    plt.title('Distribution of Missing Data Percentage')
    plt.axvline(valid_data['missing_pct'].mean(), color='red', linestyle='--',
                label=f'Mean: {valid_data["missing_pct"].mean():.2f}%')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Top 10 ETFs with most missing data
    top_missing = valid_data.nlargest(10, 'missing_pct')[['ticker', 'missing_pct']]
    plt.barh(range(len(top_missing)), top_missing['missing_pct'].values)
    plt.yticks(range(len(top_missing)), top_missing['ticker'].values)
    plt.xlabel('Missing Data %')
    plt.title('Top 10 ETFs with Most Missing Data')

    plt.tight_layout()
    plt.show()

    print(f"\\nMissing Data Statistics:")
    print(f"Average: {valid_data['missing_pct'].mean():.2f}%")
    print(f"Median: {valid_data['missing_pct'].median():.2f}%")
    print(f"Max: {valid_data['missing_pct'].max():.2f}%")
else:
    print("No valid data to analyze missing percentages")"""
    )
)

# Section 4
cells.append(nbf.v4.new_markdown_cell("## 4. Data Coverage Visualization"))

cells.append(
    nbf.v4.new_code_cell(
        """if len(valid_data) > 0 and 'num_rows' in valid_data.columns:
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(valid_data['num_rows'], valid_data['missing_pct'], alpha=0.5)
    plt.xlabel('Number of Data Points')
    plt.ylabel('Missing Data %')
    plt.title('Data Points vs Missing Data')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(valid_data['date_range_days'], bins=30, edgecolor='black')
    plt.xlabel('Date Range (Days)')
    plt.ylabel('Number of ETFs')
    plt.title('Historical Data Coverage')
    plt.axvline(730, color='green', linestyle='--', label='2 years')
    plt.axvline(1095, color='blue', linestyle='--', label='3 years')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\\nData Coverage Statistics:")
    print(f"Average rows per ETF: {valid_data['num_rows'].mean():.0f}")
    print(f"Average date range: {valid_data['date_range_days'].mean():.0f} days ({valid_data['date_range_days'].mean()/365:.1f} years)")

    # ETFs with 2+ years of data
    two_year_coverage = (valid_data['date_range_days'] >= 730).sum()
    three_year_coverage = (valid_data['date_range_days'] >= 1095).sum()

    print(f"\\nCoverage Summary:")
    print(f"ETFs with 2+ years: {two_year_coverage} ({two_year_coverage/len(valid_data)*100:.1f}%)")
    print(f"ETFs with 3+ years: {three_year_coverage} ({three_year_coverage/len(valid_data)*100:.1f}%)")
else:
    print("No valid data to analyze coverage")"""
    )
)

# Section 5
cells.append(nbf.v4.new_markdown_cell("## 5. Fundamental Data Review"))

cells.append(
    nbf.v4.new_code_cell(
        """# Load fundamentals
fundamentals = pd.read_csv(FUNDAMENTALS_FILE)

print(f"Total ETFs with fundamental data: {len(fundamentals)}")
print(f"\\nFundamental Data Sample:")
fundamentals.head(10)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Expense ratio distribution
expense_data = fundamentals['expense_ratio'].dropna()

if len(expense_data) > 0:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(expense_data, bins=30, edgecolor='black')
    plt.xlabel('Expense Ratio')
    plt.ylabel('Number of ETFs')
    plt.title('Expense Ratio Distribution')
    plt.axvline(0.01, color='green', linestyle='--', label='1% threshold')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Top 10 lowest expense ratios
    top_low_expense = fundamentals.nsmallest(10, 'expense_ratio')[['ticker', 'name', 'expense_ratio']]
    plt.barh(range(len(top_low_expense)), top_low_expense['expense_ratio'].values)
    plt.yticks(range(len(top_low_expense)), top_low_expense['ticker'].values)
    plt.xlabel('Expense Ratio')
    plt.title('Top 10 Lowest Expense Ratio ETFs')

    plt.tight_layout()
    plt.show()

    print(f"\\nExpense Ratio Statistics:")
    print(f"Mean: {expense_data.mean():.4f} ({expense_data.mean()*100:.2f}%)")
    print(f"Median: {expense_data.median():.4f} ({expense_data.median()*100:.2f}%)")
    print(f"Min: {expense_data.min():.4f} ({expense_data.min()*100:.2f}%)")
    print(f"Max: {expense_data.max():.4f} ({expense_data.max()*100:.2f}%)")
else:
    print("No expense ratio data available")"""
    )
)

# Section 6
cells.append(nbf.v4.new_markdown_cell("## 6. Sample Price Data Exploration"))

cells.append(
    nbf.v4.new_code_cell(
        """# Load and display sample ETF data (SPY)
sample_etf = 'SPY'
sample_path = PRICES_DIR / f'{sample_etf}.csv'

if sample_path.exists():
    spy_data = pd.read_csv(sample_path)
    spy_data['date'] = pd.to_datetime(spy_data['date'])

    print(f"Sample ETF: {sample_etf}")
    print(f"Data points: {len(spy_data)}")
    print(f"Date range: {spy_data['date'].min()} to {spy_data['date'].max()}")

    # Plot price chart
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(spy_data['date'], spy_data['close'], linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{sample_etf} Price History')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.bar(spy_data['date'], spy_data['volume'], width=1)
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title(f'{sample_etf} Trading Volume')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Display sample data
    print(f"\\nSample Data (first 5 rows):")
    print(spy_data.head())
else:
    print(f"Data file not found for {sample_etf}")"""
    )
)

# Section 7
cells.append(nbf.v4.new_markdown_cell("## 7. Summary & Next Steps"))

cells.append(
    nbf.v4.new_code_cell(
        """print("=" * 60)
print("DATA COLLECTION SUMMARY")
print("=" * 60)
print(f"\\n✅ ETF Universe: {len(universe)} ETFs collected")
print(f"✅ Price Data: {len(validation_results[validation_results['status'].isin(['OK', 'WARNING'])])} ETFs with valid data")
print(f"✅ Fundamentals: {len(fundamentals)} ETFs with metadata")
print(f"\\n📊 Quality Score: {quality_score}/100")

# Check if we meet success criteria
success_criteria = []
success_criteria.append((len(universe) >= 100, f"Universe size: {len(universe)} >= 100"))

if len(valid_data) > 0:
    two_yr_count = len(valid_data[valid_data['date_range_days'] >= 730])
    success_criteria.append((two_yr_count >= 100, f"2+ years data: {two_yr_count} >= 100"))
    success_criteria.append((valid_data['missing_pct'].mean() < 5,
                            f"Missing data: {valid_data['missing_pct'].mean():.2f}% < 5%"))

print("\\n📋 Success Criteria:")
for met, msg in success_criteria:
    status = "✅" if met else "❌"
    print(f"  {status} {msg}")

if all([x[0] for x in success_criteria]):
    print("\\n🎉 All success criteria met! Ready for Phase 2.")
else:
    print("\\n⚠️  Some criteria not met. Review issues above.")

print("\\n📍 Next Steps:")
print("  1. Move to Phase 2: Signal Generation Engine")
print("  2. Implement technical indicators (MACD, RSI, Bollinger Bands)")
print("  3. Build composite signal framework")"""
    )
)

# Add cells to notebook
nb.cells = cells

# Save notebook
with open("notebooks/01_data_validation.ipynb", "w") as f:
    nbf.write(nb, f)

print("✅ Notebook created successfully: notebooks/01_data_validation.ipynb")
