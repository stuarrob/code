"""
Step 1: Universe Discovery

Loads the full US ETF universe (~5,000 tickers) by merging the curated
categorized list with the NASDAQ trader file.
"""

import pandas as pd
from pathlib import Path


def discover_universe(project_root: Path) -> tuple:
    """Discover the full ETF universe.

    Returns:
        (all_tickers, categories, universe_df)
        - all_tickers: sorted list of all unique ticker symbols
        - categories: dict mapping ticker -> category name
        - universe_df: DataFrame with ticker and category columns
    """
    import sys
    sys.path.insert(0, str(project_root / "src"))

    from data_collection.comprehensive_etf_list import load_full_universe
    from data_collection.etf_filters import LEVERAGED_ETFS

    all_tickers, categories = load_full_universe()

    curated_count = sum(1 for t in all_tickers if categories[t] != "Uncategorized")
    uncategorized_count = len(all_tickers) - curated_count
    leveraged_count = sum(1 for t in all_tickers if t in LEVERAGED_ETFS)

    print(f"Full universe: {len(all_tickers)} ETFs")
    print(f"  Categorized (curated):  {curated_count}")
    print(f"  Uncategorized (NASDAQ): {uncategorized_count}")
    print(f"  Leveraged/inverse:      {leveraged_count} (kept for data collection)")

    universe_df = pd.DataFrame({
        "ticker": all_tickers,
        "category": [categories.get(t, "Uncategorized") for t in all_tickers],
    })

    return all_tickers, categories, universe_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    tickers, cats, df = discover_universe(project_root)
    print(f"\nTop categories:")
    print(df.groupby("category").size().sort_values(ascending=False).head(10))
