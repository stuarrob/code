#!/usr/bin/env python3
"""
ETF Universe Collection Script

Collects comprehensive universe of 2000+ ETFs and downloads price data in parallel.
This script implements the same logic as the 00_etf_universe_collection.ipynb notebook.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Dynamic project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.etf_universe_builder import (
    ComprehensiveETFScraper,
    ParallelETFDownloader,
)

# Define paths
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PRICES_DIR = DATA_DIR / "prices"
UNIVERSE_FILE = DATA_DIR / "etf_universe.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
PRICES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Run ETF universe collection"""
    print("=" * 80)
    print("ETF UNIVERSE COLLECTION")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Start Time: {datetime.now()}\n")

    # Step 1: Scrape ETF Universe
    print("\nStep 1: Scraping ETF Universe from Multiple Sources")
    print("=" * 80)

    scraper = ComprehensiveETFScraper()

    # Source 1: ETF Database
    print("\n[1/3] ETF Database (etfdb.com)...")
    etfdb_etfs = scraper.scrape_etfdb_all_etfs(max_pages=50)
    print(f"  → Collected {len(etfdb_etfs)} ETFs")

    # Source 2: Nasdaq
    print("\n[2/3] Nasdaq listings...")
    nasdaq_etfs = scraper.scrape_nasdaq_listings()
    print(f"  → Collected {len(nasdaq_etfs)} ETFs")

    # Source 3: Seed list
    print("\n[3/3] Comprehensive seed list...")
    seed_etfs = scraper._get_comprehensive_seed_list()
    print(f"  → Loaded {len(seed_etfs)} ETFs")

    # Step 2: Merge and Deduplicate
    print("\n\nStep 2: Merging and Deduplicating")
    print("=" * 80)

    all_sources = [etfdb_etfs, nasdaq_etfs, seed_etfs]
    merged_universe = scraper.merge_and_deduplicate(all_sources)

    print(f"\n✓ Total unique ETFs: {len(merged_universe)}")

    # Step 3: Filter Universe
    print("\n\nStep 3: Filtering Universe")
    print("=" * 80)

    filtered_universe = scraper.filter_universe(
        merged_universe, min_aum=10e6, remove_leveraged=True
    )

    print(f"\n✓ Filtered universe: {len(filtered_universe)} ETFs")

    # Step 4: Parallel Download
    print("\n\nStep 4: Downloading Price Data (Parallel)")
    print("=" * 80)

    downloader = ParallelETFDownloader(
        output_dir=PRICES_DIR, min_years=2.0, max_workers=20, max_retries=3
    )

    print(f"\nConfiguration:")
    print(f"  Parallel workers: {downloader.max_workers}")
    print(f"  Minimum data: {downloader.min_years} years")
    print(f"  ETFs to download: {len(filtered_universe)}")
    print(f"  Estimated time: {len(filtered_universe) / downloader.max_workers / 2:.0f}-{len(filtered_universe) / downloader.max_workers:.0f} minutes")
    print()

    tickers = filtered_universe["ticker"].tolist()

    start_time = datetime.now()
    download_results = downloader.download_batch(tickers)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds() / 60

    # Step 5: Analyze Results
    print("\n\nStep 5: Analyzing Results")
    print("=" * 80)

    total = len(download_results)
    successful = download_results["success"].sum()
    failed = total - successful
    success_rate = (successful / total) * 100

    print(f"\nDownload Statistics:")
    print(f"  Total ETFs: {total}")
    print(f"  Successful: {successful} ({success_rate:.1f}%)")
    print(f"  Failed: {failed} ({100-success_rate:.1f}%)")
    print(f"  Duration: {duration:.1f} minutes")
    print(f"  Rate: {total / duration:.1f} ETFs/minute")

    # Categorize failures
    if failed > 0:
        print(f"\nFailure analysis:")
        failed_results = download_results[~download_results["success"]]

        failure_categories = {}
        for msg in failed_results["message"]:
            if "No data" in msg:
                failure_categories["No data returned"] = (
                    failure_categories.get("No data returned", 0) + 1
                )
            elif "Insufficient" in msg:
                failure_categories["Insufficient history"] = (
                    failure_categories.get("Insufficient history", 0) + 1
                )
            elif "missing" in msg.lower():
                failure_categories["Too much missing data"] = (
                    failure_categories.get("Too much missing data", 0) + 1
                )
            else:
                failure_categories["Other errors"] = (
                    failure_categories.get("Other errors", 0) + 1
                )

        for reason, count in sorted(failure_categories.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    # Step 6: Save Results
    print("\n\nStep 6: Saving Results")
    print("=" * 80)

    # Merge results with universe
    final_universe = filtered_universe.merge(
        download_results[["ticker", "success", "message"]], on="ticker", how="left"
    )

    final_universe["data_collection_date"] = datetime.now().strftime("%Y-%m-%d")

    # Save
    final_universe.to_csv(UNIVERSE_FILE, index=False)
    print(f"\n✓ Saved universe: {UNIVERSE_FILE}")

    results_file = (
        RESULTS_DIR
        / f"etf_download_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    download_results.to_csv(results_file, index=False)
    print(f"✓ Saved results: {results_file}")

    # Final Summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nCollection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nETF Counts:")
    print(f"  Total scraped: {len(merged_universe)}")
    print(f"  After filtering: {len(filtered_universe)}")
    print(f"  Successfully downloaded: {successful}")
    print(f"  Failed downloads: {failed}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"\nFiles:")
    print(f"  Universe: {UNIVERSE_FILE}")
    print(f"  Price data: {PRICES_DIR} ({successful} files)")
    print(f"  Results log: {results_file}")
    print(f"\n✓ ETF Universe Collection Complete!")
    print("=" * 80)

    return successful


if __name__ == "__main__":
    try:
        successful_count = main()
        print(f"\n✓ Successfully collected {successful_count} ETFs")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠ Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
