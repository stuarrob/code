"""
OHLCV Price Data Downloader
Downloads historical price data for ETFs using yfinance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, List
import os


class PriceDownloader:
    """Download and manage OHLCV price data for ETFs"""

    def __init__(self, data_dir: str = str(Path.home() / "trade_data" / "ETFTrader" / "raw" / "prices")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fundamentals_data = []

    def download_single_etf(
        self, ticker: str, period: str = "3y", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single ETF

        Args:
            ticker: ETF ticker symbol
            period: Time period (3y = 3 years, 5y = 5 years, etc.)
            interval: Data interval (1d = daily, 1wk = weekly)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            print(f"  Downloading {ticker}...")

            # Download data
            etf = yf.Ticker(ticker)
            df = etf.history(period=period, interval=interval)

            if df.empty:
                print(f"    ⚠️  No data available for {ticker}")
                return None

            # Clean and standardize column names
            df = df.reset_index()
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]

            # Rename columns to standard format
            column_mapping = {
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }

            # Keep only relevant columns
            available_cols = [
                col for col in column_mapping.values() if col in df.columns
            ]
            df = df[available_cols]

            # Ensure date is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            # Add ticker column
            df["ticker"] = ticker

            # Calculate adjusted close (if dividends/splits occurred)
            # yfinance already provides adjusted close in the Close column

            print(f"    ✅ Downloaded {len(df)} rows of data")
            print(f"    📅 Date range: {df['date'].min()} to {df['date'].max()}")

            # Save fundamental data
            info = etf.info
            self.fundamentals_data.append(
                {
                    "ticker": ticker,
                    "name": info.get("longName", info.get("shortName", ticker)),
                    "category": info.get("category", "Unknown"),
                    "expense_ratio": info.get("annualReportExpenseRatio", None),
                    "aum": info.get("totalAssets", None),
                    "ytd_return": info.get("ytdReturn", None),
                    "inception_date": info.get("fundInceptionDate", None),
                    "exchange": info.get("exchange", None),
                    "currency": info.get("currency", "USD"),
                }
            )

            return df

        except Exception as e:
            print(f"    ❌ Error downloading {ticker}: {str(e)}")
            return None

    def save_etf_data(self, ticker: str, df: pd.DataFrame) -> None:
        """Save ETF data to individual CSV file"""
        if df is not None and not df.empty:
            file_path = self.data_dir / f"{ticker}.csv"
            df.to_csv(file_path, index=False)
            print(f"    💾 Saved to {file_path}")

    def download_etf_universe(
        self, universe_file: str, period: str = "3y", max_etfs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Download price data for entire ETF universe

        Args:
            universe_file: Path to CSV file with ETF universe
            period: Historical period to download
            max_etfs: Maximum number of ETFs to download (for testing)

        Returns:
            Summary DataFrame with download results
        """
        print("=" * 60)
        print("Starting ETF Price Data Download")
        print("=" * 60)

        # Load universe
        universe_df = pd.read_csv(universe_file)
        print(f"\nLoaded {len(universe_df)} ETFs from universe file")

        if max_etfs:
            universe_df = universe_df.head(max_etfs)
            print(f"Limiting to first {max_etfs} ETFs for testing")

        # Download data for each ETF
        results = []

        for i, row in universe_df.iterrows():
            ticker = row["ticker"]
            print(f"\n[{i+1}/{len(universe_df)}] Processing {ticker}...")

            # Download data
            df = self.download_single_etf(ticker, period=period)

            # Save to file
            if df is not None:
                self.save_etf_data(ticker, df)

                results.append(
                    {
                        "ticker": ticker,
                        "success": True,
                        "num_rows": len(df),
                        "start_date": df["date"].min(),
                        "end_date": df["date"].max(),
                        "days_of_data": (df["date"].max() - df["date"].min()).days,
                    }
                )
            else:
                results.append(
                    {
                        "ticker": ticker,
                        "success": False,
                        "num_rows": 0,
                        "start_date": None,
                        "end_date": None,
                        "days_of_data": 0,
                    }
                )

            # Rate limiting to avoid API throttling
            time.sleep(0.3)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Save fundamentals data
        fundamentals_df = pd.DataFrame(self.fundamentals_data)
        fundamentals_path = str(Path.home() / "trade_data" / "ETFTrader" / "raw" / "fundamentals.csv")
        fundamentals_df.to_csv(fundamentals_path, index=False)
        print(f"\n💾 Saved fundamentals data to: {fundamentals_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("Download Complete!")
        print("=" * 60)

        successful = results_df["success"].sum()
        failed = len(results_df) - successful

        print(f"\n📊 Summary:")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  Success rate: {successful/len(results_df)*100:.1f}%")

        if successful > 0:
            print(f"\n📈 Data Statistics:")
            print(
                f"  Average days of data: {results_df[results_df['success']]['days_of_data'].mean():.0f}"
            )
            print(
                f"  Min days: {results_df[results_df['success']]['days_of_data'].min():.0f}"
            )
            print(
                f"  Max days: {results_df[results_df['success']]['days_of_data'].max():.0f}"
            )

        return results_df

    def update_existing_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Update existing ETF data with latest prices"""
        file_path = self.data_dir / f"{ticker}.csv"

        if not file_path.exists():
            print(f"No existing data for {ticker}, downloading fresh...")
            return self.download_single_etf(ticker)

        # Load existing data
        existing_df = pd.read_csv(file_path)
        existing_df["date"] = pd.to_datetime(existing_df["date"])

        last_date = existing_df["date"].max()
        today = datetime.now()

        # Check if update is needed
        if (today - last_date).days <= 1:
            print(f"  {ticker} is already up to date")
            return existing_df

        print(f"  Updating {ticker} from {last_date.date()} to {today.date()}...")

        # Download new data
        etf = yf.Ticker(ticker)
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        new_df = etf.history(start=start_date)

        if new_df.empty:
            print(f"    No new data available")
            return existing_df

        # Clean new data
        new_df = new_df.reset_index()
        new_df.columns = [col.lower().replace(" ", "_") for col in new_df.columns]
        new_df["ticker"] = ticker

        # Combine and save
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
        combined_df = combined_df.sort_values("date")

        self.save_etf_data(ticker, combined_df)

        print(f"    ✅ Added {len(new_df)} new rows")

        return combined_df


def main():
    """Main execution function"""
    # Initialize downloader
    downloader = PriceDownloader()

    # Download data for ETF universe
    universe_file = str(Path.home() / "trade_data" / "ETFTrader" / "raw" / "etf_universe.csv")

    if not os.path.exists(universe_file):
        print(f"❌ Universe file not found: {universe_file}")
        print("Please run etf_scraper.py first to create the ETF universe")
        return

    # Download price data (3 years of daily data)
    results = downloader.download_etf_universe(
        universe_file=universe_file,
        period="3y",
        max_etfs=None,  # Set to small number for testing, None for all
    )

    # Save results summary
    results_path = str(Path.home() / "trade_data" / "ETFTrader" / "raw" / "download_results.csv")
    results.to_csv(results_path, index=False)
    print(f"\n💾 Saved download results to: {results_path}")


if __name__ == "__main__":
    main()
