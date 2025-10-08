"""
Data Validation Module
Validates ETF price data for completeness, quality, and consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os


class DataValidator:
    """Validate ETF price data quality"""

    def __init__(
        self,
        prices_dir: str = "data/raw/prices",
        fundamentals_file: str = "data/raw/fundamentals.csv",
    ):
        self.prices_dir = Path(prices_dir)
        self.fundamentals_file = fundamentals_file
        self.validation_results = []

    def validate_single_etf(self, ticker: str) -> Dict:
        """
        Validate data for a single ETF

        Returns:
            Dictionary with validation metrics
        """
        file_path = self.prices_dir / f"{ticker}.csv"

        if not file_path.exists():
            return {
                "ticker": ticker,
                "status": "MISSING",
                "error": "File not found",
            }

        try:
            df = pd.read_csv(file_path)
            df["date"] = pd.to_datetime(df["date"], utc=True)

            # Calculate metrics
            num_rows = len(df)
            start_date = df["date"].min()
            end_date = df["date"].max()
            date_range_days = (end_date - start_date).days

            # Check for missing dates (weekdays only)
            expected_days = pd.bdate_range(start=start_date, end=end_date)
            actual_dates = pd.to_datetime(df["date"], utc=True).dt.date
            missing_dates = set(expected_days.date) - set(actual_dates)
            missing_count = len(missing_dates)
            missing_pct = (
                (missing_count / len(expected_days)) * 100
                if len(expected_days) > 0
                else 0
            )

            # Check for data quality issues
            null_values = df.isnull().sum().sum()
            zero_volume_days = (df["volume"] == 0).sum()
            duplicate_dates = df["date"].duplicated().sum()

            # Check price consistency (no extreme gaps)
            df = df.sort_values("date")
            df["price_change_pct"] = df["close"].pct_change() * 100
            extreme_changes = (
                df["price_change_pct"].abs() > 50
            ).sum()  # >50% daily change

            # Check for negative prices (should never happen)
            negative_prices = (df[["open", "high", "low", "close"]] < 0).sum().sum()

            # Overall status
            issues = []
            if num_rows < 500:  # Less than ~2 years of data
                issues.append("insufficient_data")
            if missing_pct > 10:
                issues.append("high_missing_rate")
            if null_values > 0:
                issues.append("null_values")
            if extreme_changes > 5:
                issues.append("extreme_volatility")
            if negative_prices > 0:
                issues.append("negative_prices")
            if duplicate_dates > 0:
                issues.append("duplicate_dates")

            status = (
                "OK"
                if len(issues) == 0
                else "WARNING" if len(issues) <= 2 else "CRITICAL"
            )

            return {
                "ticker": ticker,
                "status": status,
                "num_rows": num_rows,
                "start_date": start_date,
                "end_date": end_date,
                "date_range_days": date_range_days,
                "missing_dates": missing_count,
                "missing_pct": round(missing_pct, 2),
                "null_values": null_values,
                "zero_volume_days": zero_volume_days,
                "duplicate_dates": duplicate_dates,
                "extreme_changes": extreme_changes,
                "negative_prices": negative_prices,
                "issues": ", ".join(issues) if issues else "none",
                "avg_volume": df["volume"].mean(),
                "avg_price": df["close"].mean(),
            }

        except Exception as e:
            return {
                "ticker": ticker,
                "status": "ERROR",
                "error": str(e),
            }

    def validate_universe(
        self, universe_file: str = "data/raw/etf_universe.csv"
    ) -> pd.DataFrame:
        """
        Validate all ETFs in the universe

        Returns:
            DataFrame with validation results for all ETFs
        """
        print("=" * 60)
        print("Data Validation Report")
        print("=" * 60)

        # Load universe
        if os.path.exists(universe_file):
            universe_df = pd.read_csv(universe_file)
            tickers = universe_df["ticker"].tolist()
        else:
            # Get all CSV files in prices directory
            tickers = [f.stem for f in self.prices_dir.glob("*.csv")]

        print(f"\nValidating {len(tickers)} ETFs...")

        # Validate each ETF
        results = []
        for i, ticker in enumerate(tickers):
            print(f"  [{i+1}/{len(tickers)}] Validating {ticker}...")
            result = self.validate_single_etf(ticker)
            results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Generate summary statistics
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)

        status_counts = results_df["status"].value_counts()
        print("\n📊 Status Distribution:")
        for status, count in status_counts.items():
            pct = count / len(results_df) * 100
            print(f"  {status}: {count} ({pct:.1f}%)")

        if "num_rows" in results_df.columns:
            ok_df = results_df[results_df["status"] == "OK"]
            if len(ok_df) > 0:
                print(f"\n📈 Data Coverage (OK status only):")
                print(f"  Average rows per ETF: {ok_df['num_rows'].mean():.0f}")
                print(
                    f"  Average date range: {ok_df['date_range_days'].mean():.0f} days"
                )
                print(f"  Average missing data: {ok_df['missing_pct'].mean():.2f}%")

        # Identify problematic ETFs
        if "status" in results_df.columns:
            critical = results_df[results_df["status"] == "CRITICAL"]
            if len(critical) > 0:
                print(f"\n⚠️  Critical Issues Found ({len(critical)} ETFs):")
                for _, row in critical.iterrows():
                    print(f"  {row['ticker']}: {row.get('issues', 'unknown')}")

        # Data quality metrics
        if "missing_pct" in results_df.columns:
            total_missing = results_df["missing_pct"].mean()
            print(f"\n📊 Overall Data Quality:")
            print(f"  Average missing data rate: {total_missing:.2f}%")

            if total_missing < 5:
                print(f"  ✅ Excellent - Less than 5% missing data")
            elif total_missing < 10:
                print(f"  ⚠️  Acceptable - Less than 10% missing data")
            else:
                print(f"  ❌ Poor - More than 10% missing data")

        return results_df

    def generate_report(self, results_df: pd.DataFrame, output_file: str) -> None:
        """Generate detailed validation report"""
        # Save full results
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved validation report to: {output_file}")

        # Create summary report
        summary_file = output_file.replace(".csv", "_summary.txt")

        with open(summary_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("ETF Data Validation Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Total ETFs validated: {len(results_df)}\n\n")

            # Status breakdown
            f.write("STATUS BREAKDOWN:\n")
            status_counts = results_df["status"].value_counts()
            for status, count in status_counts.items():
                pct = count / len(results_df) * 100
                f.write(f"  {status}: {count} ({pct:.1f}%)\n")

            # Data quality
            if "missing_pct" in results_df.columns:
                f.write(f"\nDATA QUALITY:\n")
                f.write(
                    f"  Average missing data: {results_df['missing_pct'].mean():.2f}%\n"
                )
                f.write(f"  Min missing data: {results_df['missing_pct'].min():.2f}%\n")
                f.write(f"  Max missing data: {results_df['missing_pct'].max():.2f}%\n")

            # Coverage
            if "num_rows" in results_df.columns:
                f.write(f"\nDATA COVERAGE:\n")
                f.write(
                    f"  Average rows per ETF: {results_df['num_rows'].mean():.0f}\n"
                )
                f.write(f"  Min rows: {results_df['num_rows'].min():.0f}\n")
                f.write(f"  Max rows: {results_df['num_rows'].max():.0f}\n")

            # Issues
            critical = results_df[results_df["status"] == "CRITICAL"]
            if len(critical) > 0:
                f.write(f"\nCRITICAL ISSUES:\n")
                for _, row in critical.iterrows():
                    f.write(f"  {row['ticker']}: {row.get('issues', 'unknown')}\n")

        print(f"💾 Saved summary report to: {summary_file}")

    def get_quality_score(self, results_df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100)

        Factors:
        - Status distribution (40%)
        - Missing data rate (30%)
        - Data coverage (30%)
        """
        # Status score
        status_weights = {
            "OK": 100,
            "WARNING": 60,
            "CRITICAL": 20,
            "MISSING": 0,
            "ERROR": 10,
        }
        status_score = results_df["status"].map(status_weights).mean()

        # Missing data score (inverse - lower is better)
        if "missing_pct" in results_df.columns:
            missing_score = (
                100 - results_df["missing_pct"].mean() * 5
            )  # Penalize 5 points per 1% missing
            missing_score = max(0, min(100, missing_score))
        else:
            missing_score = 50

        # Coverage score (based on number of rows)
        if "num_rows" in results_df.columns:
            avg_rows = results_df["num_rows"].mean()
            coverage_score = min(
                100, (avg_rows / 750) * 100
            )  # 750 rows = 3 years ≈ 100%
        else:
            coverage_score = 50

        # Weighted total
        total_score = (
            (status_score * 0.4) + (missing_score * 0.3) + (coverage_score * 0.3)
        )

        return round(total_score, 1)


def main():
    """Main execution function"""
    validator = DataValidator()

    # Validate entire universe
    results = validator.validate_universe()

    # Generate report
    output_file = "results/data_validation_report.csv"
    os.makedirs("results", exist_ok=True)
    validator.generate_report(results, output_file)

    # Calculate quality score
    quality_score = validator.get_quality_score(results)
    print(f"\n📊 Overall Data Quality Score: {quality_score}/100")

    if quality_score >= 90:
        print("   ✅ Excellent quality!")
    elif quality_score >= 75:
        print("   ✅ Good quality")
    elif quality_score >= 60:
        print("   ⚠️  Acceptable quality")
    else:
        print("   ❌ Poor quality - review critical issues")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
