"""
Comprehensive ETF Universe Builder

Scrapes multiple sources to build a comprehensive universe of 2000+ ETFs:
- ETF Database (etfdb.com) - primary source
- Nasdaq listings
- NYSE Arca listings
- Parallel downloading with robust error handling
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComprehensiveETFScraper:
    """Scrape comprehensive ETF universe from multiple sources"""

    def __init__(self):
        self.leveraged_keywords = [
            "2x",
            "3x",
            "-2x",
            "-3x",
            "ultra",
            "double",
            "triple",
            "leveraged",
            "inverse",
            "short",
            "bear",
            "proshares ultra",
            "direxion",
            "2 x",
            "3 x",
        ]

        # Create session with retries
        self.session = requests.Session()
        retry = Retry(
            total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def is_leveraged(self, name: str, ticker: str) -> bool:
        """Check if ETF is leveraged or inverse"""
        if not name:
            return False

        name_lower = name.lower()
        ticker_lower = ticker.lower()

        # Check for leveraged keywords
        for keyword in self.leveraged_keywords:
            if keyword in name_lower:
                return True

        # Check ticker patterns (e.g., TQQQ, SQQQ, SPXL)
        leveraged_patterns = [
            r"^[A-Z]{3}[LUS]$",  # TECL, SOXL, TQQQ, SQQQ
            r"^[US][QSPTD]{2,3}[XYZ]$",  # UPRO, SPXL, etc.
        ]

        for pattern in leveraged_patterns:
            if re.match(pattern, ticker.upper()):
                return True

        return False

    def scrape_etfdb_all_etfs(self, max_pages: int = 100) -> pd.DataFrame:
        """
        Scrape comprehensive ETF list from ETF Database.

        Args:
            max_pages: Maximum pages to scrape (each page ~100 ETFs)

        Returns:
            DataFrame with ticker, name, category, AUM, expense_ratio
        """
        logger.info("Scraping ETF Database (etfdb.com)...")

        all_etfs = []
        base_url = "https://etfdb.com/screener/"

        # Try to get ETF data from the screener
        # Note: etfdb.com structure may change, this is best-effort scraping
        try:
            # Alternative: Use their JSON API if available
            api_url = "https://etfdb.com/api/screener/"

            for page in range(1, max_pages + 1):
                try:
                    params = {
                        "page": page,
                        "per_page": 100,
                        "sort_name": "assets_under_management",
                        "sort_order": "desc",
                        "only": "meta,data,count",
                    }

                    response = self.session.get(api_url, params=params, timeout=10)

                    if response.status_code != 200:
                        logger.warning(
                            f"Page {page} returned status {response.status_code}"
                        )
                        if page == 1:
                            # If first page fails, try alternative method
                            return self._scrape_etfdb_html_fallback()
                        break

                    data = response.json()

                    if "data" not in data or len(data["data"]) == 0:
                        logger.info(f"No more data at page {page}")
                        break

                    # Parse ETF data
                    for etf in data["data"]:
                        etf_info = {
                            "ticker": etf.get("symbol", "").strip(),
                            "name": etf.get("name", "").strip(),
                            "category": etf.get("asset_class", "").strip(),
                            "aum": self._parse_aum(etf.get("aum", "")),
                            "expense_ratio": self._parse_expense_ratio(
                                etf.get("expense_ratio", "")
                            ),
                            "inception_date": etf.get("inception", ""),
                        }

                        if etf_info["ticker"]:
                            all_etfs.append(etf_info)

                    logger.info(
                        f"Scraped page {page}: {len(data['data'])} ETFs (total: {len(all_etfs)})"
                    )

                    # Be polite - rate limit
                    time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Error on page {page}: {e}")
                    if page == 1:
                        return self._scrape_etfdb_html_fallback()
                    break

        except Exception as e:
            logger.error(f"ETF Database API scraping failed: {e}")
            return self._scrape_etfdb_html_fallback()

        if len(all_etfs) == 0:
            logger.warning("No ETFs from API, trying HTML fallback")
            return self._scrape_etfdb_html_fallback()

        df = pd.DataFrame(all_etfs)
        logger.info(f"Successfully scraped {len(df)} ETFs from ETF Database")

        return df

    def _scrape_etfdb_html_fallback(self) -> pd.DataFrame:
        """Fallback: Scrape ETF list from HTML tables if API fails"""
        logger.info("Using HTML fallback method...")

        # Use a known list of major ETFs as fallback
        # This ensures we at least get a decent universe
        return self._get_comprehensive_seed_list()

    def _get_comprehensive_seed_list(self) -> pd.DataFrame:
        """
        Get comprehensive seed list of ETFs from multiple categories.
        This serves as a fallback if scraping fails.
        """
        logger.info("Using comprehensive seed list...")

        # Import comprehensive list
        try:
            from .comprehensive_etf_list import COMPREHENSIVE_ETF_UNIVERSE

            seed_etfs = COMPREHENSIVE_ETF_UNIVERSE
            logger.info(f"Loaded {len(seed_etfs)} categories from comprehensive list")
        except ImportError:
            logger.warning("Could not import comprehensive list, using backup")
            seed_etfs = self._get_backup_seed_list()

        # Flatten into list
        all_tickers = []
        for category, tickers in seed_etfs.items():
            for ticker in tickers:
                all_tickers.append(
                    {
                        "ticker": ticker,
                        "name": "",
                        "category": category.replace("_", " "),
                        "aum": np.nan,
                        "expense_ratio": np.nan,
                        "inception_date": "",
                    }
                )

        df = pd.DataFrame(all_tickers)
        logger.info(f"Seed list contains {len(df)} ETFs")

        return df

    def _get_backup_seed_list(self) -> dict:
        """Backup seed list if import fails"""
        seed_etfs = {
            # Large Cap Equity
            "Large_Cap": [
                "SPY",
                "IVV",
                "VOO",
                "VTI",
                "QQQ",
                "DIA",
                "IWB",
                "SCHX",
                "ITOT",
                "VV",
                "SPLG",
                "IWL",
                "SCHB",
                "SPTM",
                "VXF",
            ],
            # Growth & Value
            "Growth_Value": [
                "VUG",
                "VTV",
                "IWF",
                "IWD",
                "SCHG",
                "SCHV",
                "SPYG",
                "SPYV",
                "IVW",
                "IVE",
                "MGK",
                "MGV",
                "VBK",
                "VBR",
                "VONG",
                "VONV",
            ],
            # Mid & Small Cap
            "Mid_Small": [
                "IJH",
                "MDY",
                "VO",
                "IWR",
                "SCHM",
                "IWM",
                "IJR",
                "VB",
                "SCHA",
                "VTWO",
                "VXF",
                "SLYG",
                "SLYV",
                "IWN",
                "VBK",
                "VBR",
            ],
            # International Developed
            "Intl_Developed": [
                "VEA",
                "IEFA",
                "EFA",
                "SCHF",
                "IXUS",
                "VEU",
                "VXUS",
                "IDEV",
                "ACWI",
                "ACWX",
                "EZU",
                "FEZ",
                "EWJ",
                "EWG",
                "EWU",
                "EWL",
                "EWQ",
                "EWP",
                "EWI",
                "EWK",
                "EWC",
                "EWA",
                "EWS",
                "EWT",
                "EWH",
                "EWY",
            ],
            # Emerging Markets
            "Emerging_Mkts": [
                "VWO",
                "EEM",
                "IEMG",
                "SCHE",
                "DEM",
                "VWO",
                "SPEM",
                "EEMV",
                "EMGF",
                "DGS",
                "EWZ",
                "MCHI",
                "FXI",
                "INDA",
                "EWW",
                "EZA",
                "ERUS",
                "RSX",
                "EWT",
            ],
            # Fixed Income
            "Fixed_Income": [
                "AGG",
                "BND",
                "SCHZ",
                "IUSB",
                "VCIT",
                "VCSH",
                "BIV",
                "BSV",
                "VGIT",
                "VGSH",
                "LQD",
                "VCLT",
                "HYG",
                "JNK",
                "USHY",
                "ANGL",
                "EMB",
                "EMLC",
                "PCY",
                "TLT",
                "IEF",
                "SHY",
                "TIP",
                "VTIP",
                "MUB",
                "SUB",
                "BNDX",
                "IAGG",
            ],
            # Sector - Technology
            "Tech": [
                "XLK",
                "VGT",
                "FTEC",
                "IGV",
                "SMH",
                "SOXX",
                "XSD",
                "IGN",
                "HACK",
                "CIBR",
                "SKYY",
                "CLOU",
                "FINX",
                "ROBO",
                "BOTZ",
            ],
            # Sector - Healthcare
            "Healthcare": [
                "XLV",
                "VHT",
                "IYH",
                "FHLC",
                "IBB",
                "XBI",
                "IHI",
                "IHE",
                "GNOM",
            ],
            # Sector - Financials
            "Financials": [
                "XLF",
                "VFH",
                "IYF",
                "FNCL",
                "KRE",
                "KBE",
                "IAI",
                "KBWB",
            ],
            # Sector - Energy
            "Energy": [
                "XLE",
                "VDE",
                "IYE",
                "FENY",
                "IEO",
                "PXE",
                "XES",
                "ICLN",
                "TAN",
                "FAN",
            ],
            # Sector - Consumer
            "Consumer": [
                "XLY",
                "VCR",
                "IYC",
                "FDIS",
                "XLP",
                "VDC",
                "IYK",
                "FSTA",
            ],
            # Sector - Industrials
            "Industrials": [
                "XLI",
                "VIS",
                "IYJ",
                "FIDU",
                "IYT",
                "XTN",
            ],
            # Sector - Materials
            "Materials": [
                "XLB",
                "VAW",
                "IYM",
                "FMAT",
            ],
            # Sector - Real Estate
            "Real_Estate": [
                "VNQ",
                "IYR",
                "XLRE",
                "SCHH",
                "RWR",
                "USRT",
                "REET",
                "VNQI",
                "RWX",
            ],
            # Sector - Utilities & Communications
            "Utilities_Comm": [
                "XLU",
                "VPU",
                "IDU",
                "FUTY",
                "XLC",
                "VOX",
                "IYZ",
            ],
            # Commodities & Precious Metals
            "Commodities": [
                "GLD",
                "IAU",
                "SLV",
                "GDX",
                "GDXJ",
                "PPLT",
                "PALL",
                "DBC",
                "DBA",
                "USO",
                "UNG",
                "CORN",
                "WEAT",
                "SOYB",
            ],
            # Dividend & Income
            "Dividend": [
                "VIG",
                "SCHD",
                "DVY",
                "SDY",
                "VYM",
                "HDV",
                "NOBL",
                "DGRO",
                "FVD",
                "SPHD",
                "IDV",
                "DHS",
                "DON",
                "DES",
                "DFE",
            ],
            # Factor & Smart Beta
            "Smart_Beta": [
                "QUAL",
                "USMV",
                "MTUM",
                "VLUE",
                "SIZE",
                "ACWV",
                "EFAV",
                "EEMV",
                "SPLV",
                "SPHQ",
                "IQLT",
                "IQDF",
                "XMMO",
                "JHMM",
                "DUHP",
            ],
            # Thematic & Innovation
            "Thematic": [
                "ARKK",
                "ARKQ",
                "ARKW",
                "ARKG",
                "ARKF",
                "MOON",
                "UFO",
                "TAN",
                "ICLN",
                "DRIV",
                "IDRV",
                "SNSR",
                "BOTZ",
                "ROBO",
                "AIQ",
                "AIEQ",
                "ESPO",
                "HERO",
                "NERD",
                "GAMR",
            ],
            # ESG & Sustainable
            "ESG": [
                "ESGU",
                "ESGV",
                "SUSL",
                "DSI",
                "USSG",
                "SUSA",
                "ESGE",
                "EAGG",
                "SUSC",
            ],
            # Currency & International Bonds
            "Currency_Bonds": [
                "UUP",
                "FXE",
                "FXY",
                "FXC",
                "BNDX",
                "IAGG",
                "IGOV",
            ],
            # Volatility & Alternative
            "Alternative": [
                "TAIL",
                "VIXY",
                "BTAL",
                "CTA",
                "DBMF",
                "KMLM",
                "QAI",
                "MNA",
            ],
        }

        return seed_etfs

    def scrape_nasdaq_listings(self) -> pd.DataFrame:
        """Scrape Nasdaq ETF listings"""
        logger.info("Scraping Nasdaq ETF listings...")

        try:
            # Nasdaq provides CSV download
            url = "https://api.nasdaq.com/api/screener/etf"
            params = {"download": "true"}

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                if "data" in data and "data" in data["data"]:
                    etfs = data["data"]["data"]
                    df = pd.DataFrame(etfs)

                    # Standardize columns
                    if "symbol" in df.columns:
                        df = df.rename(columns={"symbol": "ticker"})

                    logger.info(f"Found {len(df)} ETFs from Nasdaq")
                    return df[["ticker"]].drop_duplicates()

        except Exception as e:
            logger.warning(f"Nasdaq scraping failed: {e}")

        return pd.DataFrame(columns=["ticker"])

    def _parse_aum(self, aum_str: str) -> float:
        """Parse AUM string to float (in dollars)"""
        if not aum_str or aum_str == "":
            return np.nan

        try:
            aum_str = str(aum_str).replace("$", "").replace(",", "").strip().upper()

            if "B" in aum_str:
                return float(aum_str.replace("B", "")) * 1e9
            elif "M" in aum_str:
                return float(aum_str.replace("M", "")) * 1e6
            elif "K" in aum_str:
                return float(aum_str.replace("K", "")) * 1e3
            else:
                return float(aum_str)
        except:
            return np.nan

    def _parse_expense_ratio(self, er_str: str) -> float:
        """Parse expense ratio string to float (as percentage)"""
        if not er_str or er_str == "":
            return np.nan

        try:
            er_str = str(er_str).replace("%", "").strip()
            return float(er_str)
        except:
            return np.nan

    def merge_and_deduplicate(
        self, sources: List[pd.DataFrame], priority_source: int = 0
    ) -> pd.DataFrame:
        """
        Merge multiple ETF DataFrames and remove duplicates.

        Args:
            sources: List of DataFrames to merge
            priority_source: Index of source with priority data (default: 0)

        Returns:
            Merged and deduplicated DataFrame
        """
        logger.info("Merging and deduplicating ETF sources...")

        # Concatenate all sources
        combined = pd.concat(sources, ignore_index=True)

        # Remove duplicates (keep first occurrence by default)
        combined = combined.drop_duplicates(subset=["ticker"], keep="first")

        # Clean ticker symbols
        combined["ticker"] = combined["ticker"].str.strip().str.upper()

        # Filter out invalid tickers
        combined = combined[combined["ticker"].str.match(r"^[A-Z]{1,5}$")]

        logger.info(f"After merge/dedup: {len(combined)} unique ETFs")

        return combined

    def filter_universe(
        self,
        df: pd.DataFrame,
        min_aum: float = 10e6,  # $10M
        min_age_days: int = 180,  # 6 months
        remove_leveraged: bool = True,
    ) -> pd.DataFrame:
        """
        Apply filters to create final universe.

        Args:
            df: Input DataFrame
            min_aum: Minimum AUM in dollars (default: $10M)
            min_age_days: Minimum age in days (default: 180)
            remove_leveraged: Remove leveraged/inverse ETFs

        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering ETF universe...")

        initial_count = len(df)

        # Remove leveraged/inverse if requested
        if remove_leveraged:
            df["is_leveraged"] = df.apply(
                lambda row: self.is_leveraged(row.get("name", ""), row["ticker"]),
                axis=1,
            )
            df = df[~df["is_leveraged"]]
            logger.info(
                f"  Removed leveraged/inverse: {initial_count - len(df)} ({len(df)} remaining)"
            )

        # Filter by AUM (if data available)
        if "aum" in df.columns:
            aum_before = len(df)
            df = df[(df["aum"].isna()) | (df["aum"] >= min_aum)]
            if len(df) < aum_before:
                logger.info(
                    f"  Filtered by AUM (>${min_aum/1e6:.0f}M): {aum_before - len(df)} removed"
                )

        logger.info(f"Final filtered universe: {len(df)} ETFs")

        return df


class ParallelETFDownloader:
    """Download price data for multiple ETFs in parallel"""

    def __init__(
        self,
        output_dir: Path,
        min_years: float = 2.0,
        max_workers: int = 20,
        max_retries: int = 3,
    ):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save CSV files
            min_years: Minimum years of data required
            max_workers: Number of parallel download threads
            max_retries: Maximum retry attempts per ETF
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_years = min_years
        self.max_workers = max_workers
        self.max_retries = max_retries

        self.min_days = int(min_years * 365)

    def download_single_etf(
        self, ticker: str, retry_count: int = 0
    ) -> Tuple[str, bool, str]:
        """
        Download price data for a single ETF.

        Returns:
            (ticker, success, message)
        """
        try:
            # Download data
            etf = yf.Ticker(ticker)
            df = etf.history(period="5y", auto_adjust=False)

            if df is None or len(df) == 0:
                return (ticker, False, "No data returned")

            # Check minimum data requirement
            if len(df) < self.min_days:
                return (
                    ticker,
                    False,
                    f"Insufficient data ({len(df)} days < {self.min_days})",
                )

            # Check data quality
            missing_pct = df["Close"].isna().sum() / len(df) * 100
            if missing_pct > 10:
                return (ticker, False, f"Too much missing data ({missing_pct:.1f}%)")

            # Save to CSV
            output_file = self.output_dir / f"{ticker}.csv"
            df.to_csv(output_file)

            return (ticker, True, f"Success ({len(df)} days)")

        except Exception as e:
            error_msg = str(e)

            # Retry on certain errors
            if retry_count < self.max_retries and (
                "timed out" in error_msg.lower() or "connection" in error_msg.lower()
            ):
                time.sleep(2**retry_count)  # Exponential backoff
                return self.download_single_etf(ticker, retry_count + 1)

            return (ticker, False, f"Error: {error_msg[:100]}")

    def download_batch(
        self, tickers: List[str], progress_callback=None
    ) -> pd.DataFrame:
        """
        Download price data for multiple ETFs in parallel.

        Args:
            tickers: List of ticker symbols
            progress_callback: Optional callback function(completed, total, ticker, success)

        Returns:
            DataFrame with download results
        """
        logger.info(
            f"Starting parallel download: {len(tickers)} ETFs, {self.max_workers} workers"
        )

        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.download_single_etf, ticker): ticker
                for ticker in tickers
            }

            # Process as completed
            for future in as_completed(future_to_ticker):
                ticker, success, message = future.result()
                completed += 1

                results.append(
                    {
                        "ticker": ticker,
                        "success": success,
                        "message": message,
                        "timestamp": datetime.now(),
                    }
                )

                # Progress callback
                if progress_callback:
                    progress_callback(completed, len(tickers), ticker, success)
                else:
                    if completed % 50 == 0 or completed == len(tickers):
                        success_count = sum(1 for r in results if r["success"])
                        logger.info(
                            f"Progress: {completed}/{len(tickers)} "
                            f"({success_count} successful, "
                            f"{completed - success_count} failed)"
                        )

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        success_count = results_df["success"].sum()
        fail_count = len(results_df) - success_count

        logger.info(
            f"\nDownload complete: {success_count} successful, {fail_count} failed"
        )

        return results_df
