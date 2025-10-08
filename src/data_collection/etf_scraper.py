"""
ETF Universe Scraper
Collects ETF ticker symbols from multiple sources and filters out leveraged ETFs
"""

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict
import re
from datetime import datetime


class ETFScraper:
    """Scrape and collect ETF universe from multiple sources"""

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
        ]
        self.etf_list = []

    def is_leveraged(self, name: str, ticker: str) -> bool:
        """Check if ETF is leveraged based on name or ticker"""
        name_lower = name.lower()
        ticker_lower = ticker.lower()

        # Check for leveraged keywords in name
        for keyword in self.leveraged_keywords:
            if keyword in name_lower:
                return True

        # Check for common leveraged ETF ticker patterns
        leveraged_patterns = [
            r"[US][QSPTD]{2,3}[XYZ]",  # TQQQ, SQQQ, SPXL, etc.
            r"[A-Z]{3}[LUS]",  # TECL, SOXL, etc.
        ]

        for pattern in leveraged_patterns:
            if re.match(pattern, ticker.upper()):
                return True

        return False

    def scrape_etfdb_list(self) -> List[Dict]:
        """Scrape ETF list from ETF Database using yfinance screen"""
        print("Collecting ETFs from yfinance screener...")

        # Use yfinance to get list of ETFs
        # We'll get major ETFs and expand from there
        major_etfs = [
            "SPY",
            "IVV",
            "VOO",
            "QQQ",
            "VTI",
            "VEA",
            "IEFA",
            "AGG",
            "BND",
            "VWO",
            "VTV",
            "VUG",
            "IWF",
            "IWM",
            "IJH",
            "VIG",
            "SCHD",
            "VNQ",
            "VNQI",
            "GLD",
            "SLV",
            "DIA",
            "EEM",
            "EFA",
            "HYG",
            "LQD",
            "TLT",
            "IEF",
            "SHY",
            "MUB",
            "VGT",
            "XLK",
            "XLF",
            "XLE",
            "XLV",
            "XLY",
            "XLP",
            "XLI",
            "XLB",
            "XLU",
            "VB",
            "VBR",
            "VBK",
            "VO",
            "VTWO",
            "VXUS",
            "BNDX",
            "VT",
            "VCIT",
            "VCSH",
            "IEMG",
            "IXUS",
            "SCHX",
            "SCHF",
            "SCHA",
            "SCHB",
            "SCHE",
            "SCHG",
            "SCHV",
            "VYM",
            "DVY",
            "HDV",
            "DGRO",
            "NOBL",
            "SDY",
            "VV",
            "MGC",
            "MGK",
            "MGV",
            "IVW",
            "IVE",
            "IJJ",
            "IJK",
            "IJR",
            "IJS",
            "VCR",
            "VDC",
            "VDE",
            "VFH",
            "VGT",
            "VHT",
            "VIS",
            "VAW",
            "VPU",
            "VOX",
            "RSP",
            "QUAL",
            "MTUM",
            "USMV",
            "SIZE",
            "VLUE",
            "VBR",
            "VONG",
            "VONV",
            "VTHR",
            "VTWG",
            "VTWV",
            "ESGV",
            "ESGU",
            "EAGG",
            "SUSL",
            "USSG",
            "DSI",
            "EEMV",
            "EFAV",
            "SPHQ",
            "SPLV",
            "ACWI",
            "ACWV",
            "ACWX",
            "AMLP",
            "ARKK",
            "ARKG",
            "ARKW",
            "ARKF",
            "ARKQ",
            "BIL",
            "BSV",
            "BSCR",
            "CWB",
            "DBC",
            "DEM",
            "DES",
            "DFE",
            "DGS",
            "DHS",
            "DIA",
            "DLS",
            "DON",
            "DSTL",
            "EBND",
            "EMLC",
            "EMB",
            "EMGF",
            "EWA",
            "EWC",
            "EWG",
            "EWH",
            "EWJ",
            "EWL",
            "EWQ",
            "EWT",
            "EWU",
            "EWY",
            "EWZ",
            "EZA",
            "EZU",
            "FDN",
            "FEZ",
            "FM",
            "FPE",
            "FXI",
            "FXE",
            "GOVERNMENT",
            "GSG",
            "GUNR",
            "GXC",
            "HEDJ",
            "HEWJ",
            "IAU",
            "IBB",
            "ICLN",
            "ICF",
            "IDV",
            "IGIB",
            "IGSB",
            "IHDG",
            "IMTB",
            "ITOT",
            "IWB",
            "IWD",
            "IWO",
            "IWP",
            "IWR",
            "IWS",
            "IWV",
            "IXC",
            "IXG",
            "IXJ",
            "IXN",
            "IYE",
            "IYF",
            "IYH",
            "IYR",
            "IYW",
            "JNK",
            "KBE",
            "KRE",
            "MDY",
            "MGV",
            "MJ",
            "MLPA",
            "MOAT",
            "MSOS",
            "OEF",
            "PFF",
            "PHO",
            "PIN",
            "PNQI",
            "PWV",
            "QLD",
            "QQQM",
            "QTEC",
            "RPG",
            "RPV",
            "RSX",
            "RWO",
            "RWR",
            "RWX",
            "SCHC",
            "SCHD",
            "SCHE",
            "SCHF",
            "SCHG",
            "SCHH",
            "SCHI",
            "SCHK",
            "SCHM",
            "SCHO",
            "SCHP",
            "SCHR",
            "SCHV",
            "SCHX",
            "SCHZ",
            "SHM",
            "SHV",
            "SHYG",
            "SMH",
            "SOXX",
            "SPAB",
            "SPBO",
            "SPDW",
            "SPEM",
            "SPHB",
            "SPHD",
            "SPIB",
            "SPLG",
            "SPLV",
            "SPMB",
            "SPMD",
            "SPSB",
            "SPSM",
            "SPTL",
            "SPTM",
            "SPTI",
            "SPTS",
            "SUSA",
            "TAN",
            "TFI",
            "TIP",
            "TLH",
            "TOK",
            "TOTL",
            "TQQQ",
            "TUR",
            "URTH",
            "USIG",
            "USRT",
            "USWM",
            "VCR",
            "VCIT",
            "VCLT",
            "VDC",
            "VDE",
            "VEU",
            "VFH",
            "VGIT",
            "VGLT",
            "VGSH",
            "VGT",
            "VHT",
            "VIG",
            "VIXY",
            "VNQ",
            "VNQI",
            "VO",
            "VOE",
            "VONE",
            "VONG",
            "VONV",
            "VOO",
            "VOT",
            "VOX",
            "VPL",
            "VPU",
            "VRP",
            "VT",
            "VTEB",
            "VTES",
            "VTI",
            "VTIP",
            "VTV",
            "VTWG",
            "VTWO",
            "VTWV",
            "VUG",
            "VV",
            "VWO",
            "VWOB",
            "VXF",
            "VXUS",
            "VYM",
            "WOOD",
            "XBI",
            "XES",
            "XHB",
            "XLB",
            "XLC",
            "XLE",
            "XLF",
            "XLI",
            "XLK",
            "XLP",
            "XLRE",
            "XLU",
            "XLV",
            "XLY",
            "XME",
            "XOP",
            "XRT",
            "XSD",
            "XTN",
        ]

        etfs = []
        for ticker in major_etfs:
            etfs.append({"ticker": ticker, "source": "curated"})

        return etfs

    def expand_etf_list(self, seed_etfs: List[str], max_etfs: int = 500) -> List[Dict]:
        """Expand ETF list by looking at similar ETFs and sector ETFs"""
        print(f"Expanding ETF list to discover more tickers...")

        all_etfs = set(seed_etfs)

        # Common ETF families/issuers
        etf_families = {
            "Vanguard": [
                "V",
                "VO",
                "VB",
                "VT",
                "VI",
                "VX",
                "VA",
                "VC",
                "VD",
                "VE",
                "VG",
                "VH",
                "VM",
                "VN",
                "VP",
                "VR",
                "VS",
                "VU",
                "VW",
                "VY",
            ],
            "iShares": [
                "I",
                "IV",
                "IW",
                "IE",
                "IX",
                "IY",
                "IJ",
                "ID",
                "IG",
                "IH",
                "IL",
                "IM",
                "IQ",
            ],
            "SPDR": [
                "SP",
                "XL",
                "XM",
                "XE",
                "XS",
                "XA",
                "XC",
                "XD",
                "XF",
                "XH",
                "XI",
                "XN",
                "XP",
                "XR",
                "XT",
            ],
            "Schwab": ["SCH", "SN"],
            "Invesco": [
                "QQ",
                "PB",
                "PC",
                "PD",
                "PE",
                "PF",
                "PG",
                "PH",
                "PI",
                "PJ",
                "PK",
                "PM",
                "PN",
                "PO",
                "PP",
                "PR",
                "PS",
                "PT",
                "PU",
                "PV",
                "PW",
                "PX",
                "PY",
                "PZ",
            ],
        }

        # Generate potential tickers
        potential_tickers = []
        for family, prefixes in etf_families.items():
            for prefix in prefixes:
                for suffix in [
                    "",
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "N",
                    "O",
                    "P",
                    "Q",
                    "R",
                    "S",
                    "T",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]:
                    ticker = prefix + suffix
                    if len(ticker) >= 3 and len(ticker) <= 5:
                        potential_tickers.append(ticker)

        # Test tickers in batches
        print(f"Testing {len(potential_tickers)} potential ETF tickers...")
        valid_etfs = []

        for i in range(0, min(len(potential_tickers), max_etfs * 2), 50):
            batch = potential_tickers[i : i + 50]
            for ticker in batch:
                if len(valid_etfs) >= max_etfs:
                    break

                try:
                    # Quick check if ticker exists
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    # Check if it's an ETF
                    if info.get("quoteType") == "ETF":
                        valid_etfs.append({"ticker": ticker, "source": "discovered"})
                        all_etfs.add(ticker)
                        print(f"  Found ETF: {ticker}")

                except Exception:
                    pass

                time.sleep(0.1)  # Rate limiting

            if len(valid_etfs) >= max_etfs:
                break

        return [{"ticker": t, "source": "curated"} for t in seed_etfs] + valid_etfs

    def get_etf_info(self, ticker: str) -> Dict:
        """Get detailed information for a single ETF"""
        try:
            etf = yf.Ticker(ticker)
            info = etf.info

            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "category": info.get("category", "Unknown"),
                "expense_ratio": (
                    info.get("annualReportExpenseRatio", info.get("totalAssets", 0))
                    / 100
                    if info.get("annualReportExpenseRatio")
                    else None
                ),
                "aum": info.get("totalAssets", None),
                "inception_date": info.get("fundInceptionDate", None),
                "exchange": info.get("exchange", None),
                "currency": info.get("currency", "USD"),
            }
        except Exception as e:
            print(f"  Error getting info for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "name": ticker,
                "category": "Unknown",
                "expense_ratio": None,
                "aum": None,
                "inception_date": None,
                "exchange": None,
                "currency": "USD",
            }

    def scrape_full_universe(self, max_etfs: int = 300) -> pd.DataFrame:
        """Main method to scrape full ETF universe"""
        print("=" * 60)
        print("Starting ETF Universe Collection")
        print("=" * 60)

        # Step 1: Get seed list
        print("\nStep 1: Getting seed ETF list...")
        seed_etfs = self.scrape_etfdb_list()
        print(f"  Collected {len(seed_etfs)} seed ETFs")

        # Step 2: Expand list (optional, can be slow)
        # Uncomment to discover more ETFs
        # print("\nStep 2: Expanding ETF list...")
        # all_etfs = self.expand_etf_list([e['ticker'] for e in seed_etfs], max_etfs)
        all_etfs = seed_etfs

        # Step 3: Get detailed info for each ETF
        print(f"\nStep 2: Collecting detailed information for {len(all_etfs)} ETFs...")
        etf_data = []

        for i, etf_dict in enumerate(all_etfs):
            ticker = etf_dict["ticker"]
            print(f"  [{i+1}/{len(all_etfs)}] Processing {ticker}...")

            info = self.get_etf_info(ticker)

            # Check if leveraged
            is_lev = self.is_leveraged(info["name"], ticker)
            info["is_leveraged"] = is_lev

            if is_lev:
                print(f"    ⚠️  LEVERAGED - will be filtered out")

            etf_data.append(info)

            # Rate limiting
            time.sleep(0.2)

        # Step 4: Create DataFrame and filter
        df = pd.DataFrame(etf_data)

        # Filter out leveraged ETFs
        print(f"\nStep 3: Filtering out leveraged ETFs...")
        print(f"  Before filtering: {len(df)} ETFs")
        df_filtered = df[df["is_leveraged"] == False].copy()
        print(f"  After filtering: {len(df_filtered)} ETFs")
        print(f"  Removed: {len(df) - len(df_filtered)} leveraged ETFs")

        # Add metadata
        df_filtered["data_collection_date"] = datetime.now().strftime("%Y-%m-%d")

        print("\n" + "=" * 60)
        print("ETF Universe Collection Complete!")
        print("=" * 60)
        print(f"\nFinal ETF count: {len(df_filtered)}")
        print(f"\nCategories distribution:")
        print(df_filtered["category"].value_counts().head(10))

        return df_filtered


def main():
    """Main execution function"""
    scraper = ETFScraper()

    # Scrape ETF universe (start with ~200-300 ETFs)
    etf_universe = scraper.scrape_full_universe(max_etfs=300)

    # Save to CSV
    output_path = "data/raw/etf_universe.csv"
    etf_universe.to_csv(output_path, index=False)
    print(f"\n✅ Saved ETF universe to: {output_path}")
    print(f"   Total ETFs: {len(etf_universe)}")

    # Display sample
    print("\nSample of collected ETFs:")
    print(etf_universe[["ticker", "name", "category", "expense_ratio", "aum"]].head(10))

    return etf_universe


if __name__ == "__main__":
    main()
