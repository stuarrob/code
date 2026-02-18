"""
Leveraged ETF Universe

Defines tradeable leveraged ETF pairs (equity + bond legs) and provides
data loading from cached IB parquet files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class LeveragedPair:
    """A leveraged equity/bond pair for the rebalancing strategy."""

    name: str
    equity_ticker: str
    bond_ticker: str
    reference_ticker: str  # unleveraged index for momentum signal
    equity_leverage: int   # 2x or 3x
    bond_leverage: int     # 1x or 3x
    equity_split: float    # target equity weight (e.g. 0.55)
    bond_split: float      # target bond weight (e.g. 0.45)


# Standard pairs to backtest
# Note: equity_split/bond_split are legacy fields from the original
# HFEA design. Current strategy uses 100% equity (override in config).
# The bond_ticker field is retained for compatibility; set to TLT
# as a placeholder when bonds are not used.

PAIRS = {
    # --- Broad index: Nasdaq 100 ---
    "TQQQ": LeveragedPair(
        name="TQQQ", equity_ticker="TQQQ",
        bond_ticker="TLT", reference_ticker="QQQ",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    "QLD": LeveragedPair(
        name="QLD", equity_ticker="QLD",
        bond_ticker="TLT", reference_ticker="QQQ",
        equity_leverage=2, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Broad index: S&P 500 ---
    "UPRO": LeveragedPair(
        name="UPRO", equity_ticker="UPRO",
        bond_ticker="TLT", reference_ticker="SPY",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    "SPXL": LeveragedPair(
        name="SPXL", equity_ticker="SPXL",
        bond_ticker="TLT", reference_ticker="SPY",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    "SSO": LeveragedPair(
        name="SSO", equity_ticker="SSO",
        bond_ticker="TLT", reference_ticker="SPY",
        equity_leverage=2, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Broad index: Russell 2000 ---
    "TNA": LeveragedPair(
        name="TNA", equity_ticker="TNA",
        bond_ticker="TLT", reference_ticker="IWM",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    "UWM": LeveragedPair(
        name="UWM", equity_ticker="UWM",
        bond_ticker="TLT", reference_ticker="IWM",
        equity_leverage=2, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Technology ---
    "TECL": LeveragedPair(
        name="TECL", equity_ticker="TECL",
        bond_ticker="TLT", reference_ticker="XLK",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    "ROM": LeveragedPair(
        name="ROM", equity_ticker="ROM",
        bond_ticker="TLT", reference_ticker="XLK",
        equity_leverage=2, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Semiconductors ---
    "SOXL": LeveragedPair(
        name="SOXL", equity_ticker="SOXL",
        bond_ticker="TLT", reference_ticker="SMH",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Financials ---
    "FAS": LeveragedPair(
        name="FAS", equity_ticker="FAS",
        bond_ticker="TLT", reference_ticker="XLF",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    "UYG": LeveragedPair(
        name="UYG", equity_ticker="UYG",
        bond_ticker="TLT", reference_ticker="XLF",
        equity_leverage=2, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Healthcare ---
    "CURE": LeveragedPair(
        name="CURE", equity_ticker="CURE",
        bond_ticker="TLT", reference_ticker="XLV",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Biotech ---
    "LABU": LeveragedPair(
        name="LABU", equity_ticker="LABU",
        bond_ticker="TLT", reference_ticker="XBI",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Homebuilders ---
    "NAIL": LeveragedPair(
        name="NAIL", equity_ticker="NAIL",
        bond_ticker="TLT", reference_ticker="ITB",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Regional Banks ---
    "DPST": LeveragedPair(
        name="DPST", equity_ticker="DPST",
        bond_ticker="TLT", reference_ticker="KRE",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
    # --- Sector: Industrials ---
    "DUSL": LeveragedPair(
        name="DUSL", equity_ticker="DUSL",
        bond_ticker="TLT", reference_ticker="XLI",
        equity_leverage=3, bond_leverage=1,
        equity_split=0.55, bond_split=0.45,
    ),
}

# Legacy aliases for backward compatibility
PAIRS["TQQQ_TMF"] = PAIRS["TQQQ"]
PAIRS["UPRO_TMF"] = PAIRS["UPRO"]
PAIRS["SPXL_TMF"] = PAIRS["SPXL"]
PAIRS["SOXL_TMF"] = PAIRS["SOXL"]
PAIRS["QLD_TLT"] = PAIRS["QLD"]
PAIRS["SSO_TLT"] = PAIRS["SSO"]


def load_prices(
    tickers: List[str],
    data_dir: Path,
) -> pd.DataFrame:
    """Load daily close prices for a list of tickers from cached parquets.

    Args:
        tickers: List of ticker symbols.
        data_dir: Directory containing {TICKER}.parquet files.

    Returns:
        DataFrame with date index and ticker columns (close prices).
    """
    data_dir = Path(data_dir)
    series = {}

    for ticker in tickers:
        path = data_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "close" not in df.columns:
            continue
        # Handle both date-as-index and date-as-column formats
        if "date" in df.columns:
            s = df.set_index("date")["close"]
        else:
            s = df["close"]  # date is already the index
        s.index = pd.to_datetime(s.index)
        s.name = ticker
        series[ticker] = s

    if not series:
        return pd.DataFrame()

    prices = pd.DataFrame(series)
    prices = prices.sort_index()
    return prices


def load_pair_data(
    pair: LeveragedPair,
    data_dir: Path,
) -> Optional[pd.DataFrame]:
    """Load all price data needed for a single pair.

    Returns DataFrame with columns: equity_ticker, bond_ticker, reference_ticker.
    Returns None if any required ticker is missing.
    """
    tickers = [pair.equity_ticker, pair.bond_ticker, pair.reference_ticker]
    # Deduplicate (reference might already be in the list)
    tickers = list(dict.fromkeys(tickers))

    prices = load_prices(tickers, data_dir)

    missing = [t for t in [pair.equity_ticker, pair.bond_ticker, pair.reference_ticker]
               if t not in prices.columns]
    if missing:
        return None

    return prices


def check_data_availability(
    data_dir: Path,
    pairs: Dict[str, LeveragedPair] = None,
) -> pd.DataFrame:
    """Check which tickers are cached and which need collecting.

    Returns summary DataFrame with ticker, cached (bool), rows, date range.
    """
    if pairs is None:
        pairs = PAIRS

    data_dir = Path(data_dir)
    all_tickers = set()
    for pair in pairs.values():
        all_tickers.update([pair.equity_ticker, pair.bond_ticker, pair.reference_ticker])

    results = []
    for ticker in sorted(all_tickers):
        path = data_dir / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            idx = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
            results.append({
                "ticker": ticker,
                "cached": True,
                "rows": len(df),
                "start": idx.min(),
                "end": idx.max(),
            })
        else:
            results.append({
                "ticker": ticker,
                "cached": False,
                "rows": 0,
                "start": None,
                "end": None,
            })

    return pd.DataFrame(results)
