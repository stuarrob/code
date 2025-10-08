"""
ETF Filtering Utilities

Filters for excluding specific ETF types:
- Leveraged ETFs (2x, 3x)
- Inverse ETFs
- Other specialized products
"""

import re
from typing import List, Set
import pandas as pd
import numpy as np


# Known leveraged ETF tickers (2x, 3x multipliers)
LEVERAGED_ETFS = {
    # 2x Bull
    "UGL", "AGQ", "UCO", "UYG", "URE", "UCC", "UYM", "UGE", "UPW", "UJB",
    "SAA", "SSO", "DDM", "MVV", "UKK", "UPV", "UXI", "UKF", "UJO",

    # 3x Bull
    "TECL", "TQQQ", "SOXL", "UPRO", "UDOW", "URTY", "FAS", "TNA", "CURE",
    "WANT", "BULZ", "PILL", "DUST", "NUGT", "JNUG", "MIDU", "ERX", "TMF",
    "TYD", "UGAZ", "BOIL", "LABU", "YINN",

    # 2x Bear
    "SKK", "SZK", "SMN", "SDD", "SDK", "SDP", "SRS", "TWM", "SBB", "DXD",
    "MZZ", "PST", "TBT", "TBF", "SSG",

    # 3x Bear
    "SQQQ", "SPXU", "SDOW", "SRTY", "FAZ", "TZA", "SOXS", "TMV", "ERY",
    "DGAZ", "KOLD", "SPXS", "YANG", "LABD", "DRV", "EDZ",

    # Other leveraged
    "UVXY", "SVXY", "VXX", "VIXY", "TVIX",  # Volatility products
}


# Patterns for identifying leveraged ETFs by name
LEVERAGED_PATTERNS = [
    r'.*\s+2[xX]',           # "2x" or "2X"
    r'.*\s+3[xX]',           # "3x" or "3X"
    r'.*[Tt]riple',          # "Triple"
    r'.*[Dd]ouble',          # "Double"
    r'.*[Uu]ltra\s*[Pp]ro',  # "UltraPro"
    r'.*[Uu]ltra',           # "Ultra"
    r'.*[Ll]everaged',       # "Leveraged"
    r'.*-2[xX]',             # "-2x"
    r'.*-3[xX]',             # "-3x"
]


def is_leveraged_etf(ticker: str, name: str = None) -> bool:
    """
    Check if ETF is leveraged (2x, 3x).

    Parameters
    ----------
    ticker : str
        ETF ticker symbol
    name : str, optional
        ETF name for pattern matching

    Returns
    -------
    bool
        True if leveraged, False otherwise
    """
    # Check known leveraged tickers
    if ticker.upper() in LEVERAGED_ETFS:
        return True

    # Check name patterns if provided
    if name:
        for pattern in LEVERAGED_PATTERNS:
            if re.match(pattern, name, re.IGNORECASE):
                return True

    return False


def filter_leveraged_etfs(
    tickers: List[str],
    etf_names: pd.Series = None
) -> List[str]:
    """
    Filter out leveraged ETFs from ticker list.

    Parameters
    ----------
    tickers : list
        List of ETF ticker symbols
    etf_names : pd.Series, optional
        Series mapping tickers to ETF names

    Returns
    -------
    list
        Filtered list without leveraged ETFs
    """
    filtered = []
    excluded = []

    for ticker in tickers:
        name = etf_names.get(ticker) if etf_names is not None else None
        if not is_leveraged_etf(ticker, name):
            filtered.append(ticker)
        else:
            excluded.append(ticker)

    if excluded:
        print(f"Filtered out {len(excluded)} leveraged ETFs: {', '.join(sorted(excluded)[:10])}"
              f"{f' and {len(excluded)-10} more' if len(excluded) > 10 else ''}")

    return filtered


def get_leveraged_etfs_from_universe(universe_file: str) -> Set[str]:
    """
    Identify leveraged ETFs in universe file.

    Parameters
    ----------
    universe_file : str
        Path to ETF universe CSV

    Returns
    -------
    set
        Set of leveraged ETF tickers
    """
    try:
        df = pd.read_csv(universe_file)

        # Check if we have a name column
        name_col = None
        for col in ['name', 'Name', 'etf_name', 'description']:
            if col in df.columns:
                name_col = col
                break

        ticker_col = 'ticker' if 'ticker' in df.columns else df.columns[0]

        leveraged = set()
        for idx, row in df.iterrows():
            ticker = row[ticker_col]
            name = row[name_col] if name_col else None
            if is_leveraged_etf(ticker, name):
                leveraged.add(ticker)

        return leveraged

    except Exception as e:
        print(f"Error reading universe file: {e}")
        return set()


def filter_by_volatility(
    prices: pd.DataFrame,
    max_volatility: float = 0.35,
    min_periods: int = 252
) -> List[str]:
    """
    Filter ETFs by maximum annualized volatility.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data (rows=dates, columns=tickers)
    max_volatility : float
        Maximum annualized volatility (default 0.35 = 35%)
    min_periods : int
        Minimum periods required for volatility calculation (default 252 = 1 year)

    Returns
    -------
    list
        Tickers that meet volatility criteria
    """
    filtered = []
    excluded = []

    for ticker in prices.columns:
        try:
            returns = prices[ticker].pct_change().dropna()

            if len(returns) < min_periods:
                excluded.append((ticker, "insufficient data"))
                continue

            # Calculate annualized volatility
            vol = returns.std() * np.sqrt(252)

            if vol <= max_volatility:
                filtered.append(ticker)
            else:
                excluded.append((ticker, f"vol {vol*100:.1f}%"))

        except Exception as e:
            excluded.append((ticker, f"error: {e}"))
            continue

    if excluded:
        print(f"\nFiltered out {len(excluded)} high-volatility ETFs (max {max_volatility*100:.0f}%):")
        # Show first 20 with their volatilities
        for ticker, reason in excluded[:20]:
            print(f"  {ticker}: {reason}")
        if len(excluded) > 20:
            print(f"  ... and {len(excluded)-20} more")

    return filtered


def apply_etf_filters(
    prices: pd.DataFrame,
    filter_leveraged: bool = True,
    filter_high_volatility: bool = True,
    max_volatility: float = 0.35,
    etf_names: pd.Series = None
) -> pd.DataFrame:
    """
    Apply all ETF filters with configurable options.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data (rows=dates, columns=tickers)
    filter_leveraged : bool
        Whether to filter out leveraged ETFs (default True)
    filter_high_volatility : bool
        Whether to filter by volatility (default True)
    max_volatility : float
        Maximum annualized volatility if filtering (default 0.35 = 35%)
    etf_names : pd.Series, optional
        Series mapping tickers to ETF names for leveraged detection

    Returns
    -------
    pd.DataFrame
        Filtered price data
    """
    tickers = prices.columns.tolist()
    original_count = len(tickers)

    # Step 1: Filter leveraged ETFs
    if filter_leveraged:
        tickers = filter_leveraged_etfs(tickers, etf_names)
        print(f"After leveraged filter: {len(tickers)}/{original_count} ETFs remaining")
    else:
        print(f"Leveraged filter: DISABLED (keeping all {original_count} ETFs)")

    # Step 2: Filter by volatility
    if filter_high_volatility:
        # Calculate on available tickers
        prices_subset = prices[tickers]
        tickers = filter_by_volatility(prices_subset, max_volatility)
        print(f"After volatility filter (max {max_volatility*100:.0f}%): {len(tickers)}/{original_count} ETFs remaining")
    else:
        print(f"Volatility filter: DISABLED")

    return prices[tickers]
