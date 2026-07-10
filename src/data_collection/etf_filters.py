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

    # Additions 2026-07-10 — caught ranking as leveraged in the
    # multi-regime diagnostic. Direxion / ProShares / MicroSectors etc.
    # Full name check would catch many of these but the diagnostic
    # runs on tickers alone, so the enumeration must be complete.
    "DRN", "DRV",                   # 3x Real Estate bull/bear
    "UBT", "TBX",                   # 2x/2x 20+Y Treasury bull/inverse
    "DGP", "DZZ",                   # 2x Gold bull/bear
    "AGQ",                          # 2x Silver bull (dup, safe)
    "DPST",                         # 3x Regional Bank bull
    "TYD", "TYO",                   # 3x 7-10Y Treasury bull/bear
    "WEBL", "WEBS",                 # 3x Internet bull/bear
    "HIBL", "HIBS",                 # 3x High-Beta bull/bear
    "FNGU", "FNGD",                 # 3x FANG+ bull/bear
    "SPXL", "SPXS",                 # 3x S&P bull/bear
    "CWEB",                         # 2x China Internet bull
    "KORU",                         # 3x Korea bull
    "GUSH", "DRIP",                 # 3x Oil & Gas Exploration bull/bear
    "NAIL",                         # 3x Homebuilders bull
    "BNKU", "BNKD",                 # 3x Bank bull/bear
    "RETL",                         # 3x Retail bull
    "SLVU", "SLVO", "SLVL",         # levered silver variants
    "MSOX",                         # 2x Cannabis bull
    "MEXX",                         # 3x Mexico bull
    "EDC", "EDZ",                   # 3x Emerging Mkt bull/bear
    "AGD",                          # 2x precious metals
    "TAWK", "MJIN",                 # 2x thematic bull
    "SOXX_dup_placeholder",         # sentinel; SOXL already listed
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


EXCLUDED_CATEGORY_PREFIXES = (
    "Commodities_",       # GLD, SLV, DBC, USO, PALL, PPLT, etc.
    "Currency",           # Single-currency ETFs (UUP, FXE, FXY)
    "Volatility",         # VIXY etc. (leveraged vol already caught above)
    "Volatility_Products",
)


def get_excluded_universe_tickers() -> Set[str]:
    """Return tickers to exclude by category, per the operator design intent
    of no-leverage, no-inverse, no-commodity, no-currency, no-volatility.

    Reads categories from `comprehensive_etf_list.COMPREHENSIVE_ETF_UNIVERSE`
    and collects every ticker in a category whose name starts with any of
    `EXCLUDED_CATEGORY_PREFIXES`. Combined with `LEVERAGED_ETFS` this
    yields the full "not eligible for the smart-beta strategy" set.

    Kept as a lazy import so tests that don't need the universe don't pay
    the module-load cost.
    """
    try:
        from src.data_collection.comprehensive_etf_list import COMPREHENSIVE_ETF_UNIVERSE
    except ModuleNotFoundError:
        return set()
    out: Set[str] = set()
    for cat_name, tickers in COMPREHENSIVE_ETF_UNIVERSE.items():
        if any(cat_name.startswith(p) for p in EXCLUDED_CATEGORY_PREFIXES):
            out.update(tickers)
    return out


def get_curated_universe_tickers() -> Set[str]:
    """The hand-curated smart-beta ETF universe from
    `src.data_collection.comprehensive_etf_list.COMPREHENSIVE_ETF_UNIVERSE`,
    MINUS the excluded categories (commodities, currency, volatility).

    This is the operator's declared design intent: a defensible universe
    of factor/sector/broad-market/dividend/international/bond ETFs,
    hand-picked category by category, with leveraged/inverse/commodity/
    currency/vol screened out. About 720 tickers.
    """
    try:
        from src.data_collection.comprehensive_etf_list import COMPREHENSIVE_ETF_UNIVERSE
    except ModuleNotFoundError:
        return set()
    out: Set[str] = set()
    for cat_name, tickers in COMPREHENSIVE_ETF_UNIVERSE.items():
        if any(cat_name.startswith(p) for p in EXCLUDED_CATEGORY_PREFIXES):
            continue
        out.update(tickers)
    return out


def filter_universe(
    tickers: List[str],
    etf_names: pd.Series = None,
    use_curated: bool = True,
) -> List[str]:
    """The full smart-beta universe screen.

    Default (use_curated=True): restrict to the operator's curated
    smart-beta universe (hand-picked categories in
    `comprehensive_etf_list.py`), minus excluded categories, minus any
    leveraged/inverse tickers that slipped through category curation.
    This is what the live strategy runs on.

    Alternative (use_curated=False): apply category + leveraged filters
    to the input list but don't restrict to the curated universe. Kept
    for diagnostic purposes only; not for live use, since the
    hand-curation excludes many niche/thematic ETFs that the flat filter
    can't detect from the ticker alone (e.g. 2x/3x names not in the
    enumeration, single-country volatility bombs, ETNs).
    """
    excluded_category = get_excluded_universe_tickers()
    filtered: List[str] = []
    dropped_leveraged = 0
    dropped_category = 0
    dropped_uncurated = 0

    if use_curated:
        curated = get_curated_universe_tickers()
        for ticker in tickers:
            if ticker not in curated:
                dropped_uncurated += 1
                continue
            name = etf_names.get(ticker) if etf_names is not None else None
            if is_leveraged_etf(ticker, name):
                dropped_leveraged += 1
                continue
            if ticker.upper() in excluded_category:
                dropped_category += 1
                continue
            filtered.append(ticker)
        print(f"filter_universe (curated): kept {len(filtered)}; dropped "
              f"{dropped_uncurated} non-curated, {dropped_leveraged} leveraged, "
              f"{dropped_category} excluded-category")
    else:
        for ticker in tickers:
            name = etf_names.get(ticker) if etf_names is not None else None
            if is_leveraged_etf(ticker, name):
                dropped_leveraged += 1
                continue
            if ticker.upper() in excluded_category:
                dropped_category += 1
                continue
            filtered.append(ticker)
        print(f"filter_universe (flat): kept {len(filtered)}; dropped "
              f"{dropped_leveraged} leveraged, {dropped_category} excluded-category")
    return filtered


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
