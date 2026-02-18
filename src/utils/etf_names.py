"""
ETF Name Lookup Utility

Resolves ticker symbols to human-readable ETF names.
Uses a built-in cache of common ETFs, supplemented by yfinance lookups
for unknown tickers.  Results are cached to disk to avoid repeated API calls.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Cache file for yfinance lookups
_CACHE_DIR = Path.home() / "trade_data" / "ETFTrader" / "processed"
_CACHE_FILE = _CACHE_DIR / "etf_name_cache.json"

# Built-in names for the most common ETFs (avoids any network call)
_BUILTIN_NAMES: Dict[str, str] = {
    # US Broad Market
    "SPY": "SPDR S&P 500 ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust (Nasdaq 100)",
    "VTI": "Vanguard Total Stock Market ETF",
    "IWM": "iShares Russell 2000 Small-Cap ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    # US Sector
    "XLK": "Technology Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR",
    "XLE": "Energy Select Sector SPDR",
    "XLV": "Health Care Select Sector SPDR",
    "XLI": "Industrial Select Sector SPDR",
    "XLY": "Consumer Discretionary Select Sector SPDR",
    "XLP": "Consumer Staples Select Sector SPDR",
    "XLU": "Utilities Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR",
    # Bonds
    "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core US Aggregate Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "LQD": "iShares Investment Grade Corporate Bond ETF",
    "HYG": "iShares High Yield Corporate Bond ETF",
    "EMLC": "VanEck EM Local Currency Bond ETF",
    # International Developed
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "IDEV": "iShares Core MSCI Intl Developed Markets ETF",
    "VXUS": "Vanguard Total International Stock ETF",
    "VSGX": "Vanguard ESG International Stock ETF",
    "VSS": "Vanguard FTSE All-World ex-US Small-Cap ETF",
    # International Country
    "EWU": "iShares MSCI United Kingdom ETF",
    "EWG": "iShares MSCI Germany ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "EWK": "iShares MSCI Belgium ETF",
    "EWO": "iShares MSCI Austria ETF",
    "EWP": "iShares MSCI Spain ETF",
    "EWQ": "iShares MSCI France ETF",
    "EWL": "iShares MSCI Switzerland ETF",
    "EWZ": "iShares MSCI Brazil ETF",
    "EZU": "iShares MSCI Eurozone ETF",
    "FEZ": "SPDR EURO STOXX 50 ETF",
    "IEV": "iShares Europe ETF",
    # International Dividend / Income
    "IDV": "iShares International Select Dividend ETF",
    "EFAS": "Global X MSCI SuperDividend EAFE ETF",
    "FGD": "First Trust Dow Jones Global Select Dividend ETF",
    "FID": "First Trust S&P Intl Dividend Aristocrats ETF",
    "DTH": "WisdomTree International High Dividend Fund",
    "DIM": "WisdomTree International MidCap Dividend Fund",
    "DLS": "WisdomTree International SmallCap Dividend Fund",
    "DWX": "SPDR S&P International Dividend ETF",
    "DVYE": "iShares Emerging Markets Dividend ETF",
    "VYMI": "Vanguard Intl High Dividend Yield ETF",
    "WDIV": "SPDR S&P Global Dividend ETF",
    "IQDY": "FlexShares Intl Quality Dividend Dynamic ETF",
    "FIDI": "Fidelity International High Dividend ETF",
    "FYLD": "Cambria Foreign Shareholder Yield ETF",
    "FNDC": "Schwab Fundamental Intl Small Equity ETF",
    "ECOW": "Pacer Emerging Markets Cash Cows 100 ETF",
    # International Multi-Factor / Smart Beta
    "RODM": "Hartford Multifactor Developed Markets (ex-US) ETF",
    "ROAM": "Hartford Multifactor Emerging Markets ETF",
    "INEQ": "Columbia International Equity Income ETF",
    "VIDI": "Vident International Equity Strategy ETF",
    "UIVM": "VictoryShares Intl Value Momentum ETF",
    "ISVL": "iShares Intl Developed Small Cap Value Factor ETF",
    "HEDJ": "WisdomTree Europe Hedged Equity Fund",
    "DBEU": "Xtrackers MSCI Europe Hedged Equity ETF",
    # Commodities / Alternatives
    "GLD": "SPDR Gold Shares",
    "IAU": "iShares Gold Trust",
    "SLV": "iShares Silver Trust",
    "GLDI": "UBS ETRACS Gold Shares Covered Call ETN",
    # Real Estate
    "VNQ": "Vanguard Real Estate ETF",
    "RWX": "SPDR Dow Jones International Real Estate ETF",
    # Leveraged (for reference)
    "TQQQ": "ProShares UltraPro QQQ (3x Nasdaq)",
    "UPRO": "ProShares UltraPro S&P 500 (3x S&P)",
    "SOXL": "Direxion Daily Semiconductor Bull 3x",
    "TMF": "Direxion Daily 20+ Year Treasury Bull 3x",
    "SPXL": "Direxion Daily S&P 500 Bull 3x",
    "TECL": "Direxion Daily Technology Bull 3x",
}


def _load_cache() -> Dict[str, str]:
    """Load cached ETF names from disk."""
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: Dict[str, str]) -> None:
    """Save ETF name cache to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def lookup_names(
    tickers: List[str],
    use_yfinance: bool = True,
) -> Dict[str, str]:
    """Look up human-readable names for a list of tickers.

    Checks built-in names first, then disk cache, then yfinance.

    Args:
        tickers: List of ticker symbols.
        use_yfinance: Whether to query yfinance for unknown tickers.

    Returns:
        Dict mapping ticker -> full name.
    """
    result: Dict[str, str] = {}
    unknown: List[str] = []
    cache = _load_cache()

    for t in tickers:
        t_upper = t.upper()
        if t_upper in _BUILTIN_NAMES:
            result[t] = _BUILTIN_NAMES[t_upper]
        elif t in cache:
            result[t] = cache[t]
        else:
            unknown.append(t)

    if unknown and use_yfinance:
        try:
            import yfinance as yf
            for t in unknown:
                try:
                    info = yf.Ticker(t).info
                    name = info.get("longName") or info.get("shortName", t)
                    result[t] = name
                    cache[t] = name
                except Exception:
                    result[t] = t
                    cache[t] = t
            _save_cache(cache)
        except ImportError:
            for t in unknown:
                result[t] = t
    else:
        for t in unknown:
            result[t] = t

    return result


def format_holdings_table(
    weights: pd.Series,
    factor_scores: Optional[pd.DataFrame] = None,
    use_yfinance: bool = True,
) -> pd.DataFrame:
    """Format portfolio holdings as a human-readable table.

    Args:
        weights: Series of ticker -> weight.
        factor_scores: Optional DataFrame with factor scores per ticker.
        use_yfinance: Whether to look up unknown names via yfinance.

    Returns:
        DataFrame with columns: Ticker, Name, Weight, and factor scores.
    """
    tickers = weights.index.tolist()
    names = lookup_names(tickers, use_yfinance=use_yfinance)

    rows = []
    for t in tickers:
        row = {
            "Ticker": t,
            "Name": names.get(t, t),
            "Weight": f"{weights[t]:.1%}",
        }
        if factor_scores is not None and t in factor_scores.index:
            for col in factor_scores.columns:
                val = factor_scores.loc[t, col]
                if pd.notna(val):
                    # Show percentile rank
                    pct = factor_scores[col].rank(pct=True).get(t, float("nan"))
                    row[col.title()] = f"{pct:.0%}" if pd.notna(pct) else "—"
                else:
                    row[col.title()] = "—"
        rows.append(row)

    return pd.DataFrame(rows)
