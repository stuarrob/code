"""Enrich trade tickers with human-readable metadata.

Used by the applet's Step 4 composition panel and the narrator to tell
the operator WHAT they're buying (fund name), WHERE (geography /
asset-class category), and WHY (dominant factor).

Sources:
- `comprehensive_etf_list.COMPREHENSIVE_ETF_UNIVERSE` — hand-curated
  category per ticker (US_Broad_Large, Intl_Developed, Sector_Technology,
  Bonds_Aggregate, etc.). Fast, offline lookup.
- FMP `stable/profile` endpoint — official fund name (companyName).
  On-demand lookup for the small handful of tickers in a rebalance
  proposal (~20-30 buys). Cached to a small parquet so subsequent
  runs are instant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

try:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)


NAME_CACHE_PATH = (
    Path.home() / "trade_data" / "ETFTrader" / "processed" / "etf_names.parquet"
)


@dataclass(frozen=True)
class TickerMetadata:
    ticker: str
    name: Optional[str]              # e.g. "Vanguard S&P 500 ETF"
    category: Optional[str]          # e.g. "US_Broad_Large"
    geography: Optional[str]         # derived from category prefix
    asset_class: Optional[str]       # derived from category prefix
    dominant_factor: Optional[str]   # e.g. "momentum"
    dominant_factor_score: Optional[float]


# ────────────────────────────────────────────────────────────────
# Category → geography / asset-class mapping
# ────────────────────────────────────────────────────────────────

_GEOGRAPHY_MAP = {
    "US": "United States",
    "Intl_Broad": "International Developed", "Intl_Asia": "International Developed",
    "Intl_Europe": "International Developed", "Intl_Developed": "International Developed",
    "Intl_Single": "International Single-Country",
    "Emerging_Asia": "Emerging Markets", "Emerging_EMEA": "Emerging Markets",
    "Emerging_Latin": "Emerging Markets", "Emerging_Broad": "Emerging Markets",
    "Emerging_Small": "Emerging Markets",
    "Bonds": "Global (bonds)",
    "Sector": "United States (sector tilt)",
    "Dividend_Intl": "International Developed",
    "Dividend": "United States",
    "Factor": "United States (factor tilt)",
    "ESG": "United States (ESG)",
    "Thematic": "Thematic / niche",
    "MultiAsset": "Multi-asset",
    "Alternative": "Alternative strategies",
    "Target": "Target-date",
}


def _geography_from_category(category: str) -> str:
    """Map a curated category to a coarse geography bucket."""
    if not category:
        return "Unknown"
    for prefix, label in _GEOGRAPHY_MAP.items():
        if category.startswith(prefix):
            return label
    return category.split("_", 1)[0]


def _asset_class_from_category(category: str) -> str:
    if not category:
        return "Unknown"
    if category.startswith("Bonds"):
        return "Bonds"
    if category.startswith("Sector"):
        return "Equity — sector"
    if category.startswith("Dividend"):
        return "Equity — dividend"
    if category.startswith("Factor"):
        return "Equity — factor"
    if category.startswith("Thematic") or category.startswith("ESG"):
        return "Equity — thematic"
    if category.startswith(("US_", "Intl_", "Emerging_")):
        return "Equity — broad"
    if category.startswith("MultiAsset"):
        return "Multi-asset"
    if category.startswith("Alternative"):
        return "Alternative"
    return "Other"


# ────────────────────────────────────────────────────────────────
# Category lookup — offline, from the curated list
# ────────────────────────────────────────────────────────────────

def build_category_map() -> dict[str, str]:
    """Return {ticker → category} for every ticker in the curated universe.

    A ticker in multiple categories takes the first one (rare).
    """
    from src.data_collection.comprehensive_etf_list import COMPREHENSIVE_ETF_UNIVERSE
    out: dict[str, str] = {}
    for category, tickers in COMPREHENSIVE_ETF_UNIVERSE.items():
        for t in tickers:
            out.setdefault(t.upper(), category)
    return out


# ────────────────────────────────────────────────────────────────
# Fund-name cache (FMP profile companyName)
# ────────────────────────────────────────────────────────────────

def _load_name_cache(path: Path = NAME_CACHE_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        return {r["ticker"]: r["name"] for _, r in df.iterrows()}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"name cache read failed: {exc}")
        return {}


def _save_name_cache(names: dict[str, str], path: Path = NAME_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [{"ticker": t, "name": n} for t, n in sorted(names.items())]
    )
    df.to_parquet(path, index=False)


def _fetch_name_from_fmp(ticker: str, api_key: str,
                         timeout_sec: float = 10.0) -> Optional[str]:
    """Fetch companyName from FMP profile. Single-ticker call."""
    try:
        r = requests.get(
            "https://financialmodelingprep.com/stable/profile",
            params={"symbol": ticker, "apikey": api_key},
            timeout=timeout_sec,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except ValueError:
        return None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0].get("companyName")
    return None


def get_names(tickers: Iterable[str],
              api_key: Optional[str] = None) -> dict[str, str]:
    """Return {ticker → fund name} for the requested tickers.

    Reads from the local cache when possible; fetches any missing names
    from FMP profile and updates the cache. Silent no-op for tickers
    with no FMP coverage or if the API key is absent.
    """
    tickers = [t.upper() for t in tickers]
    cache = _load_name_cache()
    missing = [t for t in tickers if t not in cache]

    if missing:
        key = api_key or os.environ.get("FMP_API_KEY")
        if key:
            for t in missing:
                name = _fetch_name_from_fmp(t, key)
                if name:
                    cache[t] = name
            _save_name_cache(cache)

    return {t: cache.get(t) for t in tickers}


# ────────────────────────────────────────────────────────────────
# Composite entry point
# ────────────────────────────────────────────────────────────────

def enrich_tickers(
    tickers: Iterable[str],
    factor_scores: Optional[pd.DataFrame] = None,
    fetch_names: bool = True,
) -> dict[str, TickerMetadata]:
    """Return `{ticker → TickerMetadata}` for each input ticker.

    Args:
        tickers: iterable of ticker symbols.
        factor_scores: optional DataFrame indexed by ticker with factor
            columns. When present, `dominant_factor` is set to the
            factor with the highest score for each ticker.
        fetch_names: when True, fill in fund names from FMP (cached).

    Returns dict — one entry per input ticker, even if metadata is
    partially missing (fields default to None).
    """
    tickers = [t.upper() for t in tickers]
    category_map = build_category_map()
    names = get_names(tickers) if fetch_names else {t: None for t in tickers}

    dominant_factor: dict[str, tuple[Optional[str], Optional[float]]] = {}
    if factor_scores is not None and not factor_scores.empty:
        for t in tickers:
            if t in factor_scores.index:
                row = factor_scores.loc[t].dropna()
                if not row.empty:
                    top = row.idxmax()
                    dominant_factor[t] = (top, float(row[top]))
                else:
                    dominant_factor[t] = (None, None)
            else:
                dominant_factor[t] = (None, None)

    out: dict[str, TickerMetadata] = {}
    for t in tickers:
        category = category_map.get(t)
        top, score = dominant_factor.get(t, (None, None))
        out[t] = TickerMetadata(
            ticker=t,
            name=names.get(t),
            category=category,
            geography=_geography_from_category(category) if category else None,
            asset_class=_asset_class_from_category(category) if category else None,
            dominant_factor=top,
            dominant_factor_score=score,
        )
    return out
