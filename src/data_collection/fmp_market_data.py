"""FMP market-index data (VIX, and any future index series).

Kept separate from `issuer_fundamentals.py` because indices are not
fundamentals and the pull/cache cadence is different (VIX changes
daily; fundamentals change monthly).

Verified 2026-07-10 against FMP Premium tier:
    GET /stable/historical-price-eod/full?symbol=^VIX
    → 643 rows of daily OHLCV back to 2024-01-01 (latest: 2026-07-09,
      VIX close 15.84 — matches CBOE-published level for that date).

Data is cached to `~/trade_data/ETFTrader/processed/vix_daily.parquet`.
The regime overlay reads from this cache. Weekly refresh cadence
matches the fundamentals refresh so both stay in step.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests

try:
    from src.utils.logging_config import get_logger
except ModuleNotFoundError:
    import logging
    get_logger = logging.getLogger

logger = get_logger(__name__)


_PROCESSED_DIR = Path.home() / "trade_data" / "ETFTrader" / "processed"
VIX_CACHE_PATH = _PROCESSED_DIR / "vix_daily.parquet"
SPY_CACHE_PATH = _PROCESSED_DIR / "spy_daily.parquet"
_BASE_URL = "https://financialmodelingprep.com/stable"


def fetch_daily_history(
    symbol: str,
    from_date: str = "2010-01-01",
    api_key: Optional[str] = None,
    timeout_sec: float = 30.0,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for any FMP-supported symbol.

    Verified 2026-07-10 on Premium tier for ^VIX (4176 rows 2010-2026)
    and SPY (4153 rows 2010-2026).

    Args:
        symbol: FMP symbol. Use "^VIX" for VIX (caret is URL-encoded on
            the way out — do not pre-encode).
        from_date: ISO date to fetch from.
        api_key: FMP API key. Falls back to `FMP_API_KEY` env var.
        timeout_sec: HTTP timeout.

    Returns:
        DataFrame indexed by date (DatetimeIndex, ascending), columns
        [open, high, low, close, volume]. None on failure.
    """
    key = api_key or os.environ.get("FMP_API_KEY")
    if not key:
        logger.warning(f"fetch_daily_history({symbol}): FMP_API_KEY not set")
        return None

    url = f"{_BASE_URL}/historical-price-eod/full"
    try:
        resp = requests.get(
            url,
            params={"symbol": symbol, "from": from_date, "apikey": key},
            timeout=timeout_sec,
        )
    except requests.RequestException as exc:
        logger.warning(f"fetch_daily_history({symbol}): request error: {exc}")
        return None

    if resp.status_code != 200:
        logger.warning(
            f"fetch_daily_history({symbol}): HTTP {resp.status_code}: {resp.text[:200]}"
        )
        return None

    try:
        payload = resp.json()
    except ValueError:
        logger.warning(f"fetch_daily_history({symbol}): non-JSON response")
        return None

    if isinstance(payload, dict) and "historical" in payload:
        rows = payload["historical"]
    elif isinstance(payload, list):
        rows = payload
    else:
        logger.warning(
            f"fetch_daily_history({symbol}): unrecognised payload shape: {type(payload)}"
        )
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        logger.warning(
            f"fetch_daily_history({symbol}): 'date' column missing; got {list(df.columns)}"
        )
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]


# Convenience wrappers preserve call sites of the previous `fetch_vix_history`.
def fetch_vix_history(from_date: str = "2010-01-01", **kwargs) -> Optional[pd.DataFrame]:
    return fetch_daily_history("^VIX", from_date=from_date, **kwargs)


def fetch_spy_history(from_date: str = "2010-01-01", **kwargs) -> Optional[pd.DataFrame]:
    return fetch_daily_history("SPY", from_date=from_date, **kwargs)


def _save_cache(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["fetched_at"] = datetime.utcnow().isoformat()
    out.to_parquet(path)
    logger.info(f"{label}: wrote {len(out)} rows to {path}")


def save_vix_cache(df: pd.DataFrame, path: Path = VIX_CACHE_PATH) -> None:
    _save_cache(df, path, "vix")


def save_spy_cache(df: pd.DataFrame, path: Path = SPY_CACHE_PATH) -> None:
    _save_cache(df, path, "spy")


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "fetched_at" in df.columns:
        df = df.drop(columns=["fetched_at"])
    return df


def load_vix_cache(path: Path = VIX_CACHE_PATH) -> Optional[pd.DataFrame]:
    return _load_cache(path)


def load_spy_cache(path: Path = SPY_CACHE_PATH) -> Optional[pd.DataFrame]:
    return _load_cache(path)
