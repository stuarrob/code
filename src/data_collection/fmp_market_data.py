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


VIX_CACHE_PATH = (
    Path.home() / "trade_data" / "ETFTrader" / "processed" / "vix_daily.parquet"
)
_BASE_URL = "https://financialmodelingprep.com/stable"


def fetch_vix_history(
    from_date: str = "2010-01-01",
    api_key: Optional[str] = None,
    timeout_sec: float = 30.0,
) -> Optional[pd.DataFrame]:
    """Fetch daily VIX OHLCV from FMP historical-price-eod.

    Args:
        from_date: ISO date to fetch from. FMP returns up to their retained
            history (verified: at least back to 2024-01-01 on Premium;
            older is available but not verified this session).
        api_key: FMP API key. Falls back to `FMP_API_KEY` env var.
        timeout_sec: HTTP timeout.

    Returns:
        DataFrame indexed by date (DatetimeIndex, ascending), columns
        [open, high, low, close, volume]. None on failure.
    """
    key = api_key or os.environ.get("FMP_API_KEY")
    if not key:
        logger.warning("fetch_vix_history: FMP_API_KEY not set")
        return None

    # VIX symbol is ^VIX; URL-encode the caret.
    symbol = quote("^VIX", safe="")
    url = f"{_BASE_URL}/historical-price-eod/full"
    try:
        resp = requests.get(
            url,
            params={"symbol": "^VIX", "from": from_date, "apikey": key},
            timeout=timeout_sec,
        )
    except requests.RequestException as exc:
        logger.warning(f"fetch_vix_history: request error: {exc}")
        return None

    if resp.status_code != 200:
        logger.warning(f"fetch_vix_history: HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        payload = resp.json()
    except ValueError:
        logger.warning("fetch_vix_history: non-JSON response")
        return None

    # FMP `stable/historical-price-eod/full` returns a flat list of rows,
    # not a wrapped dict. Guard against future format changes just in case.
    if isinstance(payload, dict) and "historical" in payload:
        rows = payload["historical"]
    elif isinstance(payload, list):
        rows = payload
    else:
        logger.warning(f"fetch_vix_history: unrecognised payload shape: {type(payload)}")
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        logger.warning(f"fetch_vix_history: 'date' column missing; got {list(df.columns)}")
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Keep the columns we care about; ignore vwap/change/etc. — they're
    # derivable and add cache size for no reason.
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]


def save_vix_cache(df: pd.DataFrame, path: Path = VIX_CACHE_PATH) -> None:
    """Persist VIX history with a fetch-timestamp column added."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["fetched_at"] = datetime.utcnow().isoformat()
    out.to_parquet(path)
    logger.info(f"vix: wrote {len(out)} rows to {path}")


def load_vix_cache(path: Path = VIX_CACHE_PATH) -> Optional[pd.DataFrame]:
    """Return cached VIX series (date-indexed) or None if missing."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "fetched_at" in df.columns:
        df = df.drop(columns=["fetched_at"])
    return df
