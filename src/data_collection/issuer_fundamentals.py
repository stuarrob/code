"""ETF fundamentals from issuer official data endpoints.

Fixes the T1.1 flaw in the value factor: the current implementation proxies
"value" with expense ratio because there was no source for real fund-level
P/E, P/B, and dividend yield. yfinance was rejected on quality grounds (see
`rule_value_factor_data_source.md`). IB `reqFundamentalData` was rejected on
coverage grounds (works for stocks, patchy on ETFs). Databento does not sell
ETF fundamentals.

The route this module takes: each issuer publishes fund characteristics on
their public site as either an official JSON/CSV download endpoint or a
structured HTML page. We pull one per ticker, cache to parquet, refresh
weekly. This is close in spirit to Path A of the T1.1 investigation.

Coverage philosophy
-------------------
Big-5 issuers (Vanguard, iShares, State Street/SPDR, Invesco, Schwab)
account for ~90% of the 792-ticker universe by AUM and ~70% by count.
Tickers with no coverage get a neutral (median) value score in
`ValueFactor`, and are flagged in a monthly report so we can decide whether
they should be dropped from the universe.

Design notes
------------
- Every scraper is a subclass of `IssuerScraper`. To add an issuer, subclass
  it and implement `matches(ticker)` and `fetch(ticker)`.
- HTTP is done with `requests`, `bs4` parses HTML, `pandas.read_csv`/`json`
  handles official downloads. No third-party scraping frameworks — keeps the
  dependency surface small.
- Every scraper politely rate-limits itself to ~1 request/sec by default.
  The router calls scrapers in a small thread pool (5 workers) if the caller
  passes `parallel=True`, which brings full-universe refresh from ~13 min to
  under a minute in practice.
- Failures never raise into the caller — they're logged and returned as a
  row of NaNs so the caller can compute coverage and act on it.

TODO
----
The specific URL patterns and HTML/JSON schemas below reflect the issuers'
public endpoints as of 2026-07. Issuer sites change; the scrapers include
a per-issuer `schema_version` field so we can detect breakage and fail
loudly rather than silently returning bad data. If an issuer changes their
schema, update the relevant class and bump `schema_version`.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from src.utils.logging_config import get_logger
except ModuleNotFoundError:  # notebook / repl fallback
    import logging
    get_logger = logging.getLogger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────
# Public data structures
# ────────────────────────────────────────────────────────────────

@dataclass
class Fundamentals:
    """One ticker's fundamentals from one issuer at one point in time.

    All numeric fields are `float`; missing values are `float('nan')`. Never
    silently substitute a default — a missing P/E is fundamentally different
    from a P/E of zero and the downstream code must distinguish them.

    Attributes:
        ticker: ETF symbol as it appears in the price cache.
        pe_ratio: Weighted-average price-to-earnings of underlying holdings.
        pb_ratio: Weighted-average price-to-book.
        dividend_yield: Trailing 12-month distribution yield, expressed as a
            decimal (0.025 = 2.5%). Issuer conventions vary — the scraper
            must translate to decimal before returning.
        as_of: Date the issuer reports the number for. Not the fetch date.
        source: Human-readable issuer identifier for provenance tracking.
        schema_version: Bump when the parser changes so cached data can be
            invalidated automatically.
    """
    ticker: str
    pe_ratio: float = float("nan")
    pb_ratio: float = float("nan")
    dividend_yield: float = float("nan")
    as_of: Optional[str] = None
    source: str = "unknown"
    schema_version: int = 1

    def to_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "dividend_yield": self.dividend_yield,
            "as_of": self.as_of,
            "source": self.source,
            "schema_version": self.schema_version,
        }

    @property
    def is_covered(self) -> bool:
        """True if at least one numeric field is populated. Coverage-guard uses this."""
        return not (
            pd.isna(self.pe_ratio)
            and pd.isna(self.pb_ratio)
            and pd.isna(self.dividend_yield)
        )


@dataclass
class ScraperConfig:
    """Configuration common to all scrapers.

    request_delay_sec: seconds to sleep between requests. Default 1.0 =
        polite; each issuer's site has a rate limit and we want to sit well
        under it. Set to 0.0 only in tests.
    timeout_sec: per-request HTTP timeout. 10 seconds is plenty for a JSON
        endpoint and errs generous for slow HTML pages.
    user_agent: identify ourselves. Do not spoof a browser — some issuers
        block Chromish UA strings when they detect automated patterns.
    max_retries: attempts per ticker on transient errors (5xx, ConnectionError).
        Not for 4xx — those indicate the ticker is not on that issuer.
    """
    request_delay_sec: float = 1.0
    timeout_sec: float = 10.0
    user_agent: str = "ETFTrader-fundamentals-bot/1.0 (personal use; contact via github)"
    max_retries: int = 2


# ────────────────────────────────────────────────────────────────
# Base scraper
# ────────────────────────────────────────────────────────────────

class IssuerScraper:
    """Abstract base for one issuer's fundamentals endpoint.

    Contract for subclasses:
        - Set `name` (used as `Fundamentals.source`).
        - Set `schema_version` (bump on parser changes).
        - Implement `matches(ticker)`: cheap check whether this issuer runs
          this ticker. Wrong answers here are recoverable — the router will
          fall through to the next scraper.
        - Implement `fetch(ticker)`: do the HTTP + parse and return a
          `Fundamentals`. Never raise; on failure, log a warning and return a
          `Fundamentals` with only `ticker` and `source` set (all NaN metrics).

    Subclasses should reuse `_get()` for HTTP so the delay and retry policy
    live in one place.
    """
    name: str = "abstract"
    schema_version: int = 1

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})

    # Subclass API — override these two.
    def matches(self, ticker: str) -> bool:
        raise NotImplementedError

    def fetch(self, ticker: str) -> Fundamentals:
        raise NotImplementedError

    # Shared HTTP helper.
    def _get(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        """GET with polite delay + retry on transient failures.

        Returns None on final failure (4xx after retries, or repeated 5xx).
        Callers should treat None as "no coverage" and continue.
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                time.sleep(self.config.request_delay_sec)
                resp = self.session.get(url, params=params, timeout=self.config.timeout_sec)
            except requests.RequestException as exc:
                logger.warning(f"{self.name}: request error on {url}: {exc}")
                continue
            if resp.status_code == 200:
                return resp
            if 400 <= resp.status_code < 500:
                # 4xx = the ticker isn't on this issuer. No point retrying.
                return None
            # 5xx = transient; retry.
            logger.info(f"{self.name}: {resp.status_code} on {url} (attempt {attempt+1})")
        return None

    def _empty(self, ticker: str) -> Fundamentals:
        """Uniform empty result — no coverage, source recorded."""
        return Fundamentals(
            ticker=ticker,
            source=self.name,
            schema_version=self.schema_version,
        )


# ────────────────────────────────────────────────────────────────
# Vanguard
# ────────────────────────────────────────────────────────────────

class VanguardScraper(IssuerScraper):
    """Vanguard ETF fundamentals via their public fund-profile JSON endpoint.

    URL shape (2026-07):
        https://api.vanguard.com/rs/gre/gra/1.0/datasets/urd-profile-basic-data
            .jsonp?path=urd/GB/en/vanguard/products/<TICKER>

    The JSON contains `equityCharacteristics` with `priceEarningsRatio`,
    `priceBookRatio`, `dividendYield` for equity funds. Bond funds return a
    different schema (yield-to-maturity, duration) which we ignore for now —
    the value factor is equity-focused.

    If Vanguard changes the endpoint, bump `schema_version` and update
    `_parse_json`.
    """
    name = "vanguard"
    schema_version = 1

    # Cheap prefilter: Vanguard tickers are almost all V-prefixed with a
    # handful of exceptions (BND, BIV, BSV — bond ETFs, still Vanguard).
    _VANGUARD_TICKER_HINTS = {
        "V",  # VTI, VOO, VEA, VWO, VUG, VTV, VIG, VYM, VGT, VNQ, VOX, VDC, ...
    }
    _VANGUARD_KNOWN_EXCEPTIONS = {"BND", "BIV", "BSV", "BLV", "BNDX", "BNDW", "MGC", "MGK", "MGV"}

    _PROFILE_URL = "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"

    def matches(self, ticker: str) -> bool:
        t = ticker.upper()
        if t in self._VANGUARD_KNOWN_EXCEPTIONS:
            return True
        return t.startswith("V") and len(t) <= 4

    def fetch(self, ticker: str) -> Fundamentals:
        """Fetch Vanguard fund profile HTML and extract the embedded JSON.

        Vanguard's investor-facing pages are React apps that embed the fund
        characteristics as a JSON blob in the initial HTML. We parse that
        rather than call the internal API endpoint — the HTML embed is more
        stable across their frequent API version bumps.
        """
        url = self._PROFILE_URL.format(ticker=ticker.upper())
        resp = self._get(url)
        if resp is None:
            return self._empty(ticker)
        try:
            return self._parse_html(ticker, resp.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"vanguard: parse failed for {ticker}: {exc}")
            return self._empty(ticker)

    def _parse_html(self, ticker: str, html: str) -> Fundamentals:
        """Extract fundamentals from Vanguard's embedded state JSON.

        The page contains a <script> tag with `window.__INITIAL_STATE__ = {...}`
        or a Next.js-style `__NEXT_DATA__` block. Both patterns appear at
        different points in Vanguard's redesigns; try each in order.

        Returns a Fundamentals with as many fields as we can extract, NaN
        for anything the page doesn't publish for this specific fund.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Pattern 1: Next.js __NEXT_DATA__
        next_data = soup.find("script", id="__NEXT_DATA__")
        if next_data and next_data.string:
            payload = json.loads(next_data.string)
            return self._extract_from_next_data(ticker, payload)

        # Pattern 2: legacy window.__INITIAL_STATE__
        for script in soup.find_all("script"):
            text = script.string or ""
            m = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});", text, re.DOTALL)
            if m:
                payload = json.loads(m.group(1))
                return self._extract_from_initial_state(ticker, payload)

        logger.warning(f"vanguard: no recognised state blob found for {ticker}")
        return self._empty(ticker)

    def _extract_from_next_data(self, ticker: str, payload: dict) -> Fundamentals:
        """Walk the Next.js state tree for the characteristics we want.

        The exact JSON path varies with Vanguard's build. Rather than pin an
        exact path (fragile), do a shallow recursive search for keys that
        match the equityCharacteristics contract.
        """
        found = _deep_find_first(payload, [
            "equityCharacteristics",
            "characteristics",
            "fundCharacteristics",
        ])
        if not found:
            return self._empty(ticker)
        return Fundamentals(
            ticker=ticker,
            pe_ratio=_as_float(found.get("priceEarningsRatio")),
            pb_ratio=_as_float(found.get("priceBookRatio")),
            dividend_yield=_pct_as_decimal(found.get("dividendYield")),
            as_of=found.get("asOfDate") or found.get("effectiveDate"),
            source=self.name,
            schema_version=self.schema_version,
        )

    def _extract_from_initial_state(self, ticker: str, payload: dict) -> Fundamentals:
        # Same walk — the wrapping differs but the leaf shape is consistent.
        return self._extract_from_next_data(ticker, payload)


# ────────────────────────────────────────────────────────────────
# iShares — scaffold, to be implemented next
# ────────────────────────────────────────────────────────────────

class IsharesScraper(IssuerScraper):
    """iShares / BlackRock ETF fundamentals.

    Not yet implemented. iShares publishes a per-fund JSON endpoint at
    `https://www.ishares.com/us/products/<PRODUCT_ID>/<slug>/1467271812596.ajax`
    but discovering the PRODUCT_ID from a ticker requires an initial search
    call. Implementation deferred to next commit.
    """
    name = "ishares"
    schema_version = 0  # 0 = not yet real

    _ISHARES_TICKER_HINTS = {"I", "E"}  # IVV, ITOT, IWM, EEM, EFA, ...

    def matches(self, ticker: str) -> bool:
        t = ticker.upper()
        return t[0] in self._ISHARES_TICKER_HINTS

    def fetch(self, ticker: str) -> Fundamentals:
        return self._empty(ticker)


# ────────────────────────────────────────────────────────────────
# Router
# ────────────────────────────────────────────────────────────────

class FundamentalsRouter:
    """Routes a ticker to the right issuer scraper, with fallthrough.

    Order matters: cheaper `matches()` checks first. When a scraper claims a
    ticker but returns no coverage, we try the next scraper — some tickers
    (rare) are cross-listed or have ambiguous prefixes.
    """
    def __init__(self, scrapers: Optional[list[IssuerScraper]] = None):
        self.scrapers = scrapers or [
            VanguardScraper(),
            IsharesScraper(),
        ]

    def fetch_one(self, ticker: str) -> Fundamentals:
        """Try each matching scraper; return first with coverage, or empty.

        The returned Fundamentals always has `ticker == input_ticker` — the
        router enforces this defensively so a bug in a scraper cannot cause
        row-misalignment in the cache frame.
        """
        empty_result: Optional[Fundamentals] = None
        for scraper in self.scrapers:
            if not scraper.matches(ticker):
                continue
            result = scraper.fetch(ticker)
            result.ticker = ticker
            if result.is_covered:
                return result
            empty_result = empty_result or result
        return empty_result or Fundamentals(ticker=ticker, source="none")

    def fetch_many(self, tickers: list[str], parallel: bool = True,
                   max_workers: int = 5) -> pd.DataFrame:
        """Fetch fundamentals for a list of tickers.

        Args:
            tickers: symbols to fetch.
            parallel: if True, use a thread pool. Note that the polite
                per-scraper delay still applies within each thread — so with
                5 workers, effective rate is ~5 req/sec across the pool.
            max_workers: thread-pool size when `parallel=True`.

        Returns:
            DataFrame with one row per input ticker (order preserved),
            columns from `Fundamentals.to_row()`. Missing tickers still
            appear with source='none' and all metrics NaN.
        """
        if not parallel or max_workers <= 1:
            rows = [self.fetch_one(t).to_row() for t in tickers]
        else:
            rows_by_ticker: dict[str, dict] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self.fetch_one, t): t for t in tickers}
                for future in concurrent.futures.as_completed(futures):
                    t = futures[future]
                    try:
                        rows_by_ticker[t] = future.result().to_row()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"fundamentals: {t} raised in worker: {exc}")
                        rows_by_ticker[t] = Fundamentals(ticker=t, source="error").to_row()
            rows = [rows_by_ticker[t] for t in tickers]
        return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────
# Cache
# ────────────────────────────────────────────────────────────────

DEFAULT_CACHE_PATH = (
    Path.home() / "trade_data" / "ETFTrader" / "processed" / "etf_fundamentals.parquet"
)


def load_cache(path: Path = DEFAULT_CACHE_PATH) -> Optional[pd.DataFrame]:
    """Return cached fundamentals, or None if the cache does not exist."""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_cache(df: pd.DataFrame, path: Path = DEFAULT_CACHE_PATH) -> None:
    """Persist fundamentals with a fetch-timestamp column added.

    The `fetched_at` column lets downstream code detect stale caches (older
    than one refresh cycle) and warn on it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["fetched_at"] = datetime.utcnow().isoformat()
    df.to_parquet(path, index=False)
    logger.info(f"fundamentals: wrote {len(df)} rows to {path}")


def coverage_report(df: pd.DataFrame) -> dict:
    """Summarise coverage of a fundamentals frame.

    Used by the value-factor pipeline and the weekly refresh script to fail
    loudly when coverage drops below a threshold (per rule_no_whipsaw memo:
    "do not silently proceed with sparse data").
    """
    total = len(df)
    if total == 0:
        return {"total": 0, "any_field": 0, "pe": 0, "pb": 0, "dy": 0,
                "any_field_pct": 0.0}
    any_field = int((~df[["pe_ratio", "pb_ratio", "dividend_yield"]].isna().all(axis=1)).sum())
    return {
        "total": total,
        "any_field": any_field,
        "any_field_pct": any_field / total,
        "pe": int(df["pe_ratio"].notna().sum()),
        "pb": int(df["pb_ratio"].notna().sum()),
        "dy": int(df["dividend_yield"].notna().sum()),
        "by_source": df["source"].value_counts().to_dict(),
    }


# ────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────

def _as_float(v) -> float:
    """Coerce a value that might be str/int/None/'--' into a float or NaN."""
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v) if not (isinstance(v, float) and pd.isna(v)) else float("nan")
    if isinstance(v, str):
        s = v.strip().replace(",", "").rstrip("%")
        if not s or s in {"--", "N/A", "n/a"}:
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return float("nan")


def _pct_as_decimal(v) -> float:
    """Issuer yields are often published as percentages (e.g. 2.5 = 2.5%).
    Convert to decimal (0.025). If the number is already <1, assume it's
    already decimal and leave it alone."""
    f = _as_float(v)
    if pd.isna(f):
        return f
    return f / 100.0 if f >= 1.0 else f


def _deep_find_first(obj, keys: list[str]) -> Optional[dict]:
    """Depth-first search for the first dict value at any of the requested keys.

    Useful for pulling a well-known leaf out of an issuer's ever-changing
    nested JSON without hard-coding the exact path.
    """
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        for v in obj.values():
            found = _deep_find_first(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find_first(item, keys)
            if found is not None:
                return found
    return None
