"""Unit tests for issuer fundamentals scraper — no network.

Focuses on the pieces that can silently break the value factor:
  - `_pct_as_decimal` (a yield-scaling bug puts every ETF at 100x its yield)
  - the router's coverage fallthrough (a bug here silently drops tickers)
  - `coverage_report` (the guard the pipeline relies on to fail loud)
  - `Fundamentals.is_covered` (the flag that drives the fallthrough)
  - Vanguard's `_extract_from_next_data` on a known payload shape

Network paths (`IssuerScraper._get`, `VanguardScraper.fetch` end-to-end)
are covered by a small integration script the user can run manually — we
do not want CI to hit issuer sites.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data_collection.issuer_fundamentals import (
    Fundamentals,
    FundamentalsRouter,
    IssuerScraper,
    ScraperConfig,
    VanguardScraper,
    _as_float,
    _deep_find_first,
    _pct_as_decimal,
    coverage_report,
)


pytestmark = pytest.mark.unit


class _StubScraper(IssuerScraper):
    """Deterministic scraper for testing router behaviour."""

    def __init__(self, name: str, match_prefix: str, result: Fundamentals):
        super().__init__(ScraperConfig(request_delay_sec=0.0))
        self.name = name
        self._prefix = match_prefix
        self._result = result

    def matches(self, ticker: str) -> bool:
        return ticker.upper().startswith(self._prefix)

    def fetch(self, ticker: str) -> Fundamentals:
        return self._result


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

class TestAsFloat:
    def test_int(self):
        assert _as_float(5) == 5.0

    def test_float(self):
        assert _as_float(3.14) == 3.14

    def test_string_number(self):
        assert _as_float("3.14") == 3.14

    def test_string_with_comma(self):
        assert _as_float("1,234.56") == 1234.56

    def test_string_with_percent(self):
        assert _as_float("2.5%") == 2.5

    def test_none(self):
        assert math.isnan(_as_float(None))

    def test_dashes(self):
        assert math.isnan(_as_float("--"))

    def test_empty(self):
        assert math.isnan(_as_float(""))

    def test_garbage(self):
        assert math.isnan(_as_float("abc"))


class TestPctAsDecimal:
    """The scaling bug that would multiply every yield by 100 lives here."""

    def test_percent_form(self):
        assert _pct_as_decimal(2.5) == pytest.approx(0.025)

    def test_percent_string(self):
        assert _pct_as_decimal("2.5%") == pytest.approx(0.025)

    def test_already_decimal_left_alone(self):
        assert _pct_as_decimal(0.025) == pytest.approx(0.025)

    def test_edge_one_treated_as_percent(self):
        # 1.0 could be 1% or 100% — we treat >=1.0 as percent form.
        assert _pct_as_decimal(1.0) == pytest.approx(0.01)

    def test_none(self):
        assert math.isnan(_pct_as_decimal(None))


class TestDeepFindFirst:
    def test_top_level(self):
        found = _deep_find_first(
            {"equityCharacteristics": {"peRatio": 20}},
            ["equityCharacteristics"],
        )
        assert found == {"peRatio": 20}

    def test_nested(self):
        payload = {"a": {"b": {"characteristics": {"pe": 15}}}}
        found = _deep_find_first(payload, ["characteristics"])
        assert found == {"pe": 15}

    def test_first_of_alternatives(self):
        payload = {"characteristics": {"pe": 10}, "equityCharacteristics": {"pe": 20}}
        # First matching key wins.
        found = _deep_find_first(payload, ["equityCharacteristics", "characteristics"])
        assert found == {"pe": 20}

    def test_not_found(self):
        assert _deep_find_first({"a": 1}, ["b"]) is None

    def test_ignores_non_dict_values(self):
        assert _deep_find_first({"x": 5, "y": None}, ["x"]) is None


# ────────────────────────────────────────────────────────────────
# Fundamentals dataclass
# ────────────────────────────────────────────────────────────────

class TestFundamentalsCovered:
    def test_all_nan_uncovered(self):
        assert not Fundamentals(ticker="X").is_covered

    def test_pe_only_covered(self):
        assert Fundamentals(ticker="X", pe_ratio=20.0).is_covered

    def test_yield_only_covered(self):
        assert Fundamentals(ticker="X", dividend_yield=0.03).is_covered


# ────────────────────────────────────────────────────────────────
# Router
# ────────────────────────────────────────────────────────────────

class TestRouter:
    def test_first_match_wins(self):
        good = Fundamentals(ticker="VOO", pe_ratio=22.0, source="vanguard")
        router = FundamentalsRouter([
            _StubScraper("vanguard", "V", good),
            _StubScraper("ishares", "I", Fundamentals(ticker="wrong")),
        ])
        out = router.fetch_one("VOO")
        assert out.source == "vanguard"
        assert out.pe_ratio == 22.0

    def test_falls_through_on_no_coverage(self):
        """Vanguard claims the ticker but returns no data; iShares picks it up."""
        empty = Fundamentals(ticker="VOO", source="vanguard")  # no metrics
        good = Fundamentals(ticker="VOO", pe_ratio=22.0, source="ishares")
        router = FundamentalsRouter([
            _StubScraper("vanguard", "V", empty),
            _StubScraper("ishares", "V", good),  # both match "V" for this test
        ])
        out = router.fetch_one("VOO")
        assert out.source == "ishares"
        assert out.pe_ratio == 22.0

    def test_no_match_returns_none_source(self):
        router = FundamentalsRouter([_StubScraper("vanguard", "V", Fundamentals(ticker="x"))])
        out = router.fetch_one("QQQ")
        assert out.source == "none"

    def test_returns_first_empty_if_all_uncovered(self):
        """When every scraper claims but none has coverage, keep the first
        result so downstream can see which scraper 'owns' the ticker."""
        e1 = Fundamentals(ticker="VOO", source="vanguard")
        e2 = Fundamentals(ticker="VOO", source="ishares")
        router = FundamentalsRouter([
            _StubScraper("vanguard", "V", e1),
            _StubScraper("ishares", "V", e2),
        ])
        out = router.fetch_one("VOO")
        assert out.source == "vanguard"  # first empty, not last

    def test_fetch_many_preserves_order(self):
        good = Fundamentals(ticker="X", pe_ratio=20.0, source="vanguard")
        router = FundamentalsRouter([_StubScraper("vanguard", "", good)])
        df = router.fetch_many(["VOO", "VTI", "VUG"], parallel=False)
        assert list(df["ticker"]) == ["VOO", "VTI", "VUG"]


# ────────────────────────────────────────────────────────────────
# Coverage report — the guard the pipeline relies on
# ────────────────────────────────────────────────────────────────

class TestCoverageReport:
    def test_empty_frame(self):
        df = pd.DataFrame(columns=["ticker", "pe_ratio", "pb_ratio",
                                    "dividend_yield", "source"])
        report = coverage_report(df)
        assert report["total"] == 0
        assert report["any_field"] == 0

    def test_full_coverage(self):
        df = pd.DataFrame([
            {"ticker": "A", "pe_ratio": 20.0, "pb_ratio": 3.0,
             "dividend_yield": 0.02, "source": "vanguard"},
            {"ticker": "B", "pe_ratio": 15.0, "pb_ratio": 2.5,
             "dividend_yield": 0.03, "source": "ishares"},
        ])
        report = coverage_report(df)
        assert report["total"] == 2
        assert report["any_field"] == 2
        assert report["any_field_pct"] == 1.0
        assert report["pe"] == 2
        assert report["dy"] == 2
        assert report["by_source"] == {"vanguard": 1, "ishares": 1}

    def test_partial_coverage(self):
        df = pd.DataFrame([
            {"ticker": "A", "pe_ratio": 20.0, "pb_ratio": None,
             "dividend_yield": None, "source": "vanguard"},
            {"ticker": "B", "pe_ratio": None, "pb_ratio": None,
             "dividend_yield": None, "source": "none"},
        ])
        report = coverage_report(df)
        assert report["any_field"] == 1
        assert report["any_field_pct"] == 0.5


# ────────────────────────────────────────────────────────────────
# Vanguard-specific extraction
# ────────────────────────────────────────────────────────────────

class TestVanguardExtract:
    def test_next_data_shape(self):
        payload = {
            "props": {
                "pageProps": {
                    "fundData": {
                        "equityCharacteristics": {
                            "priceEarningsRatio": 22.4,
                            "priceBookRatio": 3.1,
                            "dividendYield": 1.35,   # published as pct
                            "asOfDate": "2026-06-30",
                        }
                    }
                }
            }
        }
        scraper = VanguardScraper(ScraperConfig(request_delay_sec=0.0))
        result = scraper._extract_from_next_data("VOO", payload)
        assert result.pe_ratio == 22.4
        assert result.pb_ratio == 3.1
        # 1.35% published → 0.0135 decimal
        assert result.dividend_yield == pytest.approx(0.0135)
        assert result.as_of == "2026-06-30"
        assert result.source == "vanguard"

    def test_next_data_missing_returns_empty(self):
        scraper = VanguardScraper(ScraperConfig(request_delay_sec=0.0))
        result = scraper._extract_from_next_data("VOO", {"unrelated": "junk"})
        assert not result.is_covered
        assert result.source == "vanguard"

    def test_matches_ticker_prefix(self):
        scraper = VanguardScraper(ScraperConfig(request_delay_sec=0.0))
        assert scraper.matches("VTI")
        assert scraper.matches("VOO")
        assert scraper.matches("BND")     # known exception
        assert not scraper.matches("QQQ")
        assert not scraper.matches("SPY")
