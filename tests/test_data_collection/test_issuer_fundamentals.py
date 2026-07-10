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
    FmpScraper,
    Fundamentals,
    FundamentalsRouter,
    IssuerScraper,
    ScraperConfig,
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
# FMP profile parser — the yield-from-lastDividend calculation
# ────────────────────────────────────────────────────────────────

class TestFmpParseProfile:
    def _scraper(self):
        return FmpScraper(config=ScraperConfig(request_delay_sec=0.0), api_key="test-key")

    def test_normal_etf(self):
        """VOO-like payload: lastDividend / price gives a realistic yield."""
        payload = [{
            "symbol": "VOO",
            "price": 690.69,
            "lastDividend": 7.3456,
            "isEtf": True,
        }]
        result = self._scraper()._parse_profile("VOO", payload)
        # 7.3456 / 690.69 ≈ 0.01064
        assert result.dividend_yield == pytest.approx(0.01064, rel=1e-3)
        assert result.source == "fmp"
        assert math.isnan(result.pe_ratio)     # tier-gated, expected NaN
        assert math.isnan(result.pb_ratio)     # tier-gated, expected NaN

    def test_empty_list(self):
        """FMP returns [] for unknown symbols."""
        result = self._scraper()._parse_profile("XXXX", [])
        assert not result.is_covered
        assert result.source == "fmp"

    def test_zero_price_returns_nan(self):
        """Divide-by-zero guard."""
        payload = [{"symbol": "X", "price": 0, "lastDividend": 5}]
        result = self._scraper()._parse_profile("X", payload)
        assert math.isnan(result.dividend_yield)

    def test_missing_last_dividend_returns_nan(self):
        payload = [{"symbol": "X", "price": 100}]
        result = self._scraper()._parse_profile("X", payload)
        assert math.isnan(result.dividend_yield)

    def test_sanity_clamp_on_extreme_yield(self):
        """A 100% implied yield is almost certainly a data error, not real."""
        payload = [{"symbol": "X", "price": 5.0, "lastDividend": 6.0}]  # 120% yield
        result = self._scraper()._parse_profile("X", payload)
        assert math.isnan(result.dividend_yield)

    def test_matches_only_when_api_key_present(self):
        with_key = FmpScraper(api_key="test-key")
        without_key = FmpScraper(api_key=None)
        without_key.api_key = None  # explicitly nuke env-loaded key
        assert with_key.matches("ANYTHING")
        assert not without_key.matches("ANYTHING")
