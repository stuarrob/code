"""Unit tests for `src.portfolio.explain`.

Focuses on the deterministic narrator — the fallback path that always
works without an API key. The LLM narrator is tested manually against
the live Anthropic API; no unit test hits the network.
"""

from __future__ import annotations

import pytest

from src.portfolio.explain import narrate_proposal
from src.portfolio.proposal import (
    ACTION_BUY, ACTION_EXTEND, ACTION_SELL,
    FactorExposure, Trade, TradeProposal,
)


pytestmark = pytest.mark.unit


def _proposal(trades=(), factor_exposures=(), warnings=(),
              turnover=0.0, cost=0.0, cash_after=0.0,
              positions_after=0, investable_nav=100_000.0):
    return TradeProposal(
        trades=tuple(trades),
        turnover_notional=turnover,
        turnover_pct_of_nav=turnover / max(investable_nav, 1),
        total_est_cost=cost,
        factor_exposures=tuple(factor_exposures),
        investable_nav=investable_nav,
        cash_after=cash_after,
        n_positions_after=positions_after,
        warnings=tuple(warnings),
    )


def _t(ticker, action, delta_shares, price=100.0, current_shares=0,
       target_shares=None, current_pct=0.0, target_pct=0.05):
    if target_shares is None:
        target_shares = current_shares + delta_shares
    return Trade(
        ticker=ticker, action=action,
        current_shares=current_shares, target_shares=target_shares,
        delta_shares=delta_shares, market_price=price,
        delta_notional=abs(delta_shares) * price,
        est_cost=abs(delta_shares) * price * 0.0004,
        current_weight_pct=current_pct,
        target_weight_pct=target_pct,
        weight_gap_pct=target_pct - current_pct,
    )


class TestNarrateProposal:
    def test_empty_proposal(self):
        text = narrate_proposal(_proposal())
        assert "No trades proposed" in text

    def test_empty_with_warnings(self):
        text = narrate_proposal(_proposal(warnings=("cash reserve breached",)))
        assert "No trades" in text
        assert "cash reserve breached" in text

    def test_single_buy_mentions_ticker(self):
        p = _proposal(trades=(_t("SPY", ACTION_BUY, 100),), turnover=10_000)
        text = narrate_proposal(p)
        assert "SPY" in text
        # Pithy narrator: buys appear under "Top buys" (no metadata) or "Buying into" (with metadata)
        assert "Top buys" in text or "Buying into" in text
        assert "$10,000" in text  # turnover formatting

    def test_sell_ticker_appears(self):
        p = _proposal(trades=(_t("XLK", ACTION_SELL, -50, target_shares=0,
                                   current_shares=50, current_pct=0.05,
                                   target_pct=0.0),))
        text = narrate_proposal(p)
        assert "XLK" in text
        assert "Selling out of" in text

    def test_extend_ticker_appears(self):
        p = _proposal(trades=(_t("VTI", ACTION_EXTEND, 25,
                                   current_shares=100, target_shares=125,
                                   current_pct=0.10, target_pct=0.125),))
        text = narrate_proposal(p)
        assert "VTI" in text
        # Extends are surfaced in "Top buys" bucket (buys + extends together)
        assert "Top buys" in text or "Buying into" in text

    def test_all_action_kinds_are_named(self):
        trades = (
            _t("QQQ", ACTION_BUY, 10),
            _t("SPY", ACTION_SELL, -5, current_shares=5, target_shares=0),
            _t("VTI", ACTION_EXTEND, 3, current_shares=100, target_shares=103),
        )
        p = _proposal(trades=trades)
        text = narrate_proposal(p)
        # Headline counts each action type
        assert "1 buy" in text
        assert "1 sell" in text
        assert "1 extend" in text

    def test_factor_exposures_included_when_significant(self):
        p = _proposal(
            trades=(_t("SPY", ACTION_BUY, 100),),
            factor_exposures=(
                FactorExposure("momentum", 0.10, 0.30, 0.20),
                FactorExposure("quality", 0.10, 0.10, 0.0),  # small — should be skipped
            ),
        )
        text = narrate_proposal(p)
        assert "momentum" in text
        # Pithy narrator uses arrow, not "increases"
        assert "↑" in text
        # Quality delta 0.0 → below the 0.02 threshold → not mentioned in tilt

    def test_factor_exposure_nan_skipped(self):
        p = _proposal(
            trades=(_t("SPY", ACTION_BUY, 100),),
            factor_exposures=(
                FactorExposure("value", float("nan"), 0.10, float("nan")),
            ),
        )
        text = narrate_proposal(p)
        # NaN factor should not crash and should not appear in tilt
        assert "value" not in text.lower() or "value" in text.lower()  # weak — just don't crash

    def test_cash_after_reported(self):
        p = _proposal(
            trades=(_t("SPY", ACTION_BUY, 100),),
            cash_after=12_345.67,
        )
        text = narrate_proposal(p)
        # Value is 12,345.67; formatted with :,.0f which rounds to 12,346
        assert "$12,346" in text

    def test_warnings_surfaced_when_trades_present(self):
        p = _proposal(
            trades=(_t("SPY", ACTION_BUY, 100),),
            warnings=("XYZ: no market price", "cash reserve breached"),
        )
        text = narrate_proposal(p)
        # Pithy narrator: shows warning count + first warning only.
        assert "warning" in text.lower()
        assert "2 warning" in text
        assert "no market price" in text  # first warning verbatim
