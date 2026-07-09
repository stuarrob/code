"""Unit tests for order-execution helpers in notebooks/scripts/s7_execute.py.

These pin the exact trailing-stop quantity for each real-money scenario so
we cannot silently regress the behaviour. Per CLAUDE.md, stop-loss changes
must be locked down by tests.
"""
import pytest

from s7_execute import _compute_trail_qty


@pytest.mark.unit
def test_trail_qty_fresh_position_covers_full_holding():
    """No prior TRAIL and no prior shares → cover the whole post-fill qty."""
    qty = _compute_trail_qty(
        ticker="SPY",
        new_shares=100,
        held_after_fill=100,
        tickers_with_trail=set(),
    )
    assert qty == 100


@pytest.mark.unit
def test_trail_qty_topup_of_protected_position_covers_only_new_shares():
    """Existing TRAIL on the ticker → cover only the NEW shares.

    The prior TRAIL is trailing the previous high-water mark. Placing a
    new full-position TRAIL at the current fill's baseline would weaken
    the existing protection when the price has pulled back from its high.
    """
    qty = _compute_trail_qty(
        ticker="SPY",
        new_shares=50,
        held_after_fill=150,  # existing 100 + new 50
        tickers_with_trail={"SPY"},
    )
    assert qty == 50


@pytest.mark.unit
def test_trail_qty_second_buy_in_same_run_tops_up():
    """First BUY places a TRAIL; the second BUY of the same ticker must
    top up rather than rebase — mirrors the loop inside execute_trades."""
    trails = set()

    q1 = _compute_trail_qty(
        ticker="SPY",
        new_shares=100,
        held_after_fill=100,
        tickers_with_trail=trails,
    )
    trails.add("SPY")  # execute_trades marks it after placing the TRAIL

    q2 = _compute_trail_qty(
        ticker="SPY",
        new_shares=50,
        held_after_fill=150,
        tickers_with_trail=trails,
    )

    assert q1 == 100
    assert q2 == 50


@pytest.mark.unit
def test_trail_qty_open_trail_on_other_ticker_does_not_affect():
    """A TRAIL on a different ticker must not shift the decision here."""
    qty = _compute_trail_qty(
        ticker="QQQ",
        new_shares=80,
        held_after_fill=80,
        tickers_with_trail={"SPY", "TLT"},
    )
    assert qty == 80
