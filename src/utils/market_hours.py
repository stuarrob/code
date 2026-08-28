"""US equity market-hours helper for the execution pre-flight check.

Deliberately simple: regular trading hours are 09:30-16:00 US/Eastern,
Monday-Friday. Exchange holidays and early closes are NOT modelled — on
those days the check will say "open" when the market is closed. The
applet surfaces this caveat next to the banner. For a monthly-cadence
rebalance this is an acceptable approximation; orders sent on a holiday
simply queue exactly like out-of-hours orders do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


@dataclass(frozen=True)
class MarketState:
    """Snapshot of whether US regular trading hours are in session.

    Attributes:
        is_open: True during 09:30-16:00 ET on a weekday.
        now_et: The evaluation time, in US/Eastern.
        next_open: The next RTH open (== now if currently open).
        next_close: The next RTH close after `now_et`.
        minutes_to_open: Whole minutes until next_open (0 when open).
    """
    is_open: bool
    now_et: datetime
    next_open: datetime
    next_close: datetime
    minutes_to_open: int


def _next_weekday_at(dt: datetime, t: time) -> datetime:
    """First weekday occurrence of wall-time `t` at or after `dt`."""
    candidate = dt.replace(hour=t.hour, minute=t.minute,
                           second=0, microsecond=0)
    if candidate < dt:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:  # Sat=5, Sun=6
        candidate += timedelta(days=1)
    return candidate


def us_market_state(now: datetime | None = None) -> MarketState:
    """Classify the current moment against US regular trading hours.

    Args:
        now: Optional aware datetime (any tz). Defaults to the current
            system time.

    Returns:
        MarketState. Holidays are not modelled (see module docstring).
    """
    now = now.astimezone(ET) if now is not None else datetime.now(ET)
    is_weekday = now.weekday() < 5
    is_open = is_weekday and RTH_OPEN <= now.time() < RTH_CLOSE

    if is_open:
        next_open = now
        next_close = now.replace(hour=RTH_CLOSE.hour,
                                 minute=RTH_CLOSE.minute,
                                 second=0, microsecond=0)
        minutes = 0
    else:
        next_open = _next_weekday_at(now, RTH_OPEN)
        next_close = next_open.replace(hour=RTH_CLOSE.hour,
                                       minute=RTH_CLOSE.minute)
        minutes = max(0, int((next_open - now).total_seconds() // 60))

    return MarketState(
        is_open=is_open, now_et=now,
        next_open=next_open, next_close=next_close,
        minutes_to_open=minutes,
    )
