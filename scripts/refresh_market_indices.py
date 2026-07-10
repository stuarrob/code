"""Weekly refresh of SPY + VIX daily-close series via FMP.

Called from `weekly_fmp_refresh_windows.cmd`. Kept as a small script rather
than a cmd one-liner because early attempts at chaining Python
expressions in `-c` had ordering bugs and were hard to debug.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_collection.fmp_market_data import (
    fetch_spy_history, save_spy_cache,
    fetch_vix_history, save_vix_cache,
)


def main() -> int:
    spy = fetch_spy_history()
    if spy is None or spy.empty:
        print("refresh_market_indices: SPY fetch returned no rows", file=sys.stderr)
        return 1
    save_spy_cache(spy)

    vix = fetch_vix_history()
    if vix is None or vix.empty:
        print("refresh_market_indices: VIX fetch returned no rows", file=sys.stderr)
        return 1
    save_vix_cache(vix)

    print(f"OK — SPY {len(spy)} rows, VIX {len(vix)} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
