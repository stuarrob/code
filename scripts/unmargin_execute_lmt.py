"""Plan B, corrected: cancel VIDI's Inactive OPG, place LMT+GTC for 4 tickers.

Refinement after the initial MKT-OPG run failed on three counts:
- VIDI: OPG order sits Inactive during trading hours
- VEU / SCHF / IQDY: OPG cancelled outright by IB during trading hours
- IBKR: restricted employer stock (permanent skip)
- GDMA: IB refused as "not available for short sale" (permanent skip for
  today; manual intervention if needed)

This script:
1. Cancels VIDI's Inactive OPG order (order id known from prior log).
2. Places LMT SELL with tif=GTC at reference − 25 bps for VIDI, VEU,
   SCHF, IQDY. GTC persists across close, tries to fill in-hours and
   at each subsequent open until either filled or cancelled.
3. Audit log to a fresh JSONL file.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.portfolio.execution import DEFAULT_AUDIT_DIR
from src.portfolio.ib_state import (
    DEFAULT_IB_HOST, DEFAULT_IB_PORT,
    connect_read_only, fetch_snapshot,
)


LIMIT_OFFSET_BPS = 25.0
TICKERS_TO_SELL = ["VIDI", "VEU", "SCHF", "IQDY"]
# Order id of the Inactive VIDI OPG order from earlier run — needs cancelling
# before we place its LMT replacement to avoid a future double-sell.
CANCEL_ORDER_IDS = [6]


def main() -> int:
    print("Fresh read-only snapshot for current market prices…")
    ib_ro = None
    for cid in range(80, 100):
        try:
            ib_ro = connect_read_only(
                host=DEFAULT_IB_HOST, port=DEFAULT_IB_PORT,
                client_id=cid, timeout=25,
            )
            break
        except Exception as exc:
            print(f"  cid {cid} failed: {str(exc)[:70]}")
    if ib_ro is None:
        return 1
    try:
        snap = fetch_snapshot(ib_ro)
    finally:
        try:
            ib_ro.disconnect()
        except Exception:
            pass

    print(f"NAV ${snap.nav:,.0f}, Cash ${snap.cash:,.0f}")

    held = {p.ticker: p for p in snap.long_positions}
    orders = []
    for t in TICKERS_TO_SELL:
        p = held.get(t)
        if p is None:
            print(f"⚠ {t} not held — skipping")
            continue
        limit_price = round(p.market_price * (1 - LIMIT_OFFSET_BPS / 10_000), 2)
        orders.append({
            "ticker": t,
            "shares": int(p.shares),
            "ref_price": float(p.market_price),
            "limit_price": limit_price,
            "notional": int(p.shares) * limit_price,
        })

    print()
    print("Planned LMT-GTC SELLs (25 bps aggressive-cross):")
    for o in orders:
        print(f"  SELL {o['shares']:>4} {o['ticker']:<6} @ LMT "
              f"${o['limit_price']:.2f}  (ref ${o['ref_price']:.2f}, "
              f"est ${o['notional']:>10,.0f})")
    print(f"  → total: ~${sum(o['notional'] for o in orders):,.0f}")

    print()
    print("Opening WRITE connection to IB Gateway…")
    from ib_insync import IB, LimitOrder, Order, Stock

    ib = None
    for cid in range(70, 80):
        try:
            ib = IB()
            ib.connect(DEFAULT_IB_HOST, DEFAULT_IB_PORT,
                       clientId=cid, readonly=False, timeout=15)
            print(f"  write-connected under client_id {cid}")
            break
        except Exception as exc:
            print(f"  cid {cid} failed: {str(exc)[:70]}")
            try:
                ib.disconnect()
            except Exception:
                pass
            ib = None
    if ib is None:
        print("Could not open WRITE connection. Aborting.")
        return 1

    audit_path = DEFAULT_AUDIT_DIR / (
        f"unmargin_lmt_{datetime.utcnow():%Y%m%d_%H%M%S}.jsonl"
    )
    audit_fh = audit_path.open("w", encoding="utf-8")

    # 1. Cancel any lingering Inactive orders by ID.
    print()
    print("Cancelling prior Inactive orders…")
    try:
        open_trades = ib.openTrades()
        for cancel_id in CANCEL_ORDER_IDS:
            found = False
            for t in open_trades:
                if getattr(t.order, "orderId", None) == cancel_id:
                    ib.cancelOrder(t.order)
                    print(f"  cancelled order {cancel_id} ({t.contract.symbol})")
                    ib.sleep(1)
                    found = True
                    break
            if not found:
                print(f"  order {cancel_id} not in openTrades — nothing to cancel")
    except Exception as exc:
        print(f"  cancel step failed (continuing): {exc}")

    # 2. Place LMT+GTC replacements.
    print()
    print("Placing LMT+GTC orders…")
    results = []
    for o in orders:
        contract = Stock(o["ticker"], "SMART", "USD")
        try:
            ib.qualifyContracts(contract)
        except Exception as exc:
            results.append({**o, "status": "ERROR",
                             "message": f"qualify failed: {exc}"})
            print(f"  {o['ticker']}: qualify failed: {exc}")
            continue
        order = LimitOrder("SELL", o["shares"], o["limit_price"])
        order.tif = "GTC"
        try:
            trade = ib.placeOrder(contract, order)
            ib.sleep(2)
            status = trade.orderStatus.status
            order_id = getattr(trade.order, "orderId", None)
            print(f"  SELL {o['ticker']:<6} @ LMT ${o['limit_price']:.2f} GTC "
                  f"→ {status}  (id {order_id})")
            results.append({
                **o, "status": status, "order_id": order_id,
                "message": f"LMT GTC SELL {o['shares']} {o['ticker']} @ {o['limit_price']}",
            })
        except Exception as exc:
            print(f"  {o['ticker']}: placeOrder failed: {exc}")
            results.append({**o, "status": "ERROR",
                             "message": f"placeOrder failed: {exc}"})

        audit_fh.write(json.dumps({
            **results[-1], "timestamp": datetime.utcnow().isoformat(),
        }, default=str) + "\n")
        audit_fh.flush()

    audit_fh.close()
    try:
        ib.disconnect()
    except Exception:
        pass

    print()
    print(f"Audit written to {audit_path}")
    ok = sum(1 for r in results if r["status"] in
              ("Submitted", "PreSubmitted", "Filled"))
    print(f"{ok} OK · {len(results) - ok} failed · of {len(results)} placed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
