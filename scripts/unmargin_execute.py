"""Execute Plan B — the 7-order LMT SELL list to un-margin the 2026-07-10 book.

Fires LMT SELL orders at reference price − 25 bps for each of the 7 tickers
identified by unmargin_helper.py Plan B. Writes an audit log to
~/trade_data/ETFTrader/execution_log/unmargin_TIMESTAMP.jsonl.

Not committed for reuse — one-shot fix for the 2026-07-10 over-invest.
"""

from __future__ import annotations

import json
import sys
import time
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


# Plan B — full exits identified by unmargin_helper.py against the live
# snapshot. Uses MKT-on-Open orders (orderType="MKT", tif="OPG") so every
# ticker is guaranteed to fill at the next US market open, regardless of
# current end-of-day liquidity.
PLAN_B_TICKERS = ["DEW", "VIDI", "GDMA", "IBKR", "VEU", "SCHF", "IQDY"]


def main() -> int:
    print("Connecting to IB Gateway (read-only, for snapshot)…")
    ib_ro = None
    for cid in range(80, 100):
        try:
            ib_ro = connect_read_only(
                host=DEFAULT_IB_HOST, port=DEFAULT_IB_PORT,
                client_id=cid, timeout=25,
            )
            print(f"  connected under client_id {cid}")
            break
        except Exception as exc:
            print(f"  cid {cid} failed: {str(exc)[:70]}")
    if ib_ro is None:
        print("Could not connect. Aborting.", file=sys.stderr)
        return 1

    try:
        snap = fetch_snapshot(ib_ro)
    finally:
        try:
            ib_ro.disconnect()
        except Exception:
            pass

    print(f"Snapshot: NAV ${snap.nav:,.0f}, Cash ${snap.cash:,.0f}, "
          f"{len(snap.positions)} positions")
    time.sleep(3)  # let IB reap the read-only client id

    held = {p.ticker: p for p in snap.long_positions}
    orders = []
    for t in PLAN_B_TICKERS:
        p = held.get(t)
        if p is None:
            print(f"⚠ {t} not in current book — skipping")
            continue
        est_notional = int(p.shares) * float(p.market_price)
        orders.append({
            "ticker": t,
            "shares": int(p.shares),
            "market_price": float(p.market_price),
            "notional": est_notional,
        })

    total_notional = sum(o["notional"] for o in orders)
    print()
    print("Plan B MKT-on-Open SELLs (fill at next US market open):")
    for o in orders:
        print(f"  SELL {o['shares']:>4} {o['ticker']:<6} @ MKT-OPG   "
              f"(ref ${o['market_price']:.2f}, est ~${o['notional']:>10,.0f})")
    print(f"  → total notional to raise: ~${total_notional:,.0f}")

    # Open write connection.
    print()
    print("Opening WRITE connection to IB Gateway…")
    from ib_insync import IB, Order, Stock

    ib = None
    for cid in range(70, 80):
        try:
            ib = IB()
            ib.connect(
                DEFAULT_IB_HOST, DEFAULT_IB_PORT,
                clientId=cid, readonly=False, timeout=15,
            )
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
        print("Could not open WRITE connection. Is Read-Only API OFF on Gateway?",
              file=sys.stderr)
        return 1

    audit_dir = DEFAULT_AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / (
        f"unmargin_{datetime.utcnow():%Y%m%d_%H%M%S}.jsonl"
    )
    audit_fh = audit_path.open("w", encoding="utf-8")

    results = []
    try:
        for o in orders:
            contract = Stock(o["ticker"], "SMART", "USD")
            try:
                ib.qualifyContracts(contract)
            except Exception as exc:
                print(f"  {o['ticker']}: qualify failed: {exc}")
                results.append({**o, "status": "ERROR",
                                 "message": f"qualify failed: {exc}"})
                continue
            order = Order()
            order.action = "SELL"
            order.totalQuantity = o["shares"]
            order.orderType = "MKT"
            order.tif = "OPG"  # At the Open — will execute in the opening auction
            try:
                trade = ib.placeOrder(contract, order)
                ib.sleep(2)
                status = trade.orderStatus.status
                order_id = getattr(trade.order, "orderId", None)
                print(f"  SELL {o['ticker']:<6} @ MKT-OPG → "
                      f"{status}  (orderId {order_id})")
                results.append({
                    **o, "status": status, "order_id": order_id,
                    "message": f"MKT-OPG SELL {o['shares']} {o['ticker']}",
                })
            except Exception as exc:
                print(f"  {o['ticker']}: placeOrder failed: {exc}")
                results.append({**o, "status": "ERROR",
                                 "message": f"placeOrder failed: {exc}"})

            audit_fh.write(json.dumps({
                **results[-1], "timestamp": datetime.utcnow().isoformat(),
            }, default=str) + "\n")
            audit_fh.flush()
    finally:
        audit_fh.close()
        try:
            ib.disconnect()
        except Exception:
            pass

    print()
    print(f"Audit written to {audit_path}")
    ok = sum(1 for r in results if r["status"] in
              ("Filled", "Submitted", "PreSubmitted"))
    failed = len(results) - ok
    print(f"{ok} OK · {failed} failed · out of {len(results)} placed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
