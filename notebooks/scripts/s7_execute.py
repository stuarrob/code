"""
Step 7: Trade Execution

Executes trade plan on IB Gateway with trailing stops on all BUY fills.

SAFETY: Requires CONFIRM=True to place real orders.
"""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


TRAILING_STOP_PCT = 10  # 10% trailing stop on all buys


def load_trade_plan(trade_plan_file: Path) -> list:
    """Load trade plan from CSV file."""
    trades = []
    with open(trade_plan_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        f.seek(0)
        # Skip comment lines
        lines = [l for l in f if not l.startswith("#")]
    if not lines:
        return []
    import io
    reader = csv.DictReader(io.StringIO("".join(lines)))
    for row in reader:
        row["shares"] = int(row["shares"])
        row["price"] = float(row["price"])
        row["est_value"] = float(row["est_value"])
        trades.append(row)
    return trades


def apply_custom_instructions(trades: list, custom_instructions: dict) -> list:
    """Apply user overrides to trade instructions.

    custom_instructions: {"TICKER": "instruction"} e.g.
        {"BND": "SKIP", "DIM": "reduce to 30 shares"}
    """
    for t in trades:
        if t["ticker"] in custom_instructions:
            t["instruction"] = custom_instructions[t["ticker"]]
    return trades


def interpret_trades(trades: list) -> list:
    """Interpret trade instructions into concrete execution params.

    Uses Claude API if ANTHROPIC_API_KEY is set, otherwise keyword fallback.
    """
    result = []
    custom = [t for t in trades
              if t["instruction"].upper() not in ("APPROVE", "SKIP", "")]

    claude_map = {}
    if custom:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                trade_lines = [
                    f"  - {t['action']} {t['shares']} shares of {t['ticker']} "
                    f"@ ${t['price']:.2f} | Instruction: \"{t['instruction']}\""
                    for t in custom
                ]
                prompt = (
                    "Interpret each trade instruction into concrete parameters.\n"
                    "Return JSON array: [{\"ticker\", \"action\" (BUY/SELL/SKIP), "
                    "\"order_type\" (MARKET/LIMIT), \"shares\" (int), "
                    "\"limit_price\" (number|null), \"note\"}]\n\n"
                    f"Trades:\n{chr(10).join(trade_lines)}\n\nReturn ONLY JSON."
                )
                msg = client.messages.create(
                    model="claude-sonnet-4-5-20250929", max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                interpreted = json.loads(text)
                claude_map = {i["ticker"]: i for i in interpreted}
            except Exception as e:
                print(f"Claude failed: {e}. Using keyword fallback.")

    for t in trades:
        instr = t["instruction"].upper()
        if instr in ("SKIP", ""):
            continue

        if instr == "APPROVE":
            result.append({
                "ticker": t["ticker"], "action": t["action"],
                "order_type": "MARKET", "shares": t["shares"],
                "limit_price": None, "ref_price": t.get("price"),
                "note": "Approved",
            })
        elif t["ticker"] in claude_map:
            ci = claude_map[t["ticker"]]
            if ci["action"] != "SKIP":
                result.append(ci)
        else:
            il = t["instruction"].lower()
            if "reduce" in il:
                m = re.search(r"(\d+)\s*shares?", il)
                if m:
                    result.append({
                        "ticker": t["ticker"], "action": t["action"],
                        "order_type": "MARKET", "shares": int(m.group(1)),
                        "limit_price": None, "ref_price": t.get("price"),
                        "note": f"Reduced to {m.group(1)} shares",
                    })
                    continue
            if "limit" in il:
                m = re.search(r"\$?([\d.]+)", il)
                if m:
                    result.append({
                        "ticker": t["ticker"], "action": t["action"],
                        "order_type": "LIMIT", "shares": t["shares"],
                        "limit_price": float(m.group(1)),
                        "ref_price": t.get("price"),
                        "note": f"Limit @ ${m.group(1)}",
                    })
                    continue
            print(f"  Cannot parse '{t['instruction']}' for {t['ticker']} — skipping")

    return result


def execute_trades(
    final_trades: list,
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 15,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    confirm: bool = False,
    use_limit_orders: bool = False,
    limit_buffer_pct: float = 1.0,
) -> list:
    """Execute trades on IB Gateway.

    Places trailing stops on ALL BUY fills automatically.

    Args:
        final_trades: List of interpreted trade dicts.
        live_dir: Directory for execution log.
        confirm: Must be True to place real orders.
        trailing_stop_pct: Trailing stop percentage (default 10%).
        use_limit_orders: If True, convert MARKET orders to LIMIT with
            GTC TIF. This allows submission outside market hours.
            BUY limits = last close × (1 + buffer), SELL limits =
            last close × (1 - buffer). Orders persist until filled.
        limit_buffer_pct: Buffer percentage for limit price calculation
            (default 1.0%). Wider = more likely to fill, narrower =
            better price but risk of non-fill.

    Returns:
        List of execution result dicts.
    """
    if not confirm:
        order_mode = "LIMIT (GTC)" if use_limit_orders else "MARKET"
        print(f"DRY RUN — CONFIRM=False. No orders placed.")
        print(f"Order mode: {order_mode}")
        if use_limit_orders:
            print(f"Limit buffer: {limit_buffer_pct}% from last close")
        print(f"\nWould execute {len(final_trades)} trades:")
        for t in final_trades:
            ot = t['order_type']
            lp = ""
            if use_limit_orders and ot == "MARKET" and t.get("ref_price"):
                ref = t["ref_price"]
                is_buy = t["action"] in ("BUY", "BUY_TO_COVER")
                buf = 1 + limit_buffer_pct / 100 if is_buy else 1 - limit_buffer_pct / 100
                lp = f" limit ${ref * buf:.2f}"
                ot = "LIMIT GTC"
            print(f"  {t['action']:>13} {t['shares']:>5} {t['ticker']:<6} "
                  f"({ot}){lp}")
        print(f"\nTrailing stop: {trailing_stop_pct}% on all BUY fills")
        return []

    from ib_insync import IB, Stock, Order, LimitOrder, MarketOrder

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=False, timeout=10)
    print(f"Connected for trading: {ib.managedAccounts()[0]}")
    order_mode = "LIMIT (GTC)" if use_limit_orders else "MARKET"
    print(f"Order mode: {order_mode}")
    if use_limit_orders:
        print(f"Limit buffer: {limit_buffer_pct}% from last close")
    print()

    exec_results = []
    trailing_stops_placed = 0

    for t in final_trades:
        if t["action"] == "SKIP":
            continue

        ticker = t["ticker"]
        ib_action = "BUY" if t["action"] in ("BUY", "BUY_TO_COVER") else "SELL"
        shares = t["shares"]
        order_type = t.get("order_type", "MARKET")

        contract = Stock(ticker, "SMART", "USD")
        try:
            ib.qualifyContracts(contract)
        except Exception as e:
            print(f"  {ticker}: FAILED — {e}")
            exec_results.append({
                "ticker": ticker, "status": "FAILED", "message": str(e),
            })
            continue

        # Determine order type and price
        if order_type == "LIMIT" and t.get("limit_price"):
            # Explicit limit from custom instruction — honour it
            order = LimitOrder(ib_action, shares, t["limit_price"])
            order.tif = "GTC"
            effective_type = "LIMIT"
            limit_px = t["limit_price"]
        elif use_limit_orders:
            # Auto-convert MARKET → LIMIT GTC for outside-hours submission
            ref_price = t.get("ref_price") or t.get("limit_price")
            if not ref_price:
                print(f"  {ticker}: No reference price — using MARKET")
                order = MarketOrder(ib_action, shares)
                effective_type = "MARKET"
                limit_px = None
            else:
                is_buy = ib_action == "BUY"
                buf = 1 + limit_buffer_pct / 100 if is_buy else 1 - limit_buffer_pct / 100
                limit_px = round(ref_price * buf, 2)
                order = LimitOrder(ib_action, shares, limit_px)
                order.tif = "GTC"
                order.outsideRth = False  # Only fill during RTH
                effective_type = "LIMIT GTC"
        else:
            order = MarketOrder(ib_action, shares)
            effective_type = "MARKET"
            limit_px = None

        price_str = f" @ limit ${limit_px:.2f}" if limit_px else ""
        print(f"  {ib_action} {shares} {ticker} ({effective_type}){price_str}...",
              end="")
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(2)

        status = trade_obj.orderStatus.status
        fill = trade_obj.orderStatus.avgFillPrice
        print(f" {status}" + (f" @ ${fill:.2f}" if fill else ""))

        # Check for errors
        if trade_obj.log:
            last_log = trade_obj.log[-1]
            if last_log.errorCode and last_log.errorCode > 0:
                print(f"    Warning: {last_log.message}")

        exec_results.append({
            "ticker": ticker, "status": status,
            "order_id": trade_obj.order.orderId, "fill_price": fill,
            "limit_price": limit_px, "order_type": effective_type,
        })

        # TRAILING STOP on every BUY fill
        # For LIMIT GTC orders, the stop is placed when order is accepted
        # (Submitted/PreSubmitted) — IB will activate it after the fill
        if (t["action"] == "BUY"
                and status in ("Filled", "Submitted", "PreSubmitted")
                and (fill and fill > 0 or limit_px)):
            stop_ref = fill if fill and fill > 0 else limit_px
            ts_order = Order()
            ts_order.action = "SELL"
            ts_order.totalQuantity = shares
            ts_order.orderType = "TRAIL"
            ts_order.trailingPercent = trailing_stop_pct
            ts_order.tif = "GTC"
            ts_trade = ib.placeOrder(contract, ts_order)
            ib.sleep(1)
            init_stop = stop_ref * (1 - trailing_stop_pct / 100)
            print(f"    TRAILING STOP {trailing_stop_pct}% "
                  f"(~${init_stop:.2f}): {ts_trade.orderStatus.status}")
            trailing_stops_placed += 1

    # Log execution
    log_file = live_dir / "execution_log.csv"
    file_exists = log_file.exists()
    log_ts = datetime.now().isoformat()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "ticker", "status", "order_id",
                        "fill_price", "limit_price", "order_type"])
        for r in exec_results:
            w.writerow([log_ts, r.get("ticker"), r.get("status"),
                        r.get("order_id", ""), r.get("fill_price", ""),
                        r.get("limit_price", ""), r.get("order_type", "")])

    filled = sum(1 for r in exec_results
                 if r["status"] in ("Filled", "Submitted", "PreSubmitted"))
    failed = sum(1 for r in exec_results
                 if r["status"] in ("ERROR", "FAILED"))
    buy_count = sum(1 for t in final_trades if t["action"] == "BUY")

    print(f"\nExecution summary:")
    print(f"  Executed: {filled}  |  Failed: {failed}  |  Total: {len(exec_results)}")
    print(f"  Trailing stops placed: {trailing_stops_placed}/{buy_count} buys")
    if use_limit_orders:
        pending = sum(1 for r in exec_results
                      if r["status"] in ("Submitted", "PreSubmitted"))
        if pending:
            print(f"\n  {pending} LIMIT GTC orders pending — will fill at market open")
            print("  Check IB Gateway/TWS to monitor order status")

    ib.disconnect()
    return exec_results


def check_order_status(
    trade_plan_file: Path,
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 16,
) -> dict:
    """Connect to IB and reconcile open orders + positions against the trade plan.

    Returns a dict with keys:
        filled: list of trades that completed
        pending: list of trades with open orders still working
        missing: list of trades with no open order and no position match
        orphan_orders: list of open orders not matching any planned trade
        positions: dict of {ticker: shares} for all current holdings
        open_order_count: total open orders on account
    """
    from ib_insync import IB

    # Load what we intended
    plan = load_trade_plan(trade_plan_file)
    plan_tickers = {t["ticker"]: t for t in plan}

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10)
    account = ib.managedAccounts()[0]
    print(f"Connected (read-only): {account}\n")

    # Get current state from IB
    open_trades = ib.openTrades()
    positions = ib.positions()
    ib.sleep(1)

    # Build position map: ticker → shares held
    pos_map = {}
    for p in positions:
        sym = p.contract.symbol
        pos_map[sym] = pos_map.get(sym, 0) + p.position

    # Build open order map: ticker → list of open orders
    order_map = {}  # ticker → [order_info, ...]
    for t in open_trades:
        sym = t.contract.symbol
        info = {
            "ticker": sym,
            "action": t.order.action,
            "shares": int(t.order.totalQuantity),
            "order_type": t.order.orderType,
            "limit_price": getattr(t.order, "lmtPrice", None),
            "tif": t.order.tif,
            "status": t.orderStatus.status,
            "filled_qty": int(t.orderStatus.filled),
            "remaining": int(t.orderStatus.remaining),
            "avg_fill": t.orderStatus.avgFillPrice,
            "order_id": t.order.orderId,
            "_trade": t,  # keep reference for cancellation
        }
        order_map.setdefault(sym, []).append(info)

    # Reconcile each planned trade
    filled = []
    pending = []
    missing = []

    for t in plan:
        ticker = t["ticker"]
        action = t["action"]
        planned_shares = t["shares"]
        is_buy = action in ("BUY", "BUY_TO_COVER")
        ib_action = "BUY" if is_buy else "SELL"

        # Check for matching open orders (non-TRAIL)
        ticker_orders = order_map.get(ticker, [])
        matching_orders = [
            o for o in ticker_orders
            if o["action"] == ib_action and o["order_type"] != "TRAIL"
        ]

        if matching_orders:
            o = matching_orders[0]
            if o["status"] == "Filled":
                filled.append({
                    **t, "fill_status": "FILLED",
                    "fill_price": o["avg_fill"],
                    "order_id": o["order_id"],
                })
            else:
                # Still pending (Submitted/PreSubmitted)
                pending.append({
                    **t, "fill_status": "PENDING",
                    "ib_status": o["status"],
                    "filled_qty": o["filled_qty"],
                    "remaining": o["remaining"],
                    "limit_price": o["limit_price"],
                    "order_id": o["order_id"],
                    "_trade": o["_trade"],
                })
        else:
            # No matching open order — check if position suggests it filled
            held = pos_map.get(ticker, 0)
            if is_buy and held >= planned_shares:
                filled.append({
                    **t, "fill_status": "FILLED (position confirms)",
                    "fill_price": None, "order_id": None,
                })
            elif not is_buy and held == 0:
                filled.append({
                    **t, "fill_status": "FILLED (position confirms: 0 held)",
                    "fill_price": None, "order_id": None,
                })
            else:
                missing.append({
                    **t, "fill_status": "MISSING",
                    "current_position": held,
                })

    # Identify orphan orders (open orders not matching any planned trade)
    planned_set = set(plan_tickers.keys())
    orphan_orders = []
    for sym, orders in order_map.items():
        for o in orders:
            # Trailing stops for held positions are expected, not orphans
            if o["order_type"] == "TRAIL" and pos_map.get(sym, 0) > 0:
                continue
            if sym not in planned_set:
                orphan_orders.append(o)
            # Also flag TRAIL orders for tickers we don't hold
            elif o["order_type"] == "TRAIL" and pos_map.get(sym, 0) == 0:
                orphan_orders.append(o)

    ib.disconnect()

    # Summary
    total_planned = len(plan)
    print(f"Trade plan: {total_planned} trades")
    print(f"  Filled:  {len(filled)}")
    print(f"  Pending: {len(pending)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Orphan orders: {len(orphan_orders)}")
    print(f"\nOpen orders on account: {len(open_trades)}")
    print(f"Positions held: {len([v for v in pos_map.values() if v != 0])}")

    if not pending and not missing and not orphan_orders:
        print("\nAll trades completed successfully. No action needed.")
    else:
        if pending:
            print(f"\n{len(pending)} orders still pending:")
            for p in pending:
                print(f"  {p['action']:>13} {p['shares']:>5} {p['ticker']:<6} "
                      f"— {p['ib_status']}, filled {p['filled_qty']}/{p['shares']}, "
                      f"limit ${p.get('limit_price', 'N/A')}")
        if missing:
            print(f"\n{len(missing)} trades missing (no order, no position match):")
            for m in missing:
                print(f"  {m['action']:>13} {m['shares']:>5} {m['ticker']:<6} "
                      f"— currently hold {m['current_position']} shares")
        if orphan_orders:
            print(f"\n{len(orphan_orders)} orphan orders (no matching trade plan entry):")
            for o in orphan_orders:
                print(f"  {o['action']} {o['shares']} {o['ticker']} "
                      f"({o['order_type']}) — {o['status']}")

    return {
        "filled": filled,
        "pending": pending,
        "missing": missing,
        "orphan_orders": orphan_orders,
        "positions": pos_map,
        "open_order_count": len(open_trades),
    }


def fix_unfilled_orders(
    check_result: dict,
    trade_plan_file: Path,
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 15,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    use_limit_orders: bool = True,
    limit_buffer_pct: float = 1.0,
    confirm: bool = False,
) -> list:
    """Cancel unfilled orders and resubmit with fresh prices.

    Args:
        check_result: Output from check_order_status().
        trade_plan_file: Path to the original trade plan CSV.
        live_dir: Directory for execution log.
        trailing_stop_pct: Trailing stop percentage on BUY fills.
        use_limit_orders: Use LIMIT GTC for resubmission (recommended).
        limit_buffer_pct: Buffer % for limit price (applied to fresh price).
        confirm: Must be True to place real orders.
    """
    pending = check_result.get("pending", [])
    missing = check_result.get("missing", [])
    orphan_orders = check_result.get("orphan_orders", [])

    to_cancel = pending + [o for o in orphan_orders if "_trade" in o]
    to_resubmit = pending + missing

    if not to_cancel and not to_resubmit:
        print("Nothing to fix — all trades completed successfully.")
        return []

    print(f"Orders to cancel:  {len(to_cancel)}")
    print(f"Trades to resubmit: {len(to_resubmit)}")

    if not confirm:
        print("\nDRY RUN — CONFIRM=False. No orders placed.\n")
        if to_cancel:
            print("Would CANCEL:")
            for t in to_cancel:
                ticker = t.get("ticker", "?")
                oid = t.get("order_id", "?")
                print(f"  {ticker} order #{oid}")
        if to_resubmit:
            print("\nWould RESUBMIT (with fresh prices):")
            for t in to_resubmit:
                action = t.get("action", "?")
                shares = t.get("shares", "?")
                ticker = t.get("ticker", "?")
                mode = "LIMIT GTC" if use_limit_orders else "MARKET"
                print(f"  {action:>13} {shares:>5} {ticker:<6} ({mode})")
        return []

    from ib_insync import IB, Stock, Order, LimitOrder, MarketOrder

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=False, timeout=10)
    account = ib.managedAccounts()[0]
    print(f"Connected for trading: {account}\n")

    # Phase 1: Cancel stale orders
    cancelled = 0
    for t in to_cancel:
        trade_obj = t.get("_trade")
        if trade_obj:
            ticker = t.get("ticker", "?")
            oid = t.get("order_id", "?")
            print(f"  Cancelling {ticker} order #{oid}...", end="")
            ib.cancelOrder(trade_obj.order)
            ib.sleep(1)
            print(" done")
            cancelled += 1

    print(f"\nCancelled {cancelled} orders.\n")
    ib.sleep(2)  # Let cancellations settle

    # Phase 2: Resubmit unfilled trades with fresh prices
    exec_results = []
    trailing_stops_placed = 0

    # Build list of contracts we need prices for
    resubmit_contracts = {}
    for t in to_resubmit:
        ticker = t["ticker"]
        if ticker not in resubmit_contracts:
            contract = Stock(ticker, "SMART", "USD")
            try:
                ib.qualifyContracts(contract)
                resubmit_contracts[ticker] = contract
            except Exception as e:
                print(f"  {ticker}: Failed to qualify — {e}")

    # Get fresh prices
    fresh_prices = {}
    if resubmit_contracts:
        contracts_list = list(resubmit_contracts.values())
        tickers_list = [c.symbol for c in contracts_list]
        print("Fetching fresh prices...", end="")
        ticker_data = ib.reqTickers(*contracts_list)
        ib.sleep(2)
        for td in ticker_data:
            sym = td.contract.symbol
            # Use last price, or close, or midpoint
            price = td.last
            if not price or price <= 0:
                price = td.close
            if not price or price <= 0:
                price = (td.bid + td.ask) / 2 if td.bid and td.ask else None
            if price and price > 0:
                fresh_prices[sym] = price
        print(f" got {len(fresh_prices)} prices")

    # Place new orders
    print()
    for t in to_resubmit:
        ticker = t["ticker"]
        action = t["action"]
        is_buy = action in ("BUY", "BUY_TO_COVER")
        ib_action = "BUY" if is_buy else "SELL"
        shares = t["shares"]

        contract = resubmit_contracts.get(ticker)
        if not contract:
            exec_results.append({
                "ticker": ticker, "status": "FAILED",
                "message": "Could not qualify contract",
            })
            continue

        ref_price = fresh_prices.get(ticker)

        if use_limit_orders and ref_price:
            buf = 1 + limit_buffer_pct / 100 if is_buy else 1 - limit_buffer_pct / 100
            limit_px = round(ref_price * buf, 2)
            order = LimitOrder(ib_action, shares, limit_px)
            order.tif = "GTC"
            order.outsideRth = False
            effective_type = "LIMIT GTC"
        elif ref_price:
            order = MarketOrder(ib_action, shares)
            effective_type = "MARKET"
            limit_px = None
        else:
            # No price available — fall back to original trade plan price
            ref_price = t.get("price")
            if use_limit_orders and ref_price:
                buf = 1 + limit_buffer_pct / 100 if is_buy else 1 - limit_buffer_pct / 100
                limit_px = round(ref_price * buf, 2)
                order = LimitOrder(ib_action, shares, limit_px)
                order.tif = "GTC"
                order.outsideRth = False
                effective_type = "LIMIT GTC (stale price)"
            else:
                order = MarketOrder(ib_action, shares)
                effective_type = "MARKET"
                limit_px = None

        price_str = f" @ limit ${limit_px:.2f}" if limit_px else ""
        print(f"  {ib_action} {shares} {ticker} ({effective_type}){price_str}...",
              end="")
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(2)

        status = trade_obj.orderStatus.status
        fill = trade_obj.orderStatus.avgFillPrice
        print(f" {status}" + (f" @ ${fill:.2f}" if fill else ""))

        if trade_obj.log:
            last_log = trade_obj.log[-1]
            if last_log.errorCode and last_log.errorCode > 0:
                print(f"    Warning: {last_log.message}")

        exec_results.append({
            "ticker": ticker, "status": status,
            "order_id": trade_obj.order.orderId, "fill_price": fill,
            "limit_price": limit_px, "order_type": effective_type,
        })

        # Trailing stop on BUY orders
        if (action == "BUY"
                and status in ("Filled", "Submitted", "PreSubmitted")
                and (fill and fill > 0 or limit_px)):
            stop_ref = fill if fill and fill > 0 else limit_px
            ts_order = Order()
            ts_order.action = "SELL"
            ts_order.totalQuantity = shares
            ts_order.orderType = "TRAIL"
            ts_order.trailingPercent = trailing_stop_pct
            ts_order.tif = "GTC"
            ts_trade = ib.placeOrder(contract, ts_order)
            ib.sleep(1)
            init_stop = stop_ref * (1 - trailing_stop_pct / 100)
            print(f"    TRAILING STOP {trailing_stop_pct}% "
                  f"(~${init_stop:.2f}): {ts_trade.orderStatus.status}")
            trailing_stops_placed += 1

    # Log results
    log_file = live_dir / "execution_log.csv"
    file_exists = log_file.exists()
    log_ts = datetime.now().isoformat()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "ticker", "status", "order_id",
                        "fill_price", "limit_price", "order_type"])
        for r in exec_results:
            w.writerow([log_ts, r.get("ticker"), r.get("status"),
                        r.get("order_id", ""), r.get("fill_price", ""),
                        r.get("limit_price", ""), r.get("order_type", "")])

    resubmitted = sum(1 for r in exec_results
                      if r["status"] in ("Filled", "Submitted", "PreSubmitted"))
    failed = sum(1 for r in exec_results
                 if r["status"] in ("ERROR", "FAILED"))

    print(f"\nFix summary:")
    print(f"  Cancelled: {cancelled}")
    print(f"  Resubmitted: {resubmitted}  |  Failed: {failed}")
    print(f"  Trailing stops: {trailing_stops_placed}")
    if use_limit_orders:
        new_pending = sum(1 for r in exec_results
                         if r["status"] in ("Submitted", "PreSubmitted"))
        if new_pending:
            print(f"\n  {new_pending} new LIMIT GTC orders pending — will fill at market open")
            print("  Re-run Step 5 after market hours to verify all filled")

    ib.disconnect()
    return exec_results
