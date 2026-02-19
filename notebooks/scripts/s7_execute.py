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


def get_account_state(
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 16,
) -> dict:
    """Connect to IB and return full account state.

    Returns dict with:
        positions: list of {ticker, shares, avg_cost, mkt_value}
        long_positions: list (shares > 0)
        short_positions: list (shares < 0)
        open_orders: list of {ticker, action, shares, order_type, status, order_id}
        nlv: float
        cash: float
        margin_used: float
    """
    from ib_insync import IB

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10)
    account = ib.managedAccounts()[0]
    print(f"Connected (read-only): {account}")

    # Account summary
    summary = {}
    for av in ib.accountSummary():
        if av.currency == "USD":
            try:
                summary[av.tag] = float(av.value)
            except (ValueError, TypeError):
                pass

    nlv = summary.get("NetLiquidation", 0)
    cash = summary.get("TotalCashValue", 0)
    margin_used = summary.get("MaintMarginReq", 0)
    buying_power = summary.get("BuyingPower", 0)

    # Positions
    positions = []
    for p in ib.positions():
        shares = float(p.position)
        if shares == 0:
            continue
        avg_cost = float(p.avgCost)
        mkt_value = shares * avg_cost  # approximate
        positions.append({
            "ticker": p.contract.symbol,
            "shares": shares,
            "avg_cost": round(avg_cost, 2),
            "mkt_value": round(mkt_value, 2),
        })

    positions.sort(key=lambda x: x["ticker"])
    long_pos = [p for p in positions if p["shares"] > 0]
    short_pos = [p for p in positions if p["shares"] < 0]

    # Open orders
    open_orders = []
    for t in ib.openTrades():
        open_orders.append({
            "ticker": t.contract.symbol,
            "action": t.order.action,
            "shares": int(t.order.totalQuantity),
            "order_type": t.order.orderType,
            "tif": t.order.tif,
            "status": t.orderStatus.status,
            "order_id": t.order.orderId,
            "limit_price": getattr(t.order, "lmtPrice", None),
        })

    ib.disconnect()

    # Print summary
    print(f"\n{'═' * 60}")
    print("ACCOUNT STATE")
    print(f"{'═' * 60}")
    print(f"  NLV:          ${nlv:>12,.0f}")
    print(f"  Cash:         ${cash:>12,.0f}")
    print(f"  Margin used:  ${margin_used:>12,.0f}")
    print(f"  Buying power: ${buying_power:>12,.0f}")
    print(f"  Open orders:  {len(open_orders)}")

    if long_pos:
        total_long = sum(p["mkt_value"] for p in long_pos)
        print(f"\n  LONG positions ({len(long_pos)}):"
              f"  total ~${total_long:,.0f}")
        for p in long_pos:
            print(f"    {p['ticker']:<6} {p['shares']:>8.0f} shares"
                  f"  avg ${p['avg_cost']:>8.2f}"
                  f"  ~${p['mkt_value']:>10,.0f}")

    if short_pos:
        total_short = sum(abs(p["mkt_value"]) for p in short_pos)
        print(f"\n  SHORT positions ({len(short_pos)}):"
              f"  total ~${total_short:,.0f}")
        print("  *** WARNING: SHORT POSITIONS DETECTED ***")
        for p in short_pos:
            print(f"    {p['ticker']:<6} {p['shares']:>8.0f} shares"
                  f"  avg ${p['avg_cost']:>8.2f}"
                  f"  ~${abs(p['mkt_value']):>10,.0f}")

    if cash < 0:
        print(f"\n  *** WARNING: NEGATIVE CASH (${cash:,.0f}) ***")
        print("  *** ACCOUNT IS USING MARGIN/LEVERAGE ***")

    if open_orders:
        print(f"\n  Open orders ({len(open_orders)}):")
        for o in open_orders:
            lp = f" limit ${o['limit_price']:.2f}" if o.get("limit_price") else ""
            print(f"    {o['action']:>4} {o['shares']:>5} {o['ticker']:<6}"
                  f" ({o['order_type']} {o['tif']}){lp}"
                  f" — {o['status']}")
    print(f"{'═' * 60}")

    return {
        "positions": positions,
        "long_positions": long_pos,
        "short_positions": short_pos,
        "open_orders": open_orders,
        "nlv": nlv,
        "cash": cash,
        "margin_used": margin_used,
        "buying_power": buying_power,
        "account": account,
        "summary": summary,
    }


def verify_portfolio_state(
    state_file: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 16,
    cash_tolerance_pct: float = 2.0,
) -> dict:
    """Verify current IB account matches portfolio_state.json snapshot.

    Connects read-only to IB. Compares positions, cash, open orders.

    Returns:
        dict with valid (bool), state, discrepancies, current_state.
    """
    import hashlib
    from ib_insync import IB

    discrepancies = []

    # Load state file
    state_path = Path(state_file)
    if not state_path.exists():
        print("STATE FILE NOT FOUND")
        return {
            "valid": False,
            "state": None,
            "discrepancies": [{
                "field": "state_file",
                "expected": str(state_path),
                "actual": "missing",
                "severity": "CRITICAL",
            }],
            "current_state": None,
        }

    with open(state_path) as f:
        state = json.load(f)

    print(f"State file: {state_path}")
    print(f"Generated: {state['generated_at']}")
    print(f"Account:   {state['account']}")

    # Connect to IB
    ib = IB()
    try:
        ib.connect(
            ib_host, ib_port,
            clientId=ib_client_id,
            readonly=True, timeout=10,
        )
        account = ib.managedAccounts()[0]

        # Verify account matches
        if account != state["account"]:
            discrepancies.append({
                "field": "account",
                "expected": state["account"],
                "actual": account,
                "severity": "CRITICAL",
            })

        # Get current positions
        pos_map = {}
        for p in ib.positions():
            shares = float(p.position)
            if shares != 0:
                sym = p.contract.symbol
                pos_map[sym] = pos_map.get(sym, 0) + shares

        # Get current cash
        current_cash = 0
        for av in ib.accountSummary():
            if (av.currency == "USD"
                    and av.tag == "TotalCashValue"):
                try:
                    current_cash = float(av.value)
                except (ValueError, TypeError):
                    pass

        # Check open orders
        open_orders = len(ib.openTrades())

        ib.disconnect()
    except Exception as e:
        print(f"IB connection failed: {e}")
        return {
            "valid": False,
            "state": state,
            "discrepancies": [{
                "field": "ib_connection",
                "expected": "connected",
                "actual": str(e),
                "severity": "CRITICAL",
            }],
            "current_state": None,
        }

    # Compare positions
    expected_pos = state["pre_trade_state"]["positions"]
    expected_tickers = set(expected_pos.keys())
    actual_tickers = set(pos_map.keys())

    # Missing positions (expected but not in IB)
    for ticker in expected_tickers - actual_tickers:
        exp_shares = expected_pos[ticker]["shares"]
        discrepancies.append({
            "field": f"position:{ticker}",
            "expected": f"{exp_shares:.0f} shares",
            "actual": "not held",
            "severity": "CRITICAL",
        })

    # Unexpected positions (in IB but not expected)
    for ticker in actual_tickers - expected_tickers:
        discrepancies.append({
            "field": f"position:{ticker}",
            "expected": "not held",
            "actual": f"{pos_map[ticker]:.0f} shares",
            "severity": "CRITICAL",
        })

    # Share count mismatches
    for ticker in expected_tickers & actual_tickers:
        exp_shares = expected_pos[ticker]["shares"]
        act_shares = pos_map[ticker]
        if abs(exp_shares - act_shares) > 0.5:
            discrepancies.append({
                "field": f"position:{ticker}",
                "expected": f"{exp_shares:.0f} shares",
                "actual": f"{act_shares:.0f} shares",
                "severity": "CRITICAL",
            })

    # Check for shorts
    for ticker, shares in pos_map.items():
        if shares < 0:
            discrepancies.append({
                "field": f"short:{ticker}",
                "expected": "no shorts",
                "actual": f"{shares:.0f} shares",
                "severity": "CRITICAL",
            })

    # Check cash (with tolerance)
    expected_cash = state["pre_trade_state"]["cash"]
    if expected_cash > 0:
        cash_diff_pct = (
            abs(current_cash - expected_cash) / expected_cash * 100
        )
        if cash_diff_pct > cash_tolerance_pct:
            severity = (
                "CRITICAL" if cash_diff_pct > 10
                else "WARNING"
            )
            discrepancies.append({
                "field": "cash",
                "expected": f"${expected_cash:,.0f}",
                "actual": f"${current_cash:,.0f}"
                          f" ({cash_diff_pct:+.1f}%)",
                "severity": severity,
            })

    # Check open orders (should be 0 after cleanup)
    if open_orders > 0:
        discrepancies.append({
            "field": "open_orders",
            "expected": "0",
            "actual": str(open_orders),
            "severity": "CRITICAL",
        })

    # Verify checksums
    live_dir = state_path.parent
    for label, filename in [
        ("trade_plan_csv", "trade_plan.csv"),
        ("target_portfolio_csv", "target_portfolio_latest.csv"),
    ]:
        expected_hash = state.get("checksums", {}).get(label)
        if expected_hash:
            fpath = live_dir / filename
            if fpath.exists():
                actual_hash = (
                    "sha256:"
                    + hashlib.sha256(
                        fpath.read_bytes()
                    ).hexdigest()
                )
                if actual_hash != expected_hash:
                    discrepancies.append({
                        "field": f"checksum:{filename}",
                        "expected": expected_hash[:20] + "...",
                        "actual": actual_hash[:20] + "...",
                        "severity": "CRITICAL",
                    })

    # Print summary
    critical = [
        d for d in discrepancies if d["severity"] == "CRITICAL"
    ]
    warnings = [
        d for d in discrepancies if d["severity"] == "WARNING"
    ]
    valid = len(critical) == 0

    print(f"\n{'═' * 50}")
    if valid:
        print("STATE VERIFICATION: PASSED")
    else:
        print("STATE VERIFICATION: FAILED")
    print(f"{'═' * 50}")
    print(
        f"  Positions: {len(expected_tickers)} expected,"
        f" {len(actual_tickers)} actual"
    )
    print(
        f"  Cash: ${current_cash:,.0f}"
        f" (expected ${expected_cash:,.0f})"
    )
    print(f"  Open orders: {open_orders}")

    if critical:
        print(f"\n  CRITICAL issues ({len(critical)}):")
        for d in critical:
            print(
                f"    {d['field']}: expected={d['expected']},"
                f" actual={d['actual']}"
            )
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for d in warnings:
            print(
                f"    {d['field']}: expected={d['expected']},"
                f" actual={d['actual']}"
            )
    if not discrepancies:
        print("  No discrepancies found")

    print(f"{'═' * 50}")

    current_state = {
        "positions": pos_map,
        "cash": current_cash,
        "open_orders": open_orders,
    }

    return {
        "valid": valid,
        "state": state,
        "discrepancies": discrepancies,
        "current_state": current_state,
    }


def generate_reversal_trades(
    account_state: dict,
    target_weights: "pd.Series" = None,
    cash_reserve: float = 70_000,
) -> list:
    """Generate trades to flatten the account back to a safe state.

    Covers all short positions and optionally sells excess longs to
    restore a positive cash balance with the specified reserve.

    Args:
        account_state: Output from get_account_state().
        target_weights: If provided, keeps positions matching targets.
            If None, closes ALL positions (full liquidation to cash).
        cash_reserve: Minimum cash to maintain after reversal.

    Returns:
        List of trade dicts for execute_trades().
    """
    trades = []

    # Phase 1: Cover ALL short positions (mandatory)
    for p in account_state["short_positions"]:
        shares = int(abs(p["shares"]))
        trades.append({
            "ticker": p["ticker"],
            "action": "BUY_TO_COVER",
            "order_type": "MARKET",
            "shares": shares,
            "limit_price": None,
            "ref_price": p["avg_cost"],
            "note": f"Cover short ({p['shares']:.0f} shares)",
        })

    if target_weights is None:
        # Full liquidation — sell everything
        for p in account_state["long_positions"]:
            if p["ticker"] == "IBKR":
                continue
            shares = int(p["shares"])
            if shares > 0:
                trades.append({
                    "ticker": p["ticker"],
                    "action": "SELL",
                    "order_type": "MARKET",
                    "shares": shares,
                    "limit_price": None,
                    "ref_price": p["avg_cost"],
                    "note": f"Liquidate long ({shares} shares)",
                })
    else:
        # Selective: sell positions NOT in target, trim oversized positions
        targets = target_weights.to_dict()
        nlv = account_state["nlv"]

        for p in account_state["long_positions"]:
            ticker = p["ticker"]
            if ticker == "IBKR":
                continue
            shares = int(p["shares"])
            tw = targets.get(ticker, 0)

            if tw == 0:
                # Not in target — sell all
                trades.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "order_type": "MARKET",
                    "shares": shares,
                    "limit_price": None,
                    "ref_price": p["avg_cost"],
                    "note": f"Not in target — sell all ({shares})",
                })
            else:
                # In target — check if oversized
                target_shares = int(tw * nlv / p["avg_cost"])
                excess = shares - target_shares
                if excess > 0:
                    trades.append({
                        "ticker": ticker,
                        "action": "SELL",
                        "order_type": "MARKET",
                        "shares": excess,
                        "limit_price": None,
                        "ref_price": p["avg_cost"],
                        "note": f"Trim excess ({shares} → {target_shares})",
                    })

    # Summary
    cover_cost = sum(
        t["shares"] * t["ref_price"]
        for t in trades if t["action"] == "BUY_TO_COVER"
    )
    sell_proceeds = sum(
        t["shares"] * t["ref_price"]
        for t in trades if t["action"] == "SELL"
    )

    print(f"\nReversal plan: {len(trades)} trades")
    covers = [t for t in trades if t["action"] == "BUY_TO_COVER"]
    sells = [t for t in trades if t["action"] == "SELL"]
    if covers:
        print(f"  Cover {len(covers)} shorts: ~${cover_cost:,.0f}")
    if sells:
        print(f"  Sell {len(sells)} positions: ~${sell_proceeds:,.0f}")
    net = sell_proceeds - cover_cost
    print(f"  Estimated net cash change: ${net:+,.0f}")
    est_cash = account_state["cash"] + net
    print(f"  Estimated cash after: ~${est_cash:,.0f}")

    return trades


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
    ib.connect(
        ib_host, ib_port,
        clientId=ib_client_id, readonly=False, timeout=10,
    )
    acct = ib.managedAccounts()[0]
    print(f"Connected for trading: {acct}")
    order_mode = (
        "LIMIT (GTC)" if use_limit_orders else "MARKET"
    )
    print(f"Order mode: {order_mode}")
    if use_limit_orders:
        print(
            f"Limit buffer: {limit_buffer_pct}%"
            " from last close"
        )

    # ── SAFETY: snapshot positions + cancel duplicates ──
    pos_map = {}
    for p in ib.positions():
        sym = p.contract.symbol
        pos_map[sym] = pos_map.get(sym, 0) + float(p.position)

    trade_tickers = set(
        t["ticker"] for t in final_trades
        if t["action"] != "SKIP"
    )
    existing_orders = ib.openTrades()
    cancelled = 0
    for ot in existing_orders:
        sym = ot.contract.symbol
        if sym in trade_tickers:
            print(
                f"  CANCEL duplicate: {ot.order.action}"
                f" {int(ot.order.totalQuantity)} {sym}"
                f" ({ot.order.orderType})"
            )
            ib.cancelOrder(ot.order)
            cancelled += 1
    if cancelled:
        ib.sleep(2)
        print(
            f"  Cancelled {cancelled} existing orders"
            " to prevent duplicates\n"
        )

    # Get account cash for buy budget check
    cash_available = 0
    for av in ib.accountSummary():
        if (av.currency == "USD"
                and av.tag == "TotalCashValue"):
            try:
                cash_available = float(av.value)
            except (ValueError, TypeError):
                pass

    print(
        f"  Cash available: ${cash_available:,.0f}"
    )
    print(
        f"  Positions: {len(pos_map)} tickers\n"
    )

    exec_results = []
    trailing_stops_placed = 0
    cumulative_buy_spend = 0

    for t in final_trades:
        if t["action"] == "SKIP":
            continue

        ticker = t["ticker"]
        is_buy = t["action"] in ("BUY", "BUY_TO_COVER")
        ib_action = "BUY" if is_buy else "SELL"
        shares = t["shares"]
        order_type = t.get("order_type", "MARKET")

        # ── SAFETY: prevent short selling ──────────
        if not is_buy:
            held = pos_map.get(ticker, 0)
            if held <= 0:
                print(
                    f"  {ticker}: BLOCKED — would go"
                    f" short (hold {held:.0f},"
                    f" sell {shares})"
                )
                exec_results.append({
                    "ticker": ticker,
                    "status": "BLOCKED_SHORT",
                    "message": (
                        f"Hold {held:.0f},"
                        f" sell {shares} = short"
                    ),
                })
                continue
            if shares > held:
                print(
                    f"  {ticker}: REDUCED — hold"
                    f" {held:.0f}, sell capped"
                    f" from {shares}"
                )
                shares = int(held)

        # ── SAFETY: prevent negative cash ──────────
        if is_buy:
            ref = (
                t.get("ref_price")
                or t.get("limit_price")
                or 0
            )
            est_cost = shares * ref if ref else 0
            headroom = cash_available - cumulative_buy_spend
            if headroom <= 0:
                print(
                    f"  {ticker}: BLOCKED — no cash"
                    f" left (${headroom:,.0f})"
                )
                exec_results.append({
                    "ticker": ticker,
                    "status": "BLOCKED_CASH",
                    "message": "No cash remaining",
                })
                continue
            if est_cost > headroom and ref > 0:
                old_shares = shares
                shares = int(headroom / ref)
                if shares <= 0:
                    print(
                        f"  {ticker}: BLOCKED — cost"
                        f" ${est_cost:,.0f} > cash"
                        f" ${headroom:,.0f}"
                    )
                    exec_results.append({
                        "ticker": ticker,
                        "status": "BLOCKED_CASH",
                        "message": (
                            f"Cost ${est_cost:,.0f}"
                            f" > cash ${headroom:,.0f}"
                        ),
                    })
                    continue
                est_cost = shares * ref
                print(
                    f"  {ticker}: REDUCED {old_shares}"
                    f" → {shares} shares (cash cap)"
                )
            cumulative_buy_spend += est_cost

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
                buf = (
                    1 + limit_buffer_pct / 100
                    if is_buy
                    else 1 - limit_buffer_pct / 100
                )
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
            "order_id": trade_obj.order.orderId,
            "fill_price": fill,
            "limit_price": limit_px,
            "order_type": effective_type,
        })

        # Update position map after trade
        if status in ("Filled", "Submitted", "PreSubmitted"):
            if is_buy:
                pos_map[ticker] = (
                    pos_map.get(ticker, 0) + shares
                )
            else:
                pos_map[ticker] = (
                    pos_map.get(ticker, 0) - shares
                )

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
    ib.connect(
        ib_host, ib_port,
        clientId=ib_client_id, readonly=False, timeout=10,
    )
    account = ib.managedAccounts()[0]
    print(f"Connected for trading: {account}\n")

    # ── SAFETY: snapshot positions for short check ──
    pos_map = {}
    for p in ib.positions():
        sym = p.contract.symbol
        pos_map[sym] = (
            pos_map.get(sym, 0) + float(p.position)
        )

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

        # ── SAFETY: prevent short selling ──────────
        if not is_buy:
            held = pos_map.get(ticker, 0)
            if held <= 0:
                print(
                    f"  {ticker}: BLOCKED — would go"
                    f" short (hold {held:.0f},"
                    f" sell {shares})"
                )
                exec_results.append({
                    "ticker": ticker,
                    "status": "BLOCKED_SHORT",
                    "message": (
                        f"Hold {held:.0f},"
                        f" sell {shares}"
                    ),
                })
                continue
            if shares > held:
                print(
                    f"  {ticker}: REDUCED — hold"
                    f" {held:.0f}, sell capped"
                    f" from {shares}"
                )
                shares = int(held)

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


def verify_execution(
    state_file: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 16,
    cash_tolerance_pct: float = 5.0,
) -> dict:
    """Verify post-trade positions match expected state.

    Run after market hours once all orders should have filled.

    Returns:
        dict with complete (bool), matches, mismatches,
        unexpected, pending_orders, cash comparison.
    """
    from ib_insync import IB

    state_path = Path(state_file)
    if not state_path.exists():
        print("State file not found — cannot verify execution")
        return {"complete": False, "matches": [], "mismatches": [],
                "unexpected": [], "pending_orders": [],
                "estimated_cash_diff": 0}

    with open(state_path) as f:
        state = json.load(f)

    expected = state["expected_post_trade"]
    expected_pos = expected["positions"]
    expected_cash = expected["estimated_cash"]

    ib = IB()
    try:
        ib.connect(
            ib_host, ib_port,
            clientId=ib_client_id,
            readonly=True, timeout=10,
        )
        account = ib.managedAccounts()[0]
        print(f"Connected (read-only): {account}")

        # Current positions
        pos_map = {}
        for p in ib.positions():
            shares = float(p.position)
            if shares != 0:
                sym = p.contract.symbol
                pos_map[sym] = pos_map.get(sym, 0) + shares

        # Current cash
        current_cash = 0
        for av in ib.accountSummary():
            if (av.currency == "USD"
                    and av.tag == "TotalCashValue"):
                try:
                    current_cash = float(av.value)
                except (ValueError, TypeError):
                    pass

        # Pending orders
        pending_orders = []
        for t in ib.openTrades():
            if t.order.orderType != "TRAIL":
                pending_orders.append({
                    "ticker": t.contract.symbol,
                    "action": t.order.action,
                    "shares": int(t.order.totalQuantity),
                    "order_type": t.order.orderType,
                    "tif": t.order.tif,
                    "status": t.orderStatus.status,
                    "filled_qty": int(t.orderStatus.filled),
                    "remaining": int(t.orderStatus.remaining),
                })

        ib.disconnect()
    except Exception as e:
        print(f"IB connection failed: {e}")
        return {"complete": False, "matches": [], "mismatches": [],
                "unexpected": [], "pending_orders": [],
                "estimated_cash_diff": 0}

    # Compare positions
    matches = []
    mismatches = []
    unexpected = []

    expected_tickers = set(expected_pos.keys())
    actual_tickers = set(pos_map.keys())

    for ticker in expected_tickers:
        exp_shares = expected_pos[ticker]["shares"]
        act_shares = pos_map.get(ticker, 0)
        entry = {
            "ticker": ticker,
            "expected_shares": exp_shares,
            "actual_shares": act_shares,
        }
        if abs(exp_shares - act_shares) < 0.5:
            matches.append(entry)
        else:
            entry["diff"] = act_shares - exp_shares
            mismatches.append(entry)

    for ticker in actual_tickers - expected_tickers:
        # IBKR stock is expected to remain
        if ticker == "IBKR":
            continue
        unexpected.append({
            "ticker": ticker,
            "shares": pos_map[ticker],
        })

    cash_diff = current_cash - expected_cash
    complete = (
        len(mismatches) == 0
        and len(unexpected) == 0
        and len(pending_orders) == 0
    )

    # Print summary
    print(f"\n{'═' * 50}")
    if complete:
        print("EXECUTION VERIFICATION: COMPLETE")
    else:
        print("EXECUTION VERIFICATION: INCOMPLETE")
    print(f"{'═' * 50}")
    print(
        f"  Positions matched:  {len(matches)}"
        f" / {len(expected_tickers)}"
    )
    if mismatches:
        print(f"  Mismatches:         {len(mismatches)}")
        for m in mismatches:
            print(
                f"    {m['ticker']:<6}"
                f" expected={m['expected_shares']:.0f}"
                f" actual={m['actual_shares']:.0f}"
                f" (diff={m['diff']:+.0f})"
            )
    if unexpected:
        print(f"  Unexpected:         {len(unexpected)}")
        for u in unexpected:
            print(
                f"    {u['ticker']:<6}"
                f" {u['shares']:.0f} shares"
            )
    if pending_orders:
        print(
            f"  Pending orders:     {len(pending_orders)}"
        )
        for o in pending_orders:
            print(
                f"    {o['action']} {o['shares']}"
                f" {o['ticker']}"
                f" ({o['order_type']} {o['tif']})"
                f" — {o['status']}"
            )
    print(
        f"  Cash: ${current_cash:,.0f}"
        f" (expected ${expected_cash:,.0f},"
        f" diff ${cash_diff:+,.0f})"
    )
    print(f"{'═' * 50}")

    return {
        "complete": complete,
        "matches": matches,
        "mismatches": mismatches,
        "unexpected": unexpected,
        "pending_orders": pending_orders,
        "estimated_cash_diff": cash_diff,
        "current_cash": current_cash,
        "expected_cash": expected_cash,
    }
