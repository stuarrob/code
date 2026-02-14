#!/usr/bin/env python3
"""
Step 2: Read trade plan, interpret instructions with Claude, execute.

Reads the trade plan file (edited by the user), uses Claude to
interpret any custom instructions, then executes on IB after
confirmation.

Usage:
    # Dry run (show what would execute, don't trade)
    python scripts/ib_execute_trades.py --dry-run

    # Execute trades
    python scripts/ib_execute_trades.py

    # Use a specific plan file
    python scripts/ib_execute_trades.py --plan path/to/plan.csv

Requirements:
    ANTHROPIC_API_KEY env var must be set for Claude interpretation.
    If not set, only APPROVE/SKIP instructions are supported.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import nest_asyncio
from ib_insync import IB, Order, Stock, MarketOrder, LimitOrder, StopOrder

nest_asyncio.apply()

PROJECT_ROOT = Path(__file__).parent.parent
TRADING_DIR = Path.home() / "trading"
LIVE_DIR = TRADING_DIR / "live_portfolio"
TRADE_PLAN_FILE = LIVE_DIR / "trade_plan.csv"
EXECUTION_LOG = LIVE_DIR / "execution_log.csv"

IB_HOST = "127.0.0.1"
IB_PORT = 4001
IB_CLIENT_ID = 2  # Different from reader script

TRAILING_STOP_PCT = 10  # 10% trailing stop


# ── Read trade plan ──────────────────────────────────────────


def read_trade_plan(plan_path):
    """Read trade plan CSV, skipping comment lines."""
    trades = []
    with open(plan_path, "r") as f:
        # Skip comment lines
        lines = [
            line for line in f
            if not line.startswith("#") and line.strip()
        ]

    reader = csv.DictReader(lines)
    for row in reader:
        row["shares"] = int(row["shares"])
        row["price"] = float(row["price"])
        row["est_value"] = float(row["est_value"])
        row["instruction"] = row.get("instruction", "").strip()
        trades.append(row)

    return trades


# ── Claude interpretation ─────────────────────────────────────


def interpret_with_claude(trades):
    """Use Claude to interpret custom instructions.

    Returns a list of interpreted trades with final action details.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  No ANTHROPIC_API_KEY set.")
        print("  Falling back to keyword parsing.\n")
        return interpret_without_claude(trades)

    # Only send trades with custom instructions to Claude
    custom = [
        t for t in trades
        if t["instruction"].upper() not in ("APPROVE", "SKIP", "")
    ]

    if not custom:
        return interpret_without_claude(trades)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(f"  Claude API error: {e}")
        print("  Falling back to keyword parsing.\n")
        return interpret_without_claude(trades)

    # Build the prompt
    trade_lines = []
    for t in custom:
        trade_lines.append(
            f"  - {t['action']} {t['shares']} shares of "
            f"{t['ticker']} @ ${t['price']:.2f} | "
            f"Reason: {t['reason']} | "
            f"User instruction: \"{t['instruction']}\""
        )

    prompt = f"""You are a trade execution assistant. The user has a trade plan with custom instructions that need to be interpreted into concrete trade parameters.

For each trade below, interpret the user's instruction and return a JSON array of trade objects.

Each trade object must have exactly these fields:
- "ticker": string (the stock ticker)
- "action": "BUY" | "SELL" | "BUY_TO_COVER" | "SKIP"
- "order_type": "MARKET" | "LIMIT" | "STOP"
- "shares": integer (number of shares)
- "limit_price": number or null (for LIMIT orders)
- "stop_price": number or null (for STOP orders)
- "note": string (brief explanation of interpretation)

Trades to interpret:
{chr(10).join(trade_lines)}

Rules:
- If the user says "skip", set action to "SKIP"
- If the user specifies a number of shares, use that number
- If the user mentions a limit price, use LIMIT order type
- If the user says "reduce", reduce the shares accordingly
- If the user says "only if price drops below X", use LIMIT order at X
- Default to MARKET order if no order type specified
- Be conservative: if the instruction is ambiguous, default to SKIP with a note explaining why

Return ONLY the JSON array, no other text."""

    print("  Sending custom instructions to Claude...")

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        interpreted = json.loads(response_text)
        print(f"  Claude interpreted {len(interpreted)} trade(s).\n")

        # Build a map for lookup
        claude_map = {}
        for item in interpreted:
            claude_map[item["ticker"]] = item

    except Exception as e:
        print(f"  Claude interpretation failed: {e}")
        print("  Falling back to keyword parsing.\n")
        return interpret_without_claude(trades)

    # Merge Claude interpretations with simple ones
    result = []
    for t in trades:
        instr = t["instruction"].upper()

        if instr == "SKIP" or instr == "":
            continue

        if instr == "APPROVE":
            result.append({
                "ticker": t["ticker"],
                "action": t["action"],
                "order_type": "MARKET",
                "shares": t["shares"],
                "limit_price": None,
                "stop_price": None,
                "note": "Approved as-is",
            })
        elif t["ticker"] in claude_map:
            ci = claude_map[t["ticker"]]
            if ci["action"] != "SKIP":
                result.append(ci)
            else:
                print(f"  Skipping {t['ticker']}: {ci.get('note', '')}")
        else:
            # Custom instruction but not sent to Claude (shouldn't happen)
            result.append({
                "ticker": t["ticker"],
                "action": t["action"],
                "order_type": "MARKET",
                "shares": t["shares"],
                "limit_price": None,
                "stop_price": None,
                "note": f"Uninterpreted: {t['instruction']}",
            })

    return result


def interpret_without_claude(trades):
    """Simple keyword-based interpretation (no Claude API)."""
    result = []
    for t in trades:
        instr = t["instruction"].strip().upper()

        if instr == "SKIP" or instr == "":
            continue

        if instr == "APPROVE":
            result.append({
                "ticker": t["ticker"],
                "action": t["action"],
                "order_type": "MARKET",
                "shares": t["shares"],
                "limit_price": None,
                "stop_price": None,
                "note": "Approved as-is",
            })
            continue

        # Try basic keyword parsing
        instr_lower = t["instruction"].lower()

        # "reduce to N shares"
        if "reduce" in instr_lower:
            import re
            match = re.search(r"(\d+)\s*shares?", instr_lower)
            if match:
                result.append({
                    "ticker": t["ticker"],
                    "action": t["action"],
                    "order_type": "MARKET",
                    "shares": int(match.group(1)),
                    "limit_price": None,
                    "stop_price": None,
                    "note": f"Reduced to {match.group(1)} shares",
                })
                continue

        # "limit at $X" or "limit $X"
        if "limit" in instr_lower:
            import re
            match = re.search(r"\$?([\d.]+)", instr_lower)
            if match:
                result.append({
                    "ticker": t["ticker"],
                    "action": t["action"],
                    "order_type": "LIMIT",
                    "shares": t["shares"],
                    "limit_price": float(match.group(1)),
                    "stop_price": None,
                    "note": f"Limit order at ${match.group(1)}",
                })
                continue

        # Unrecognized - warn and skip
        print(
            f"  WARNING: Cannot parse instruction for "
            f"{t['ticker']}: \"{t['instruction']}\""
        )
        print(
            f"           Set ANTHROPIC_API_KEY for Claude "
            f"interpretation, or use APPROVE/SKIP."
        )
        print(f"           Skipping this trade.\n")

    return result


# ── IB execution ──────────────────────────────────────────────


def connect_ib_rw():
    """Connect to IB Gateway in read-write mode for trading."""
    ib = IB()
    print(f"Connecting to IB Gateway ({IB_HOST}:{IB_PORT}) "
          f"for trading...")
    try:
        ib.connect(
            IB_HOST, IB_PORT,
            clientId=IB_CLIENT_ID,
            readonly=False,  # Need write access for orders
            timeout=10,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    accts = ib.managedAccounts()
    print(f"Connected. Account: {accts[0]}\n")
    return ib


def execute_trade(ib, trade):
    """Execute a single trade on IB. Returns execution result dict."""
    ticker = trade["ticker"]
    action = trade["action"]
    shares = trade["shares"]
    order_type = trade.get("order_type", "MARKET")

    # Map action to IB action string
    if action in ("BUY", "BUY_TO_COVER"):
        ib_action = "BUY"
    elif action == "SELL":
        ib_action = "SELL"
    else:
        return {"ticker": ticker, "status": "SKIPPED",
                "message": f"Unknown action: {action}"}

    # Create contract
    contract = Stock(ticker, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
        if not contract.conId:
            return {"ticker": ticker, "status": "FAILED",
                    "message": "Contract not found"}
    except Exception as e:
        return {"ticker": ticker, "status": "FAILED",
                "message": str(e)}

    # Create order
    if order_type == "LIMIT":
        limit_price = trade.get("limit_price")
        if not limit_price:
            return {"ticker": ticker, "status": "FAILED",
                    "message": "LIMIT order missing price"}
        order = LimitOrder(ib_action, shares, limit_price)
    elif order_type == "STOP":
        stop_price = trade.get("stop_price")
        if not stop_price:
            return {"ticker": ticker, "status": "FAILED",
                    "message": "STOP order missing price"}
        order = StopOrder(ib_action, shares, stop_price)
    else:
        order = MarketOrder(ib_action, shares)

    # Place order
    try:
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(2)  # Wait for acknowledgement

        status = trade_obj.orderStatus.status
        fill_price = trade_obj.orderStatus.avgFillPrice

        return {
            "ticker": ticker,
            "status": status,
            "order_id": trade_obj.order.orderId,
            "fill_price": fill_price,
            "message": f"{ib_action} {shares} {ticker} "
                       f"@ {order_type}",
        }

    except Exception as e:
        return {"ticker": ticker, "status": "ERROR",
                "message": str(e)}


def place_trailing_stop(ib, ticker, shares, entry_price):
    """Place a trailing stop order after a BUY fill.

    Uses IB's TRAIL order type with a percentage-based trailing amount.
    The stop price automatically ratchets up as the market price rises,
    locking in gains while protecting against drawdowns.
    """
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    order = Order()
    order.action = "SELL"
    order.totalQuantity = shares
    order.orderType = "TRAIL"
    order.trailingPercent = TRAILING_STOP_PCT
    order.tif = "GTC"

    initial_stop = round(entry_price * (1 - TRAILING_STOP_PCT / 100), 2)

    try:
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(1)
        return {
            "ticker": ticker,
            "type": "TRAILING_STOP",
            "trailing_pct": TRAILING_STOP_PCT,
            "initial_stop": initial_stop,
            "status": trade_obj.orderStatus.status,
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "type": "TRAILING_STOP",
            "status": "ERROR",
            "message": str(e),
        }


# ── Logging ───────────────────────────────────────────────────


def log_executions(results):
    """Append execution results to log file."""
    file_exists = EXECUTION_LOG.exists()
    ts = datetime.now().isoformat()

    with open(EXECUTION_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "ticker", "status",
                "order_id", "fill_price", "message",
            ])
        for r in results:
            writer.writerow([
                ts,
                r.get("ticker", ""),
                r.get("status", ""),
                r.get("order_id", ""),
                r.get("fill_price", ""),
                r.get("message", ""),
            ])

    print(f"\n  Execution log: {EXECUTION_LOG.name}")


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Execute trades from the trade plan",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would execute without trading",
    )
    parser.add_argument(
        "--plan", type=str, default=str(TRADE_PLAN_FILE),
        help="Path to trade plan CSV",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"Trade plan not found: {plan_path}")
        print("Run ib_update_and_recommend.py first.")
        sys.exit(1)

    # Read the plan
    print(f"Reading trade plan: {plan_path.name}")
    raw_trades = read_trade_plan(plan_path)
    print(f"  {len(raw_trades)} trade(s) in plan.\n")

    if not raw_trades:
        print("No trades in plan. Nothing to do.")
        return

    # Show raw plan
    print("=" * 70)
    print("  TRADE PLAN")
    print("=" * 70)
    for t in raw_trades:
        instr = t["instruction"] or "(blank=skip)"
        print(
            f"  {t['action']:<14} {t['ticker']:<6} "
            f"{t['shares']:>5} sh  @ ${t['price']:>8.2f}  "
            f"| {instr}"
        )

    # Interpret instructions
    print("\n" + "=" * 70)
    print("  INTERPRETING INSTRUCTIONS")
    print("=" * 70)
    final_trades = interpret_with_claude(raw_trades)

    if not final_trades:
        print("\n  No trades to execute after interpretation.")
        return

    # Show final execution plan
    print("=" * 70)
    print("  EXECUTION PLAN")
    print("=" * 70)
    total_buy = 0
    total_sell = 0

    for t in final_trades:
        lp = (f"  limit=${t['limit_price']}"
              if t.get("limit_price") else "")
        sp = (f"  stop=${t['stop_price']}"
              if t.get("stop_price") else "")
        print(
            f"  {t['action']:<14} {t['ticker']:<6} "
            f"{t['shares']:>5} sh  "
            f"{t['order_type']:<7}{lp}{sp}"
            f"  ({t['note']})"
        )
        est = t["shares"] * (
            t.get("limit_price") or t.get("stop_price") or 0
        )
        if "BUY" in t["action"]:
            total_buy += est
        else:
            total_sell += est

    print(f"\n  {len(final_trades)} trade(s) to execute.")

    if args.dry_run:
        print("\n  DRY RUN - no trades placed.")
        return

    # Confirm
    print("\n" + "=" * 70)
    print("  CONFIRM EXECUTION")
    print("=" * 70)
    print("  THIS WILL PLACE REAL ORDERS ON YOUR IB ACCOUNT.")
    confirm = input("\n  Type 'YES' to execute: ").strip()
    if confirm != "YES":
        print("  Cancelled.")
        return

    # Connect and execute
    ib = connect_ib_rw()
    results = []

    try:
        for t in final_trades:
            if t["action"] == "SKIP":
                continue

            print(f"\n  Executing: {t['action']} {t['shares']} "
                  f"{t['ticker']} ({t['order_type']})...")

            result = execute_trade(ib, t)
            results.append(result)

            status = result.get("status", "UNKNOWN")
            print(f"    -> {status}: {result.get('message', '')}")

            # Auto-place trailing stop for BUY orders
            if (t["action"] == "BUY"
                    and status in ("Filled", "Submitted", "PreSubmitted")
                    and t["order_type"] == "MARKET"):
                fill = result.get("fill_price", 0)
                if fill and fill > 0:
                    initial_stop = fill * (1 - TRAILING_STOP_PCT / 100)
                    print(f"    Placing {TRAILING_STOP_PCT}% trailing stop "
                          f"(initial stop ~${initial_stop:.2f})...")
                    sl_result = place_trailing_stop(
                        ib, t["ticker"], t["shares"], fill
                    )
                    results.append(sl_result)
                    print(f"    -> Trailing stop: "
                          f"{sl_result.get('status', 'UNKNOWN')}")

        # Log results
        log_executions(results)

        # Summary
        print("\n" + "=" * 70)
        print("  EXECUTION SUMMARY")
        print("=" * 70)
        filled = sum(
            1 for r in results
            if r.get("status") in ("Filled", "Submitted", "PreSubmitted")
        )
        failed = sum(
            1 for r in results if r.get("status") in ("ERROR", "FAILED")
        )
        print(f"  Executed: {filled}")
        print(f"  Failed:   {failed}")
        print(f"  Total:    {len(results)}")

    finally:
        ib.disconnect()
        print("\nDisconnected from IB Gateway.")


if __name__ == "__main__":
    main()
