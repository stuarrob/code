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
                "limit_price": None, "note": "Approved",
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
                        "limit_price": None,
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
) -> list:
    """Execute trades on IB Gateway.

    Places trailing stops on ALL BUY fills automatically.

    Args:
        final_trades: List of interpreted trade dicts.
        live_dir: Directory for execution log.
        confirm: Must be True to place real orders.
        trailing_stop_pct: Trailing stop percentage (default 10%).

    Returns:
        List of execution result dicts.
    """
    if not confirm:
        print("DRY RUN — CONFIRM=False. No orders placed.")
        print(f"\nWould execute {len(final_trades)} trades:")
        for t in final_trades:
            print(f"  {t['action']:>5} {t['shares']:>5} {t['ticker']:<6} "
                  f"({t['order_type']})")
        print(f"\nTrailing stop: {trailing_stop_pct}% on all BUY fills")
        return []

    from ib_insync import IB, Stock, Order, LimitOrder, MarketOrder

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=False, timeout=10)
    print(f"Connected for trading: {ib.managedAccounts()[0]}\n")

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

        if order_type == "LIMIT" and t.get("limit_price"):
            order = LimitOrder(ib_action, shares, t["limit_price"])
        else:
            order = MarketOrder(ib_action, shares)

        print(f"  {ib_action} {shares} {ticker} ({order_type})...", end="")
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(2)

        status = trade_obj.orderStatus.status
        fill = trade_obj.orderStatus.avgFillPrice
        print(f" {status}" + (f" @ ${fill:.2f}" if fill else ""))

        exec_results.append({
            "ticker": ticker, "status": status,
            "order_id": trade_obj.order.orderId, "fill_price": fill,
        })

        # TRAILING STOP on every BUY fill
        if (t["action"] == "BUY"
                and status in ("Filled", "Submitted", "PreSubmitted")
                and fill and fill > 0):
            ts_order = Order()
            ts_order.action = "SELL"
            ts_order.totalQuantity = shares
            ts_order.orderType = "TRAIL"
            ts_order.trailingPercent = trailing_stop_pct
            ts_order.tif = "GTC"
            ts_trade = ib.placeOrder(contract, ts_order)
            ib.sleep(1)
            init_stop = fill * (1 - trailing_stop_pct / 100)
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
            w.writerow(["timestamp", "ticker", "status", "order_id", "fill_price"])
        for r in exec_results:
            w.writerow([log_ts, r.get("ticker"), r.get("status"),
                        r.get("order_id", ""), r.get("fill_price", "")])

    filled = sum(1 for r in exec_results
                 if r["status"] in ("Filled", "Submitted", "PreSubmitted"))
    failed = sum(1 for r in exec_results
                 if r["status"] in ("ERROR", "FAILED"))
    buy_count = sum(1 for t in final_trades if t["action"] == "BUY")

    print(f"\nExecution summary:")
    print(f"  Executed: {filled}  |  Failed: {failed}  |  Total: {len(exec_results)}")
    print(f"  Trailing stops placed: {trailing_stops_placed}/{buy_count} buys")

    ib.disconnect()
    return exec_results
