"""
Step 6: Trade Recommendations

Connects to IB, pulls live positions, compares vs target portfolio,
generates trade plan with $70k cash reserve enforcement.
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


CASH_RESERVE = 70_000


def generate_trades(
    target_weights: pd.Series,
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 5,
    cash_reserve: float = CASH_RESERVE,
    drift_rebalance_pct: float = 0.03,
) -> list:
    """Generate trade recommendations by comparing target vs live positions.

    Saves trade plan to live_dir/trade_plan.csv.

    Returns list of trade dicts.
    """
    from ib_insync import IB, Stock

    ib = IB()
    try:
        ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10)
        print(f"Connected: {ib.managedAccounts()[0]}")
    except Exception as e:
        print(f"IB not available: {e}")
        print("Cannot generate trade recommendations without IB.")
        return []

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

    # Current positions
    ib_positions = [
        {"ticker": p.contract.symbol, "shares": float(p.position),
         "avg_cost": float(p.avgCost)}
        for p in ib.positions()
    ]

    # Live prices
    all_tickers = sorted(
        set(p["ticker"] for p in ib_positions) | set(target_weights.index)
    )
    ib.reqMarketDataType(3)
    live_prices = {}
    contracts = []
    for t in all_tickers:
        c = Stock(t, "SMART", "USD")
        try:
            ib.qualifyContracts(c)
            if c.conId:
                contracts.append(c)
        except Exception:
            pass
    snaps = [(c.symbol, ib.reqMktData(c, snapshot=True)) for c in contracts]
    ib.sleep(4)
    for sym, td in snaps:
        for attr in ("last", "close", "bid", "ask"):
            v = getattr(td, attr, None)
            if v is not None and v == v and v > 0:
                live_prices[sym] = float(v)
                break
        ib.cancelMktData(td.contract)
    ib.reqMarketDataType(1)

    deployable = max(0, cash - cash_reserve)
    print(f"\nAccount: NLV=${nlv:,.0f}  Cash=${cash:,.0f}  "
          f"Reserve=${cash_reserve:,}  Deployable=${deployable:,.0f}")
    print(f"Positions: {len(ib_positions)}  |  Live prices: {len(live_prices)}")

    # Build trades
    trades = []
    targets = target_weights.to_dict()
    ib_map = {}
    pos_data = []

    for pos in ib_positions:
        t = pos["ticker"]
        if pos["shares"] == 0:
            continue
        price = live_prices.get(t, pos["avg_cost"])
        mv = pos["shares"] * price
        tw = targets.get(t, 0)
        aw = mv / nlv if nlv else 0
        ib_map[t] = pos
        pos_data.append({
            "ticker": t, "shares": pos["shares"], "price": price,
            "mkt_value": mv, "target_w": tw, "actual_w": aw,
            "drift": aw - tw, "in_target": t in targets,
        })

    # Sells: exit non-targets
    special = {"IBKR"}
    for p in pos_data:
        if not p["in_target"] and p["ticker"] not in special:
            action = "SELL" if p["shares"] > 0 else "BUY_TO_COVER"
            trades.append({
                "action": action, "ticker": p["ticker"],
                "shares": int(abs(p["shares"])), "price": round(p["price"], 2),
                "est_value": round(abs(p["mkt_value"]), 2),
                "reason": "Not in strategy target", "instruction": "APPROVE",
            })

    # Cash available after sells
    sell_proceeds = sum(t["est_value"] for t in trades if t["action"] == "SELL")
    cover_cost = sum(t["est_value"] for t in trades if t["action"] == "BUY_TO_COVER")
    available = max(0, cash + sell_proceeds - cover_cost - cash_reserve)

    # Buys: new positions + rebalance drifted
    buy_candidates = []
    for ticker, tw in targets.items():
        if ticker not in ib_map:
            price = live_prices.get(ticker)
            if price and price > 0:
                sh = int(tw * nlv / price)
                if sh > 0:
                    buy_candidates.append({
                        "action": "BUY", "ticker": ticker, "shares": sh,
                        "price": round(price, 2), "est_value": round(sh * price, 2),
                        "reason": f"New position (target {tw*100:.0f}%)",
                        "instruction": "APPROVE",
                    })

    for p in pos_data:
        if p["target_w"] > 0 and abs(p["drift"]) > drift_rebalance_pct:
            diff = p["target_w"] * nlv - p["mkt_value"]
            ds = int(abs(diff) / p["price"])
            if ds > 0:
                if diff > 0:
                    buy_candidates.append({
                        "action": "BUY", "ticker": p["ticker"], "shares": ds,
                        "price": round(p["price"], 2),
                        "est_value": round(ds * p["price"], 2),
                        "reason": f"Rebalance (drift {p['drift']*100:+.1f}%)",
                        "instruction": "APPROVE",
                    })
                else:
                    trades.append({
                        "action": "SELL", "ticker": p["ticker"], "shares": ds,
                        "price": round(p["price"], 2),
                        "est_value": round(ds * p["price"], 2),
                        "reason": f"Rebalance (drift {p['drift']*100:+.1f}%)",
                        "instruction": "APPROVE",
                    })

    # Apply cash reserve cap on buys
    running_spend = 0
    for bc in buy_candidates:
        if running_spend + bc["est_value"] <= available:
            trades.append(bc)
            running_spend += bc["est_value"]
        else:
            remaining = available - running_spend
            if remaining > bc["price"]:
                reduced = int(remaining / bc["price"])
                if reduced > 0:
                    bc["shares"] = reduced
                    bc["est_value"] = round(reduced * bc["price"], 2)
                    bc["reason"] += f" [capped by ${cash_reserve:,} reserve]"
                    trades.append(bc)
                    running_spend += bc["est_value"]

    trades.sort(key=lambda t: (0 if "SELL" in t["action"] else 1, t["ticker"]))

    # Archive previous plan and save new one
    trade_plan_file = live_dir / "trade_plan.csv"
    archive_dir = live_dir / "trade_plan_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if trade_plan_file.exists():
        mtime = datetime.fromtimestamp(trade_plan_file.stat().st_mtime)
        dest = archive_dir / f"trade_plan_{mtime.strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy2(trade_plan_file, dest)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(trade_plan_file, "w", newline="") as f:
        f.write(f"# TRADE PLAN - Generated {ts}\n#\n")
        w = csv.DictWriter(
            f,
            fieldnames=["action", "ticker", "shares", "price",
                        "est_value", "reason", "instruction"],
        )
        w.writeheader()
        for t in trades:
            w.writerow(t)

    total_buys = sum(t["est_value"] for t in trades if "BUY" in t["action"])
    total_sells = sum(t["est_value"] for t in trades if t["action"] == "SELL")
    print(f"\nTrade plan: {len(trades)} trades")
    print(f"  Buys: ${total_buys:,.0f}  |  Sells: ${total_sells:,.0f}")
    print(f"  Cash after: ${cash + total_sells - total_buys:,.0f} "
          f"(reserve: ${cash_reserve:,})")
    print(f"Saved: {trade_plan_file}")

    ib.disconnect()
    return trades
