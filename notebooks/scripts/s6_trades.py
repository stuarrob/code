"""
Step 6: Trade Recommendations

Connects to IB, pulls live positions, compares vs target portfolio,
generates trade plan.

Three-phase workflow (when called from notebook):
  1. cleanup_account() — cancel orphan orders, cover shorts
  2. generate_trades() — compare target vs live, produce trade plan
  3. write_portfolio_state() — snapshot for execution notebook

Two sizing modes:
  - deploy_cash=None: sizes to full NLV, caps buys by cash_reserve
  - deploy_cash=X:    sizes to (invested + X), caps buys by X
"""

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


CASH_RESERVE = 70_000


def cleanup_account(
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 6,
) -> dict:
    """Cancel all open orders and cover any short positions.

    Puts the account in a clean state before generating a trade plan.

    Returns:
        dict with cleanup results including positions/cash after cleanup.
    """
    from ib_insync import IB, Stock, MarketOrder

    ib = IB()
    try:
        ib.connect(
            ib_host, ib_port,
            clientId=ib_client_id,
            readonly=False, timeout=10,
        )
        account = ib.managedAccounts()[0]
        print(f"Connected for cleanup: {account}")

        # Phase 1: Cancel ALL open orders
        open_trades = ib.openTrades()
        orders_cancelled = 0
        if open_trades:
            print(f"\n  Cancelling {len(open_trades)} open orders...")
            for t in open_trades:
                sym = t.contract.symbol
                action = t.order.action
                qty = int(t.order.totalQuantity)
                otype = t.order.orderType
                print(
                    f"    CANCEL: {action} {qty} {sym}"
                    f" ({otype} {t.order.tif})"
                )
                ib.cancelOrder(t.order)
                orders_cancelled += 1
            ib.sleep(3)
            print(f"  Cancelled {orders_cancelled} orders")
        else:
            print("  No open orders to cancel")

        # Phase 2: Cover ALL short positions
        shorts_covered = []
        positions = ib.positions()
        for p in positions:
            shares = float(p.position)
            if shares < 0:
                sym = p.contract.symbol
                cover_qty = int(abs(shares))
                print(
                    f"\n  Covering short: BUY {cover_qty}"
                    f" {sym} (currently {shares:.0f})"
                )
                contract = Stock(sym, "SMART", "USD")
                try:
                    ib.qualifyContracts(contract)
                    order = MarketOrder("BUY", cover_qty)
                    trade = ib.placeOrder(contract, order)
                    ib.sleep(3)
                    status = trade.orderStatus.status
                    fill = trade.orderStatus.avgFillPrice
                    print(
                        f"    {status}"
                        + (f" @ ${fill:.2f}" if fill else "")
                    )
                    shorts_covered.append({
                        "ticker": sym,
                        "shares": cover_qty,
                        "action": "BUY_TO_COVER",
                        "status": status,
                        "fill_price": fill,
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
                    shorts_covered.append({
                        "ticker": sym,
                        "shares": cover_qty,
                        "action": "BUY_TO_COVER",
                        "status": "FAILED",
                        "fill_price": 0,
                    })

        # Phase 3: Re-read clean state
        ib.sleep(2)
        summary = {}
        for av in ib.accountSummary():
            if av.currency == "USD":
                try:
                    summary[av.tag] = float(av.value)
                except (ValueError, TypeError):
                    pass

        positions_after = {}
        for p in ib.positions():
            shares = float(p.position)
            if shares != 0:
                positions_after[p.contract.symbol] = {
                    "shares": shares,
                    "avg_cost": round(float(p.avgCost), 2),
                }

        remaining_orders = len(ib.openTrades())
        short_count = sum(
            1 for v in positions_after.values()
            if v["shares"] < 0
        )

        cleanup_at = datetime.now().isoformat()

        print(f"\n{'─' * 50}")
        print("CLEANUP COMPLETE")
        print(f"  Orders cancelled: {orders_cancelled}")
        print(f"  Shorts covered:   {len(shorts_covered)}")
        print(f"  Remaining orders: {remaining_orders}")
        print(f"  Remaining shorts: {short_count}")
        print(
            f"  Cash after:       "
            f"${summary.get('TotalCashValue', 0):,.0f}"
        )
        print(
            f"  NLV after:        "
            f"${summary.get('NetLiquidation', 0):,.0f}"
        )
        print(f"{'─' * 50}")

        if remaining_orders > 0:
            print("  WARNING: Some orders could not be cancelled")
        if short_count > 0:
            print("  WARNING: Some shorts could not be covered")

        return {
            "orders_cancelled": orders_cancelled,
            "shorts_covered": shorts_covered,
            "positions_after": positions_after,
            "cash_after": summary.get("TotalCashValue", 0),
            "nlv_after": summary.get("NetLiquidation", 0),
            "open_orders_after": remaining_orders,
            "short_count_after": short_count,
            "cleanup_at": cleanup_at,
            "account": account,
        }

    finally:
        ib.disconnect()


def write_portfolio_state(
    cleanup_result: dict,
    trades: list,
    target_weights: "pd.Series",
    live_dir: Path,
    deploy_cash: float = None,
    sizing_basis: float = None,
    ib_positions: list = None,
    live_prices: dict = None,
    cash: float = None,
    nlv: float = None,
    account: str = None,
) -> Path:
    """Write portfolio_state.json as the contract between notebooks.

    Returns:
        Path to the written state file.
    """
    # Build pre-trade positions from cleanup result
    pre_positions = {}
    for ticker, info in cleanup_result["positions_after"].items():
        price = (
            live_prices.get(ticker, info["avg_cost"])
            if live_prices else info["avg_cost"]
        )
        pre_positions[ticker] = {
            "shares": info["shares"],
            "price": round(price, 2),
        }

    # Compute expected post-trade positions
    expected_positions = {}
    # Start with current positions
    for ticker, info in pre_positions.items():
        expected_positions[ticker] = {
            "shares": info["shares"],
            "price": info["price"],
        }
    # Apply trades
    for t in trades:
        ticker = t["ticker"]
        shares = t["shares"]
        price = t.get("price", 0)
        if t["action"] == "SELL":
            if ticker in expected_positions:
                remaining = (
                    expected_positions[ticker]["shares"] - shares
                )
                if remaining <= 0:
                    expected_positions.pop(ticker, None)
                else:
                    expected_positions[ticker]["shares"] = remaining
        elif t["action"] in ("BUY", "BUY_TO_COVER"):
            if ticker in expected_positions:
                expected_positions[ticker]["shares"] += shares
            else:
                expected_positions[ticker] = {
                    "shares": shares,
                    "price": price,
                }

    # Remove zero-share positions
    expected_positions = {
        k: v for k, v in expected_positions.items()
        if v["shares"] != 0
    }

    # Compute expected cash
    total_buys = sum(
        t["est_value"] for t in trades
        if t["action"] in ("BUY", "BUY_TO_COVER")
    )
    total_sells = sum(
        t["est_value"] for t in trades
        if t["action"] == "SELL"
    )
    estimated_cash = (cash or 0) + total_sells - total_buys

    # Compute checksums
    checksums = {}
    trade_plan_file = live_dir / "trade_plan.csv"
    target_file = live_dir / "target_portfolio_latest.csv"
    for label, fpath in [
        ("trade_plan_csv", trade_plan_file),
        ("target_portfolio_csv", target_file),
    ]:
        if fpath.exists():
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            checksums[label] = f"sha256:{h}"

    # Build trade summary for state file
    trade_summary = []
    for t in trades:
        trade_summary.append({
            "action": t["action"],
            "ticker": t["ticker"],
            "shares": t["shares"],
            "price": t.get("price", 0),
            "est_value": t.get("est_value", 0),
        })

    state = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "account": account or cleanup_result.get("account", ""),
        "cleanup_performed": {
            "orders_cancelled": cleanup_result["orders_cancelled"],
            "shorts_covered": cleanup_result["shorts_covered"],
            "cleanup_at": cleanup_result["cleanup_at"],
        },
        "pre_trade_state": {
            "positions": pre_positions,
            "cash": round(cash or 0, 2),
            "nlv": round(nlv or 0, 2),
            "open_orders": cleanup_result["open_orders_after"],
        },
        "trade_plan": {
            "file": "trade_plan.csv",
            "deploy_cash": deploy_cash,
            "sizing_basis": round(sizing_basis or 0, 2),
            "trades": trade_summary,
            "total_buys": round(total_buys, 2),
            "total_sells": round(total_sells, 2),
        },
        "expected_post_trade": {
            "positions": expected_positions,
            "estimated_cash": round(estimated_cash, 2),
        },
        "target_weights": (
            target_weights.round(6).to_dict()
            if target_weights is not None else {}
        ),
        "checksums": checksums,
    }

    state_file = live_dir / "portfolio_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)

    print(f"Portfolio state written: {state_file}")
    print(f"  Pre-trade positions:  {len(pre_positions)}")
    print(f"  Planned trades:       {len(trades)}")
    print(f"  Expected positions:   {len(expected_positions)}")
    print(f"  Expected cash:        ${estimated_cash:,.0f}")

    return state_file


def generate_trades(
    target_weights: pd.Series,
    live_dir: Path,
    combined_scores: pd.Series = None,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 5,
    cash_reserve: float = CASH_RESERVE,
    drift_rebalance_pct: float = 0.03,
    deploy_cash: float = None,
    return_context: bool = False,
):
    """Generate trade recommendations.

    Compares target vs live positions.
    Saves trade plan to live_dir/trade_plan.csv.

    Args:
        target_weights: Target portfolio weights.
        live_dir: Directory for trade plan output.
        combined_scores: Factor scores (for explaining selections).
        ib_host: IB Gateway host.
        ib_port: IB Gateway port.
        ib_client_id: IB client ID.
        cash_reserve: Minimum cash to keep (used when deploy_cash is None).
        drift_rebalance_pct: Drift threshold for rebalancing.
        deploy_cash: Explicit cash amount to deploy. When set, position
            sizing uses (current invested + deploy_cash) as the portfolio
            basis and buy spending is capped at deploy_cash. Overrides
            cash_reserve.
        return_context: If True, return (trades, context_dict) tuple.

    Returns:
        list of trade dicts, or (list, dict) if return_context=True.
    """
    from ib_insync import IB, Stock

    ib = IB()
    try:
        ib.connect(
            ib_host, ib_port,
            clientId=ib_client_id,
            readonly=True, timeout=10,
        )
        account_name = ib.managedAccounts()[0]
        print(f"Connected: {account_name}")
    except Exception as e:
        print(f"IB not available: {e}")
        print("Cannot generate trade recommendations without IB.")
        if return_context:
            return [], {}
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
        {
            "ticker": p.contract.symbol,
            "shares": float(p.position),
            "avg_cost": float(p.avgCost),
        }
        for p in ib.positions()
    ]

    # Live prices
    all_tickers = sorted(
        set(p["ticker"] for p in ib_positions)
        | set(target_weights.index)
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
    snaps = [
        (c.symbol, ib.reqMktData(c, snapshot=True))
        for c in contracts
    ]
    ib.sleep(4)
    for sym, td in snaps:
        for attr in ("last", "close", "bid", "ask"):
            v = getattr(td, attr, None)
            if v is not None and v == v and v > 0:
                live_prices[sym] = float(v)
                break
        ib.cancelMktData(td.contract)
    ib.reqMarketDataType(1)

    # Compute current invested value (for deploy_cash mode)
    invested_value = sum(
        p["shares"] * live_prices.get(p["ticker"], p["avg_cost"])
        for p in ib_positions
        if p["shares"] > 0
    )

    # Determine sizing basis and deployable cash
    if deploy_cash is not None:
        sizing_basis = invested_value + deploy_cash
        deployable = deploy_cash
        print(
            f"\nAccount: NLV=${nlv:,.0f}  Cash=${cash:,.0f}  "
            f"Invested=${invested_value:,.0f}"
        )
        print(
            f"Deploy mode: deploying ${deploy_cash:,.0f} additional cash"
        )
        print(
            f"Sizing basis: ${sizing_basis:,.0f} "
            f"(invested + deploy_cash)"
        )
    else:
        sizing_basis = nlv
        deployable = max(0, cash - cash_reserve)
        print(
            f"\nAccount: NLV=${nlv:,.0f}  Cash=${cash:,.0f}  "
            f"Reserve=${cash_reserve:,}  Deployable=${deployable:,.0f}"
        )
    print(
        f"Positions: {len(ib_positions)}  |  "
        f"Live prices: {len(live_prices)}"
    )

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
        aw = mv / sizing_basis if sizing_basis else 0
        ib_map[t] = pos
        pos_data.append({
            "ticker": t, "shares": pos["shares"],
            "price": price, "mkt_value": mv,
            "target_w": tw, "actual_w": aw,
            "drift": aw - tw, "in_target": t in targets,
        })

    # Sells: exit non-targets
    special = {"IBKR"}
    for p in pos_data:
        if not p["in_target"] and p["ticker"] not in special:
            action = (
                "SELL" if p["shares"] > 0
                else "BUY_TO_COVER"
            )
            trades.append({
                "action": action,
                "ticker": p["ticker"],
                "shares": int(abs(p["shares"])),
                "price": round(p["price"], 2),
                "est_value": round(abs(p["mkt_value"]), 2),
                "reason": "Not in strategy target",
                "instruction": "APPROVE",
            })

    # Cash available after sells
    sell_proceeds = sum(
        t["est_value"] for t in trades
        if t["action"] == "SELL"
    )
    cover_cost = sum(
        t["est_value"] for t in trades
        if t["action"] == "BUY_TO_COVER"
    )
    if deploy_cash is not None:
        available = max(
            0, deploy_cash + sell_proceeds - cover_cost
        )
    else:
        available = max(
            0, cash + sell_proceeds - cover_cost - cash_reserve
        )

    # Buys: new positions + rebalance drifted
    buy_candidates = []
    for ticker, tw in targets.items():
        if ticker not in ib_map:
            price = live_prices.get(ticker)
            if price and price > 0:
                sh = int(tw * sizing_basis / price)
                if sh > 0:
                    pct = tw * 100
                    buy_candidates.append({
                        "action": "BUY",
                        "ticker": ticker,
                        "shares": sh,
                        "price": round(price, 2),
                        "est_value": round(sh * price, 2),
                        "reason": f"New position (target {pct:.0f}%)",
                        "instruction": "APPROVE",
                    })

    for p in pos_data:
        drift_pct = p["drift"] * 100
        if p["target_w"] > 0 and abs(p["drift"]) > drift_rebalance_pct:
            diff = p["target_w"] * sizing_basis - p["mkt_value"]
            ds = int(abs(diff) / p["price"])
            if ds > 0:
                if diff > 0:
                    buy_candidates.append({
                        "action": "BUY",
                        "ticker": p["ticker"],
                        "shares": ds,
                        "price": round(p["price"], 2),
                        "est_value": round(ds * p["price"], 2),
                        "reason": f"Rebalance (drift {drift_pct:+.1f}%)",
                        "instruction": "APPROVE",
                    })
                else:
                    trades.append({
                        "action": "SELL",
                        "ticker": p["ticker"],
                        "shares": ds,
                        "price": round(p["price"], 2),
                        "est_value": round(ds * p["price"], 2),
                        "reason": f"Rebalance (drift {drift_pct:+.1f}%)",
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
                    bc["est_value"] = round(
                        reduced * bc["price"], 2
                    )
                    if deploy_cash is not None:
                        bc["reason"] += (
                            f" [capped by ${deploy_cash:,.0f} budget]"
                        )
                    else:
                        bc["reason"] += (
                            f" [capped by ${cash_reserve:,} reserve]"
                        )
                    trades.append(bc)
                    running_spend += bc["est_value"]

    trades.sort(
        key=lambda t: (
            0 if "SELL" in t["action"] else 1,
            t["ticker"],
        )
    )

    # Archive previous plan and save new one
    trade_plan_file = live_dir / "trade_plan.csv"
    archive_dir = live_dir / "trade_plan_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if trade_plan_file.exists():
        mtime = datetime.fromtimestamp(
            trade_plan_file.stat().st_mtime
        )
        ts_str = mtime.strftime("%Y%m%d_%H%M%S")
        dest = archive_dir / f"trade_plan_{ts_str}.csv"
        shutil.copy2(trade_plan_file, dest)

    # Resolve ETF names
    try:
        from utils.etf_names import lookup_names
        all_trade_tickers = [t["ticker"] for t in trades]
        names = lookup_names(all_trade_tickers, use_yfinance=True)
    except ImportError:
        names = {}

    # Write trade plan with names
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fieldnames = [
        "action", "ticker", "name", "shares",
        "price", "est_value", "reason", "instruction",
    ]
    with open(trade_plan_file, "w", newline="") as f:
        f.write(f"# TRADE PLAN - Generated {ts}\n#\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in trades:
            row = dict(t)
            row["name"] = names.get(t["ticker"], "")
            w.writerow(row)

    total_buys = sum(
        t["est_value"] for t in trades if "BUY" in t["action"]
    )
    total_sells = sum(
        t["est_value"] for t in trades if t["action"] == "SELL"
    )
    cash_after = cash + total_sells - total_buys
    print(f"\nTrade plan: {len(trades)} trades")
    print(f"  Buys: ${total_buys:,.0f}  |  Sells: ${total_sells:,.0f}")
    if deploy_cash is not None:
        print(
            f"  Cash after: ${cash_after:,.0f} "
            f"(deployed: ${deploy_cash:,.0f})"
        )
    else:
        print(
            f"  Cash after: ${cash_after:,.0f} "
            f"(reserve: ${cash_reserve:,})"
        )
    print(f"Saved: {trade_plan_file}")

    # Print human-readable trade summary
    _print_trade_summary(
        trades, names, pos_data,
        target_weights, combined_scores,
    )

    ib.disconnect()

    if return_context:
        context = {
            "ib_positions": ib_positions,
            "live_prices": live_prices,
            "cash": cash,
            "nlv": nlv,
            "sizing_basis": sizing_basis,
            "account": account_name,
        }
        return trades, context
    return trades


def _print_trade_summary(
    trades, names, pos_data,
    target_weights, combined_scores,
):
    """Print a clear summary explaining each trade."""
    sells = [t for t in trades if "SELL" in t["action"]]
    buys = [t for t in trades if t["action"] == "BUY"]
    covers = [
        t for t in trades if t["action"] == "BUY_TO_COVER"
    ]

    # Count how many current holdings are retained
    held_tickers = set(p["ticker"] for p in pos_data)
    target_tickers = set(target_weights.index)
    retained = held_tickers & target_tickers
    exiting = held_tickers - target_tickers - {"IBKR"}
    entering = target_tickers - held_tickers

    print(f"\n{'─' * 70}")
    print("PORTFOLIO TRANSITION SUMMARY")
    print(f"{'─' * 70}")
    print(
        f"  Retained:  {len(retained)} positions "
        f"(already held & still in target)"
    )
    print(
        f"  Exiting:   {len(exiting)} positions "
        f"(held but no longer in target)"
    )
    print(
        f"  Entering:  {len(entering)} positions "
        f"(new to portfolio)"
    )

    if len(exiting) > len(retained):
        print(
            "\n  WHY SO MANY SELLS? The factor model re-ranks"
            " the full"
        )
        print(
            "  universe each run. When the universe expands or"
            " scores"
        )
        print(
            "  shift, the top-N can change significantly. This"
            " is normal"
        )
        print(
            "  for a momentum-driven strategy — it follows"
            " where the"
        )
        print("  strongest factor signals are today.")

    if sells:
        print(f"\n  SELLS ({len(sells)}):")
        for t in sells:
            name = names.get(t["ticker"], "")
            nm = name[:35] if len(name) > 35 else name
            print(
                f"    {t['ticker']:<6} ${t['est_value']:>10,.0f}"
                f"  {nm}"
            )

    if covers:
        print(f"\n  COVERS ({len(covers)}):")
        for t in covers:
            name = names.get(t["ticker"], "")
            nm = name[:35] if len(name) > 35 else name
            print(
                f"    {t['ticker']:<6} ${t['est_value']:>10,.0f}"
                f"  {nm}"
            )

    if buys:
        print(f"\n  BUYS ({len(buys)}):")
        for t in buys:
            name = names.get(t["ticker"], "")
            nm = name[:35] if len(name) > 35 else name
            score = ""
            if combined_scores is not None:
                s = combined_scores.get(t["ticker"])
                if s is not None:
                    rank = combined_scores.rank(pct=True).get(
                        t["ticker"], 0
                    )
                    score = f"  (score rank: {rank:.0%})"
            print(
                f"    {t['ticker']:<6} ${t['est_value']:>10,.0f}"
                f"  {nm}{score}"
            )

    print(f"{'─' * 70}")
