#!/usr/bin/env python3
"""
IB Portfolio Manager - One Stop Shop

Connects to Interactive Brokers, pulls live data, evaluates
performance against strategy targets, and generates trade
recommendations.

Usage:
    # Full review: positions, performance, and trade recs
    python scripts/ib_portfolio_manager.py

    # Review only (no trade generation)
    python scripts/ib_portfolio_manager.py --review

    # Generate and confirm trades
    python scripts/ib_portfolio_manager.py --trades
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd
from ib_insync import IB, Stock

nest_asyncio.apply()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_DIR = PROJECT_ROOT / "data"
TRADING_DIR = Path.home() / "trading"
LIVE_DIR = TRADING_DIR / "live_portfolio"
SNAPSHOT_DIR = LIVE_DIR / "snapshots"

IB_HOST = "127.0.0.1"
IB_PORT = 4001
IB_CLIENT_ID = 1

# Stop-loss parameters
ENTRY_STOP_LOSS_PCT = 0.12
TRAILING_STOP_THRESHOLD = 0.10
TRAILING_STOP_DISTANCE = 0.08

# Cash reserve
CASH_RESERVE = 70_000  # Minimum cash reserve - trades must not breach this


def connect_ib():
    """Connect to IB Gateway and return IB instance."""
    ib = IB()
    print(f"Connecting to IB Gateway ({IB_HOST}:{IB_PORT})...")
    try:
        ib.connect(
            IB_HOST, IB_PORT,
            clientId=IB_CLIENT_ID,
            readonly=True,
            timeout=10,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        print("Is IB Gateway running on port 4001?")
        sys.exit(1)

    acct = ib.managedAccounts()
    print(f"Connected. Account: {acct[0]}\n")
    return ib


def get_account_summary(ib):
    """Pull account summary from IB."""
    vals = ib.accountSummary()
    summary = {}
    for av in vals:
        if av.currency == "USD":
            try:
                summary[av.tag] = float(av.value)
            except (ValueError, TypeError):
                pass
    return summary


def get_positions(ib):
    """Pull positions from IB and return as list of dicts."""
    positions = []
    for pos in ib.positions():
        c = pos.contract
        positions.append({
            "ticker": c.symbol,
            "sec_type": c.secType,
            "shares": float(pos.position),
            "avg_cost": float(pos.avgCost),
        })
    return positions


def get_live_prices(ib, tickers):
    """Fetch live/delayed prices for tickers from IB.

    Requests delayed data (market data type 3) which is free,
    then falls back to snapshot if available.
    """
    prices = {}
    contracts = []
    valid_tickers = []

    # Qualify contracts first to filter out invalid ones
    for t in tickers:
        c = Stock(t, "SMART", "USD")
        contracts.append(c)

    if not contracts:
        return prices

    qualified = []
    for c in contracts:
        try:
            ib.qualifyContracts(c)
            if c.conId:  # Valid contract
                qualified.append(c)
                valid_tickers.append(c.symbol)
        except Exception:
            pass

    if not qualified:
        return prices

    # Request delayed market data (type 3 = delayed, free)
    ib.reqMarketDataType(3)

    snapshots = []
    for c in qualified:
        td = ib.reqMktData(c, snapshot=True)
        snapshots.append((c.symbol, td))

    ib.sleep(4)

    for symbol, td in snapshots:
        # Try multiple price fields in priority order
        # Delayed fields: last/close may arrive as delayed
        price = None
        for attr in ["last", "close", "bid", "ask"]:
            val = getattr(td, attr, None)
            if val is not None and val == val and val > 0:
                price = float(val)
                break

        if price:
            prices[symbol] = price
        ib.cancelMktData(td.contract)

    # Switch back to live for future requests
    ib.reqMarketDataType(1)

    return prices


def load_target_portfolio():
    """Load strategy target portfolio."""
    path = LIVE_DIR / "target_portfolio_latest.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, index_col=0)
    targets = {}
    for ticker, row in df.iterrows():
        targets[ticker] = float(row.iloc[0])
    return targets


def load_factor_scores():
    """Load latest factor scores."""
    path = DATA_DIR / "factor_scores_latest.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    return df.iloc[:, 0].sort_values(ascending=False)


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    print(f"\n--- {title} " + "-" * (74 - len(title)))


def show_account_summary(summary):
    """Display account summary."""
    print_header(
        f"ACCOUNT OVERVIEW  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    nlv = summary.get("NetLiquidation", 0)
    cash = summary.get("TotalCashValue", 0)
    pos_val = summary.get("GrossPositionValue", 0)
    upnl = summary.get("UnrealizedPnL", 0)
    rpnl = summary.get("RealizedPnL", 0)
    avail = summary.get("AvailableFunds", 0)

    print(f"  Net Liquidation:     ${nlv:>12,.2f}")
    print(f"  Cash:                ${cash:>12,.2f}")
    print(f"  Position Value:      ${pos_val:>12,.2f}")
    print(f"  Unrealized P&L:      ${upnl:>+12,.2f}")
    print(f"  Realized P&L:        ${rpnl:>+12,.2f}")
    print(f"  Available Funds:     ${avail:>12,.2f}")
    invested_pct = (pos_val / nlv * 100) if nlv > 0 else 0
    print(f"  Invested:            {invested_pct:>11.1f}%")

    # Cash reserve status
    reserve_ok = cash >= CASH_RESERVE
    reserve_status = "OK" if reserve_ok else "BELOW MINIMUM"
    print(f"\n  Cash Reserve:        ${CASH_RESERVE:>12,.0f} minimum")
    print(f"  Current Cash:        ${cash:>12,.2f}  [{reserve_status}]")
    deployable = max(0, cash - CASH_RESERVE)
    print(f"  Deployable Cash:     ${deployable:>12,.2f}")
    if not reserve_ok:
        shortfall = CASH_RESERVE - cash
        print(f"  SHORTFALL:           ${shortfall:>12,.2f}  "
              f"** Buy trades will be restricted **")


def show_positions_vs_targets(
    ib_positions, live_prices, targets, factor_scores, nlv
):
    """Show current positions vs strategy targets with performance."""
    print_header("POSITIONS vs STRATEGY TARGETS")

    # Build position data with live prices
    pos_data = []
    ib_tickers = set()

    for pos in ib_positions:
        ticker = pos["ticker"]
        if pos["shares"] == 0:
            continue
        ib_tickers.add(ticker)

        live_price = live_prices.get(ticker, pos["avg_cost"])
        mkt_value = pos["shares"] * live_price
        cost_basis = pos["shares"] * pos["avg_cost"]
        pnl = mkt_value - cost_basis
        pnl_pct = (pnl / cost_basis) if cost_basis != 0 else 0

        target_weight = targets.get(ticker, 0)
        actual_weight = mkt_value / nlv if nlv > 0 else 0
        drift = actual_weight - target_weight

        score = (
            factor_scores.get(ticker) if ticker in factor_scores.index
            else None
        )

        stop_price = pos["avg_cost"] * (1 - ENTRY_STOP_LOSS_PCT)
        stop_dist = (
            (live_price - stop_price) / live_price
        ) if live_price > 0 else 0

        pos_data.append({
            "ticker": ticker,
            "shares": pos["shares"],
            "avg_cost": pos["avg_cost"],
            "live_price": live_price,
            "mkt_value": mkt_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "target_w": target_weight,
            "actual_w": actual_weight,
            "drift": drift,
            "score": score,
            "stop_price": stop_price,
            "stop_dist": stop_dist,
            "in_target": ticker in targets,
        })

    # Sort: strategy positions first (by target weight desc), then others
    pos_data.sort(
        key=lambda x: (-x["target_w"], -x["mkt_value"])
    )

    # Print positions table
    hdr = (
        f"  {'Ticker':<6} {'Shares':>7} {'Avg Cost':>9} "
        f"{'Price':>9} {'Mkt Val':>10} {'P&L':>9} {'P&L%':>7} "
        f"{'Target':>7} {'Actual':>7} {'Drift':>7} "
        f"{'Score':>6} {'Stop':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    total_value = 0
    total_pnl = 0
    strategy_value = 0

    for p in pos_data:
        ticker_display = p["ticker"]
        if not p["in_target"]:
            ticker_display = p["ticker"] + "*"

        score_str = f"{p['score']:.3f}" if p["score"] is not None else "  n/a"
        target_str = f"{p['target_w']*100:>5.1f}%" if p["target_w"] > 0 else "    -"
        drift_str = f"{p['drift']*100:>+5.1f}%" if p["target_w"] > 0 else "    -"

        pnl_sign = "+" if p["pnl"] >= 0 else ""

        print(
            f"  {ticker_display:<6} {p['shares']:>7.0f} "
            f"${p['avg_cost']:>8.2f} ${p['live_price']:>8.2f} "
            f"${p['mkt_value']:>9,.0f} "
            f"{pnl_sign}${abs(p['pnl']):>8,.0f} "
            f"{p['pnl_pct']:>+6.1%} "
            f"{target_str} "
            f"{p['actual_w']*100:>5.1f}% "
            f"{drift_str} "
            f"{score_str} "
            f"${p['stop_price']:>7.2f}"
        )

        total_value += p["mkt_value"]
        total_pnl += p["pnl"]
        if p["in_target"]:
            strategy_value += p["mkt_value"]

    print("  " + "-" * (len(hdr) - 2))
    pnl_sign = "+" if total_pnl >= 0 else ""
    print(
        f"  {'TOTAL':<6} {'':>7} {'':>9} {'':>9} "
        f"${total_value:>9,.0f} "
        f"{pnl_sign}${abs(total_pnl):>8,.0f}"
    )

    # Positions not in target
    non_target = [p for p in pos_data if not p["in_target"]]
    if non_target:
        print(
            f"\n  * = Not in strategy target portfolio "
            f"({len(non_target)} position(s))"
        )

    # Missing targets
    missing = set(targets.keys()) - ib_tickers
    if missing:
        print_section("MISSING TARGET POSITIONS")
        for ticker in sorted(missing):
            w = targets[ticker]
            target_val = w * nlv
            price = live_prices.get(ticker)
            score = (
                factor_scores.get(ticker) if ticker in factor_scores.index
                else None
            )
            score_str = f"{score:.3f}" if score else "n/a"
            if price:
                shares_needed = int(target_val / price)
                print(
                    f"  {ticker:<6}  target={w*100:.0f}%  "
                    f"~${target_val:,.0f}  "
                    f"@${price:.2f} = {shares_needed} shares  "
                    f"score={score_str}"
                )
            else:
                print(
                    f"  {ticker:<6}  target={w*100:.0f}%  "
                    f"~${target_val:,.0f}  "
                    f"(no live price)  score={score_str}"
                )

    return pos_data


def show_performance_summary(pos_data, summary):
    """Show performance analysis."""
    print_header("PERFORMANCE ANALYSIS")

    nlv = summary.get("NetLiquidation", 0)
    strategy_positions = [p for p in pos_data if p["in_target"]]
    non_strategy = [p for p in pos_data if not p["in_target"]]

    total_pnl = sum(p["pnl"] for p in pos_data)
    total_cost = sum(
        p["shares"] * p["avg_cost"] for p in pos_data
    )
    total_return = total_pnl / total_cost if total_cost > 0 else 0

    winners = [p for p in pos_data if p["pnl"] > 0]
    losers = [p for p in pos_data if p["pnl"] < 0]

    print(f"  Total Positions:     {len(pos_data)}")
    print(f"  Strategy Positions:  {len(strategy_positions)}")
    print(f"  Non-Strategy:        {len(non_strategy)}")
    print(f"  Winners / Losers:    {len(winners)} / {len(losers)}")
    win_rate = len(winners) / len(pos_data) if pos_data else 0
    print(f"  Win Rate:            {win_rate:.0%}")
    print(f"  Total P&L:           ${total_pnl:>+,.2f}")
    print(f"  Portfolio Return:    {total_return:>+.2%}")

    if winners:
        best = max(pos_data, key=lambda p: p["pnl_pct"])
        print(
            f"  Best:                {best['ticker']} "
            f"({best['pnl_pct']:+.1%}, ${best['pnl']:+,.0f})"
        )
    if losers:
        worst = min(pos_data, key=lambda p: p["pnl_pct"])
        print(
            f"  Worst:               {worst['ticker']} "
            f"({worst['pnl_pct']:+.1%}, ${worst['pnl']:+,.0f})"
        )

    # Stop-loss proximity check
    print_section("STOP-LOSS STATUS")
    at_risk = [
        p for p in pos_data
        if p["stop_dist"] < 0.05 and p["shares"] > 0
    ]
    if at_risk:
        print("  POSITIONS NEAR STOP-LOSS (<5% buffer):")
        for p in sorted(at_risk, key=lambda x: x["stop_dist"]):
            print(
                f"    {p['ticker']:<6} price=${p['live_price']:.2f}  "
                f"stop=${p['stop_price']:.2f}  "
                f"buffer={p['stop_dist']:.1%}"
            )
    else:
        print("  All positions have >5% buffer above stop-loss.")

    triggered = [
        p for p in pos_data
        if p["live_price"] <= p["stop_price"] and p["shares"] > 0
    ]
    if triggered:
        print("\n  STOP-LOSS TRIGGERED:")
        for p in triggered:
            print(
                f"    {p['ticker']:<6} price=${p['live_price']:.2f} "
                f"< stop=${p['stop_price']:.2f}  "
                f"SELL {int(p['shares'])} shares"
            )

    # Weight drift analysis
    print_section("WEIGHT DRIFT ANALYSIS")
    drifted = [
        p for p in pos_data
        if p["target_w"] > 0 and abs(p["drift"]) > 0.02
    ]
    if drifted:
        print("  Positions drifted >2% from target:")
        for p in sorted(drifted, key=lambda x: -abs(x["drift"])):
            print(
                f"    {p['ticker']:<6} target={p['target_w']*100:.0f}%  "
                f"actual={p['actual_w']*100:.1f}%  "
                f"drift={p['drift']*100:+.1f}%"
            )
    else:
        print("  All positions within 2% of target weight.")

    max_drift = max(
        (abs(p["drift"]) for p in pos_data if p["target_w"] > 0),
        default=0,
    )
    print(f"  Max absolute drift: {max_drift*100:.1f}%")
    rebal_threshold = 5.0
    if max_drift * 100 > rebal_threshold:
        print(
            f"  REBALANCE RECOMMENDED "
            f"(drift {max_drift*100:.1f}% > {rebal_threshold:.0f}% threshold)"
        )
    else:
        print(f"  No rebalance needed (threshold: {rebal_threshold:.0f}%)")


def generate_trade_recommendations(
    pos_data, targets, live_prices, factor_scores, nlv, cash
):
    """Generate trades to align portfolio with strategy."""
    print_header("TRADE RECOMMENDATIONS")

    ib_tickers = {p["ticker"]: p for p in pos_data}
    trades = []

    # 1. Sell positions not in target (except special holdings)
    special = {"IBKR"}  # Keep broker stock
    for p in pos_data:
        if not p["in_target"] and p["ticker"] not in special:
            if p["shares"] > 0:
                trades.append({
                    "action": "SELL",
                    "ticker": p["ticker"],
                    "shares": int(abs(p["shares"])),
                    "price": p["live_price"],
                    "value": abs(p["mkt_value"]),
                    "reason": "Not in strategy target",
                })
            elif p["shares"] < 0:
                trades.append({
                    "action": "BUY TO COVER",
                    "ticker": p["ticker"],
                    "shares": int(abs(p["shares"])),
                    "price": p["live_price"],
                    "value": abs(p["mkt_value"]),
                    "reason": "Close short position",
                })

    # Calculate available cash for buys (cash - reserve + sell proceeds)
    sell_proceeds = sum(
        t["value"] for t in trades if t["action"] == "SELL"
    )
    cover_cost = sum(
        t["value"] for t in trades if t["action"] == "BUY TO COVER"
    )
    available_for_buys = cash + sell_proceeds - cover_cost - CASH_RESERVE
    if available_for_buys < 0:
        available_for_buys = 0

    # Collect buy candidates
    buy_candidates = []

    # 2. Buy missing target positions
    for ticker, target_w in targets.items():
        if ticker not in ib_tickers:
            price = live_prices.get(ticker)
            if price and price > 0:
                target_val = target_w * nlv
                shares = int(target_val / price)
                if shares > 0:
                    buy_candidates.append({
                        "action": "BUY",
                        "ticker": ticker,
                        "shares": shares,
                        "price": price,
                        "value": shares * price,
                        "reason": (
                            f"New target position ({target_w*100:.0f}%)"
                        ),
                    })

    # 3. Rebalance existing positions with >3% drift
    for p in pos_data:
        if p["target_w"] > 0 and abs(p["drift"]) > 0.03:
            target_val = p["target_w"] * nlv
            current_val = p["mkt_value"]
            diff_val = target_val - current_val
            diff_shares = int(abs(diff_val) / p["live_price"])

            if diff_shares > 0:
                if diff_val > 0:
                    buy_candidates.append({
                        "action": "BUY",
                        "ticker": p["ticker"],
                        "shares": diff_shares,
                        "price": p["live_price"],
                        "value": diff_shares * p["live_price"],
                        "reason": (
                            f"Rebalance (drift {p['drift']*100:+.1f}%)"
                        ),
                    })
                else:
                    trades.append({
                        "action": "SELL",
                        "ticker": p["ticker"],
                        "shares": diff_shares,
                        "price": p["live_price"],
                        "value": diff_shares * p["live_price"],
                        "reason": (
                            f"Rebalance (drift {p['drift']*100:+.1f}%)"
                        ),
                    })

    # Apply cash reserve cap to buy candidates
    running_spend = 0
    for bc in buy_candidates:
        if running_spend + bc["value"] <= available_for_buys:
            trades.append(bc)
            running_spend += bc["value"]
        else:
            remaining = available_for_buys - running_spend
            if remaining > bc["price"]:
                reduced = int(remaining / bc["price"])
                if reduced > 0:
                    bc["shares"] = reduced
                    bc["value"] = reduced * bc["price"]
                    bc["reason"] += (
                        f" [capped by ${CASH_RESERVE:,.0f} reserve]"
                    )
                    trades.append(bc)
                    running_spend += bc["value"]

    if not trades:
        print("  No trades needed. Portfolio is aligned with strategy.")
        return trades

    # Sort: sells first, then buys
    trades.sort(
        key=lambda t: (0 if "SELL" in t["action"] else 1, t["ticker"])
    )

    # Print trade table
    print(
        f"  {'Action':<14} {'Ticker':<6} {'Shares':>7} "
        f"{'Price':>9} {'Value':>10}  Reason"
    )
    print("  " + "-" * 72)

    total_buy = 0
    total_sell = 0

    for t in trades:
        print(
            f"  {t['action']:<14} {t['ticker']:<6} {t['shares']:>7} "
            f"${t['price']:>8.2f} ${t['value']:>9,.0f}  {t['reason']}"
        )
        if "BUY" in t["action"]:
            total_buy += t["value"]
        else:
            total_sell += t["value"]

    print("  " + "-" * 72)
    print(f"  Total sells: ${total_sell:>10,.0f}")
    print(f"  Total buys:  ${total_buy:>10,.0f}")
    print(f"  Net cash:    ${total_sell - total_buy:>+10,.0f}")
    cash_after = cash + total_sell - total_buy
    print(f"  Cash after:  ${cash_after:>10,.0f}  "
          f"(reserve: ${CASH_RESERVE:,.0f})")
    print(f"\n  {len(trades)} trade(s) recommended.")

    return trades


def save_snapshot(pos_data, summary, trades):
    """Save portfolio snapshot to CSV for record keeping."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save positions snapshot
    pos_file = SNAPSHOT_DIR / f"positions_{ts}.csv"
    with open(pos_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticker", "shares", "avg_cost", "live_price",
                "mkt_value", "pnl", "pnl_pct", "target_w",
                "actual_w", "drift", "score", "stop_price",
            ],
        )
        writer.writeheader()
        for p in pos_data:
            writer.writerow({
                "ticker": p["ticker"],
                "shares": p["shares"],
                "avg_cost": round(p["avg_cost"], 2),
                "live_price": round(p["live_price"], 2),
                "mkt_value": round(p["mkt_value"], 2),
                "pnl": round(p["pnl"], 2),
                "pnl_pct": round(p["pnl_pct"], 4),
                "target_w": round(p["target_w"], 4),
                "actual_w": round(p["actual_w"], 4),
                "drift": round(p["drift"], 4),
                "score": (
                    round(p["score"], 4)
                    if p["score"] is not None else ""
                ),
                "stop_price": round(p["stop_price"], 2),
            })

    # Save account summary
    acct_file = SNAPSHOT_DIR / f"account_{ts}.csv"
    with open(acct_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    # Save trades if any
    if trades:
        trades_file = SNAPSHOT_DIR / f"trades_{ts}.csv"
        with open(trades_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "action", "ticker", "shares",
                    "price", "value", "reason",
                ],
            )
            writer.writeheader()
            for t in trades:
                writer.writerow(t)
        print(f"  Trades saved:    {trades_file.name}")

    print(f"\n  Snapshot saved to: {SNAPSHOT_DIR.name}/")
    print(f"  Positions saved: {pos_file.name}")
    print(f"  Account saved:   {acct_file.name}")

    # Also update the "latest" symlink-style files
    latest_pos = LIVE_DIR / "positions_latest.csv"
    import shutil
    shutil.copy2(pos_file, latest_pos)
    print(f"  Updated:         positions_latest.csv")


def main():
    parser = argparse.ArgumentParser(
        description="IB Portfolio Manager - One Stop Shop",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Review only (no trade generation)",
    )
    parser.add_argument(
        "--trades", action="store_true",
        help="Show trade recommendations",
    )
    args = parser.parse_args()

    # If neither flag set, show everything
    show_all = not args.review and not args.trades

    # Connect to IB
    ib = connect_ib()

    try:
        # Pull live data
        summary = get_account_summary(ib)
        ib_positions = get_positions(ib)
        nlv = summary.get("NetLiquidation", 0)

        # Load strategy data
        targets = load_target_portfolio()
        factor_scores = load_factor_scores()

        # Get all tickers we need prices for
        all_tickers = set(p["ticker"] for p in ib_positions)
        all_tickers.update(targets.keys())
        all_tickers = sorted(all_tickers)

        print(f"Fetching live prices for {len(all_tickers)} tickers...")
        live_prices = get_live_prices(ib, all_tickers)
        print(f"Got prices for {len(live_prices)} tickers.\n")

        # Show account summary
        show_account_summary(summary)

        # Show positions vs targets
        pos_data = show_positions_vs_targets(
            ib_positions, live_prices, targets, factor_scores, nlv
        )

        # Show performance
        show_performance_summary(pos_data, summary)

        # Trade recommendations
        trades = []
        if show_all or args.trades:
            cash = summary.get("TotalCashValue", 0)
            trades = generate_trade_recommendations(
                pos_data, targets, live_prices, factor_scores, nlv, cash
            )

        # Save snapshot
        print_section("SAVING RECORDS")
        save_snapshot(pos_data, summary, trades)

    finally:
        ib.disconnect()
        print("\nDisconnected from IB Gateway.")


if __name__ == "__main__":
    main()
