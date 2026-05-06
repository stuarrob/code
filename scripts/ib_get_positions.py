"""Fetch current positions and open orders from Interactive Brokers Gateway."""

import csv
from pathlib import Path

import nest_asyncio
from ib_insync import IB

# Allow nested event loops (needed for ib_insync sync calls within asyncio.run)
nest_asyncio.apply()

IB_HOST = "127.0.0.1"
IB_PORT = 4001  # Live trading port

TRADING_DIR = Path.home() / "trading"
LIVE_DIR = TRADING_DIR / "live_portfolio"
TARGET_PORTFOLIO_FILE = LIVE_DIR / "target_portfolio_latest.csv"


def load_target_tickers():
    """Load target portfolio tickers from CSV."""
    if not TARGET_PORTFOLIO_FILE.exists():
        return None

    tickers = set()
    with open(TARGET_PORTFOLIO_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() and row[0].strip().isalpha():
                tickers.add(row[0].strip().upper())
    return tickers


def main():
    ib = IB()

    print(f"Connecting to IB Gateway on {IB_HOST}:{IB_PORT}...")
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=1, readonly=True, timeout=10)
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Is IB Gateway running?")
        return

    accounts = ib.managedAccounts()
    print(f"Connected. Account: {accounts[0] if accounts else 'N/A'}\n")

    # --- Account Summary ---
    print("=" * 80)
    print("ACCOUNT SUMMARY")
    print("=" * 80)

    account_values = ib.accountSummary()

    key_fields = [
        "NetLiquidation",
        "TotalCashValue",
        "GrossPositionValue",
        "AvailableFunds",
        "BuyingPower",
        "UnrealizedPnL",
        "RealizedPnL",
    ]

    for av in account_values:
        if av.tag in key_fields and av.currency == "USD":
            print(f"  {av.tag:.<30} ${float(av.value):>14,.2f}")

    # --- Positions ---
    print("\n" + "=" * 80)
    print("CURRENT POSITIONS")
    print("=" * 80)

    positions = ib.positions()

    position_tickers = set()
    long_tickers = set()

    if not positions:
        print("  No positions found.")
    else:
        print(
            f"  {'Ticker':<8} {'Type':<6} {'Shares':>10} {'Avg Cost':>12} "
            f"{'Market Value':>14} {'Currency':<5}"
        )
        print("  " + "-" * 60)

        total_value = 0.0
        for pos in positions:
            c = pos.contract
            shares = float(pos.position)
            avg_cost = float(pos.avgCost)
            mkt_val = shares * avg_cost
            total_value += mkt_val

            position_tickers.add(c.symbol)
            if shares > 0:
                long_tickers.add(c.symbol)

            print(
                f"  {c.symbol:<8} {c.secType:<6} {shares:>10,.2f} "
                f"${avg_cost:>11,.2f} ${mkt_val:>13,.2f} {c.currency:<5}"
            )

        print("  " + "-" * 60)
        print(f"  {'TOTAL':<8} {'':6} {'':>10} {'':>12} ${total_value:>13,.2f}")
        print(f"\n  Total positions: {len(positions)}")

    # --- Open Orders ---
    print("\n" + "=" * 80)
    print("OPEN ORDERS")
    print("=" * 80)

    open_trades = ib.openTrades()

    trailing_stops = []
    other_orders = []

    for trade in open_trades:
        c = trade.contract
        o = trade.order
        s = trade.orderStatus

        order_info = {
            "ticker": c.symbol,
            "action": o.action,
            "shares": int(o.totalQuantity),
            "order_type": o.orderType,
            "limit_price": o.lmtPrice if o.lmtPrice else None,
            "trailing_pct": o.trailingPercent if o.trailingPercent else None,
            "status": s.status,
            "order_id": o.orderId,
        }

        if o.orderType == "TRAIL":
            trailing_stops.append(order_info)
        else:
            other_orders.append(order_info)

    if not open_trades:
        print("  No open orders.")
    else:
        if other_orders:
            print(
                f"\n  {'Ticker':<8} {'Action':<6} {'Shares':>8} "
                f"{'Type':<6} {'Limit':>10} {'Status':<12} {'Order ID':>10}"
            )
            print("  " + "-" * 68)

            for od in other_orders:
                lp = f"${od['limit_price']:,.2f}" if od["limit_price"] else ""
                print(
                    f"  {od['ticker']:<8} {od['action']:<6} "
                    f"{od['shares']:>8,} {od['order_type']:<6} "
                    f"{lp:>10} {od['status']:<12} {od['order_id']:>10}"
                )

        if trailing_stops:
            print(f"\n  TRAILING STOP ORDERS")
            print(
                f"  {'Ticker':<8} {'Action':<6} {'Shares':>8} "
                f"{'Trail %':>8} {'Status':<12} {'Order ID':>10}"
            )
            print("  " + "-" * 58)

            for ts in trailing_stops:
                tp = f"{ts['trailing_pct']:.1f}%" if ts["trailing_pct"] else "N/A"
                print(
                    f"  {ts['ticker']:<8} {ts['action']:<6} "
                    f"{ts['shares']:>8,} {tp:>8} "
                    f"{ts['status']:<12} {ts['order_id']:>10}"
                )

        print(f"\n  Total open orders: {len(open_trades)} "
              f"({len(trailing_stops)} trailing stops, "
              f"{len(other_orders)} other)")

    # --- Trailing Stop Coverage ---
    print("\n" + "=" * 80)
    print("TRAILING STOP COVERAGE")
    print("=" * 80)

    tickers_with_trailing = {ts["ticker"] for ts in trailing_stops}
    covered = long_tickers & tickers_with_trailing
    missing = long_tickers - tickers_with_trailing

    if not long_tickers:
        print("  No long positions to check.")
    else:
        if covered:
            print(f"\n  Protected ({len(covered)}):")
            for t in sorted(covered):
                ts_info = next(
                    (s for s in trailing_stops if s["ticker"] == t), None
                )
                tp = (f"{ts_info['trailing_pct']:.1f}%"
                      if ts_info and ts_info["trailing_pct"] else "N/A")
                print(f"    {t:<8} trailing stop {tp}")

        if missing:
            print(f"\n  ** MISSING TRAILING STOPS ({len(missing)}) **")
            for t in sorted(missing):
                print(f"    {t:<8} <-- NO TRAILING STOP")

        print(f"\n  Coverage: {len(covered)}/{len(long_tickers)} "
              f"long positions protected")

    # --- Portfolio Health Check ---
    print("\n" + "=" * 80)
    print("PORTFOLIO HEALTH CHECK")
    print("=" * 80)

    issues = []

    # Check for short positions
    short_positions = [
        p for p in positions if float(p.position) < 0
    ]
    if short_positions:
        issues.append("SHORT POSITIONS DETECTED (should be long-only):")
        for p in short_positions:
            issues.append(
                f"    {p.contract.symbol}: "
                f"{float(p.position):,.2f} shares"
            )

    # Check cash and NLV from account values
    nlv = None
    cash = None
    gross_pos = None

    for av in account_values:
        if av.currency != "USD":
            continue
        if av.tag == "NetLiquidation":
            nlv = float(av.value)
        elif av.tag == "TotalCashValue":
            cash = float(av.value)
        elif av.tag == "GrossPositionValue":
            gross_pos = float(av.value)

    if cash is not None and cash < 0:
        issues.append(
            f"  NEGATIVE CASH: ${cash:,.2f} (margin in use)"
        )

    if nlv is not None and gross_pos is not None:
        leverage = gross_pos / nlv if nlv > 0 else 0
        print(f"\n  Net Liquidation:     ${nlv:>14,.2f}")
        print(f"  Gross Position Value: ${gross_pos:>13,.2f}")
        print(f"  Leverage Ratio:       {leverage:>13.2f}x")

        if leverage > 1.0:
            issues.append(
                f"  LEVERAGE: {leverage:.2f}x "
                f"(gross positions exceed NLV)"
            )

    # Check against target portfolio
    target_tickers = load_target_tickers()
    if target_tickers is not None:
        off_target = position_tickers - target_tickers
        if off_target:
            issues.append(
                "  OFF-TARGET POSITIONS (not in target portfolio):"
            )
            for t in sorted(off_target):
                issues.append(f"    {t}")
    else:
        print(f"\n  Target portfolio not found: {TARGET_PORTFOLIO_FILE}")

    # Print issues or all-clear
    if issues:
        print(f"\n  ** {len(issues)} issue(s) found **\n")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n  All checks passed.")

    ib.disconnect()
    print("\nDisconnected from IB Gateway.")


if __name__ == "__main__":
    main()
