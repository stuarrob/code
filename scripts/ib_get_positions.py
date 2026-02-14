"""Fetch current positions from Interactive Brokers Gateway."""

import asyncio
import nest_asyncio
from ib_insync import IB, util

# Allow nested event loops (needed for ib_insync sync calls within asyncio.run)
nest_asyncio.apply()

IB_HOST = "127.0.0.1"
IB_PORT = 4001  # Live trading port


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

            print(
                f"  {c.symbol:<8} {c.secType:<6} {shares:>10,.2f} "
                f"${avg_cost:>11,.2f} ${mkt_val:>13,.2f} {c.currency:<5}"
            )

        print("  " + "-" * 60)
        print(f"  {'TOTAL':<8} {'':6} {'':>10} {'':>12} ${total_value:>13,.2f}")
        print(f"\n  Total positions: {len(positions)}")

    ib.disconnect()
    print("\nDisconnected from IB Gateway.")


if __name__ == "__main__":
    main()
