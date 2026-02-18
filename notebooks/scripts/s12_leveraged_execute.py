"""
Step 12: Leveraged ETF Strategy Execution

Manages the TECL + NAIL (50/50) leveraged momentum strategy:
- Detect if strategy is running (IB positions)
- Compute live signals (SMA 200 + vol filter on reference indices)
- Generate trade plan (buy/sell/rebalance/hold)
- Execute via LIMIT GTC orders (no trailing stops)
- Check order status and fix unfilled

Config C: SMA 200 + vol filter, no stops, no bonds, 100% equity per leg.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Default strategy parameters ──────────────────────────
DEFAULT_TICKERS = ["TECL", "NAIL"]
DEFAULT_REFERENCES = {"TECL": "XLK", "NAIL": "ITB"}
DEFAULT_VOL_THRESHOLDS = {"XLK": 0.22, "ITB": 0.22}
DEFAULT_TARGET_WEIGHTS = {"TECL": 0.50, "NAIL": 0.50}
DRIFT_THRESHOLD = 0.05  # 5% drift triggers rebalance


def detect_strategy_state(
    tickers: List[str],
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 17,
) -> dict:
    """Connect to IB and detect current leveraged strategy positions.

    Returns dict with:
        positions: {ticker: {shares, avg_cost, market_value}}
        open_orders: [{ticker, action, shares, order_type, status}]
        cash_available: float
        strategy_running: bool
        total_value: float (sum of position market values)
    """
    from ib_insync import IB

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10)
    account = ib.managedAccounts()[0]

    # Get positions
    all_positions = ib.positions()
    ib.sleep(1)
    pos_map = {}
    for p in all_positions:
        sym = p.contract.symbol
        if sym in tickers:
            pos_map[sym] = {
                "shares": int(p.position),
                "avg_cost": p.avgCost,
                "market_value": 0,  # filled below
            }

    # Get market values via reqTickers if positions exist
    if pos_map:
        from ib_insync import Stock
        contracts = []
        for ticker in pos_map:
            c = Stock(ticker, "SMART", "USD")
            contracts.append(c)
        ib.qualifyContracts(*contracts)
        ticker_data = ib.reqTickers(*contracts)
        ib.sleep(2)
        for td in ticker_data:
            sym = td.contract.symbol
            price = td.last or td.close or td.marketPrice()
            if sym in pos_map and price and price > 0:
                pos_map[sym]["market_value"] = pos_map[sym]["shares"] * price
                pos_map[sym]["price"] = price

    # Get open orders for these tickers
    open_trades = ib.openTrades()
    relevant_orders = []
    for t in open_trades:
        sym = t.contract.symbol
        if sym in tickers:
            relevant_orders.append({
                "ticker": sym,
                "action": t.order.action,
                "shares": int(t.order.totalQuantity),
                "order_type": t.order.orderType,
                "status": t.orderStatus.status,
                "order_id": t.order.orderId,
                "_trade": t,
            })

    # Get account cash
    account_summary = ib.accountSummary(account)
    ib.sleep(1)
    cash = 0
    for item in account_summary:
        if item.tag == "TotalCashValue" and item.currency == "USD":
            cash = float(item.value)
            break

    ib.disconnect()

    total_value = sum(p["market_value"] for p in pos_map.values())
    strategy_running = any(p["shares"] > 0 for p in pos_map.values())

    # Summary
    if strategy_running:
        print("Strategy RUNNING:")
        for ticker, info in pos_map.items():
            print(f"  {ticker}: {info['shares']} shares "
                  f"(${info['market_value']:,.0f}, "
                  f"avg cost ${info['avg_cost']:.2f})")
        print(f"  Total invested: ${total_value:,.0f}")
    else:
        print("Strategy NOT running — no positions in TECL or NAIL")

    print(f"  Account cash: ${cash:,.0f}")
    if relevant_orders:
        print(f"  Open orders: {len(relevant_orders)}")
        for o in relevant_orders:
            print(f"    {o['action']} {o['shares']} {o['ticker']} "
                  f"({o['order_type']}) — {o['status']}")

    return {
        "positions": pos_map,
        "open_orders": relevant_orders,
        "cash_available": cash,
        "strategy_running": strategy_running,
        "total_value": total_value,
    }


def compute_live_signals(
    data_dir: Path,
    references: Dict[str, str] = None,
    vol_thresholds: Dict[str, float] = None,
) -> dict:
    """Compute current signal state for each leveraged ETF.

    Loads reference index prices from IB cache and runs SMA 200 +
    vol filter signal computation.

    Returns dict: {ticker: {signal_state, sma_signal, ref_vol,
                            price, sma_200, above_sma}}
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from leveraged.signals import compute_signals

    if references is None:
        references = DEFAULT_REFERENCES
    if vol_thresholds is None:
        vol_thresholds = DEFAULT_VOL_THRESHOLDS

    data_dir = Path(data_dir)
    results = {}

    for lev_ticker, ref_ticker in references.items():
        path = data_dir / f"{ref_ticker}.parquet"
        if not path.exists():
            print(f"  {ref_ticker}: No cached data at {path}")
            results[lev_ticker] = {
                "signal_state": "risk_off",
                "error": f"No data for {ref_ticker}",
            }
            continue

        df = pd.read_parquet(path)
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        prices = df["close"].sort_index()

        threshold = vol_thresholds.get(ref_ticker, 0.20)

        signals = compute_signals(
            prices,
            signal_mode="sma_only",
            vol_filter_threshold=threshold,
            vol_filter_lookback=20,
        )

        latest = signals.iloc[-1]
        sma_val = latest.get("sma", None)
        ref_vol = latest.get("ref_vol", None)

        results[lev_ticker] = {
            "signal_state": latest["signal_state"],
            "sma_signal": bool(latest["sma_signal"]),
            "ref_vol": float(ref_vol) if ref_vol and not np.isnan(ref_vol) else None,
            "vol_filter": bool(latest.get("vol_filter", True)),
            "price": float(prices.iloc[-1]),
            "sma_200": float(sma_val) if sma_val and not np.isnan(sma_val) else None,
            "above_sma": bool(latest["sma_signal"]),
            "ref_ticker": ref_ticker,
            "vol_threshold": threshold,
            "data_date": str(prices.index[-1].date()),
        }

    # Display
    print("Current Signals:")
    print(f"{'Ticker':<8} {'Ref':<6} {'Signal':<12} {'Price':>10} {'SMA200':>10} "
          f"{'RefVol':>8} {'VolThresh':>10}")
    print("-" * 72)
    for ticker, info in results.items():
        if "error" in info:
            print(f"{ticker:<8} {'ERROR':<6} {info['error']}")
            continue
        state = info["signal_state"].upper()
        marker = ">> " if state == "RISK_ON" else "   "
        print(f"{marker}{ticker:<8} {info['ref_ticker']:<6} {state:<12} "
              f"${info['price']:>8.2f} ${info['sma_200']:>8.2f} "
              f"{info['ref_vol']:>7.1%} {info['vol_threshold']:>9.0%}")

    return results


def generate_leveraged_trades(
    state: dict,
    signals: dict,
    tickers: List[str] = None,
    budget: float = 30_000,
    target_weights: Dict[str, float] = None,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 17,
) -> list:
    """Generate trade plan based on current state and signals.

    Decision logic:
    - No positions + RISK_ON → BUY (allocate from budget)
    - No positions + RISK_OFF → nothing
    - Holding + RISK_ON → check drift, rebalance if >5%
    - Holding + RISK_OFF → SELL everything
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if target_weights is None:
        target_weights = DEFAULT_TARGET_WEIGHTS

    positions = state.get("positions", {})
    strategy_running = state.get("strategy_running", False)
    trades = []

    # Determine which tickers are RISK_ON
    risk_on_tickers = [t for t in tickers if signals.get(t, {}).get("signal_state") == "risk_on"]
    risk_off_tickers = [t for t in tickers if t not in risk_on_tickers]

    # Get fresh prices for share calculation
    prices = {}
    from ib_insync import IB, Stock
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10)
    contracts = {}
    for ticker in tickers:
        c = Stock(ticker, "SMART", "USD")
        contracts[ticker] = c
    ib.qualifyContracts(*contracts.values())
    ticker_data = ib.reqTickers(*contracts.values())
    ib.sleep(2)
    for td in ticker_data:
        sym = td.contract.symbol
        price = td.last or td.close
        if not price or price <= 0:
            price = (td.bid + td.ask) / 2 if td.bid and td.ask else None
        if price and price > 0:
            prices[sym] = price
    ib.disconnect()

    if not strategy_running:
        # ── NEW STRATEGY: allocate from budget ──
        if not risk_on_tickers:
            print("\nNo positions and all signals RISK_OFF. Nothing to do.")
            print("Wait for SMA 200 crossover + vol filter to clear.")
            return []

        # Allocate budget among risk-on tickers
        total_weight = sum(target_weights.get(t, 0) for t in risk_on_tickers)
        for ticker in risk_on_tickers:
            px = prices.get(ticker)
            if not px:
                print(f"  {ticker}: No price available — skipping")
                continue
            weight = target_weights.get(ticker, 0.5) / total_weight
            alloc = budget * weight
            shares = int(alloc / px)
            if shares > 0:
                trades.append({
                    "ticker": ticker, "action": "BUY",
                    "shares": shares, "ref_price": px,
                    "est_value": shares * px,
                    "reason": f"New position ({weight:.0%} of ${budget:,.0f} budget)",
                })

        if risk_off_tickers:
            for t in risk_off_tickers:
                sig = signals.get(t, {})
                print(f"\n  {t}: RISK_OFF (vol={sig.get('ref_vol', 0):.1%}, "
                      f"above SMA={sig.get('above_sma', False)}). Not buying.")

    else:
        # ── EXISTING STRATEGY: rebalance / exit ──

        # First: SELL anything that's turned RISK_OFF
        for ticker in risk_off_tickers:
            pos = positions.get(ticker, {})
            shares = pos.get("shares", 0)
            if shares > 0:
                px = prices.get(ticker, pos.get("price", 0))
                trades.append({
                    "ticker": ticker, "action": "SELL",
                    "shares": shares, "ref_price": px,
                    "est_value": shares * px,
                    "reason": f"Signal RISK_OFF — exit position",
                })

        # Then: check drift on RISK_ON positions
        if risk_on_tickers:
            held_tickers = [t for t in risk_on_tickers if positions.get(t, {}).get("shares", 0) > 0]
            unheld_tickers = [t for t in risk_on_tickers if t not in held_tickers]

            # Calculate current total value of risk-on holdings
            total_invested = sum(
                positions.get(t, {}).get("market_value", 0)
                for t in held_tickers
            )

            # BUY any risk-on tickers we don't hold yet
            for ticker in unheld_tickers:
                px = prices.get(ticker)
                if not px:
                    continue
                # Allocate proportionally from current invested capital
                total_weight = sum(target_weights.get(t, 0) for t in risk_on_tickers)
                weight = target_weights.get(ticker, 0.5) / total_weight
                # Use remaining budget or current portfolio value
                alloc_base = max(budget, total_invested) if total_invested > 0 else budget
                alloc = alloc_base * weight
                shares = int(alloc / px)
                if shares > 0:
                    trades.append({
                        "ticker": ticker, "action": "BUY",
                        "shares": shares, "ref_price": px,
                        "est_value": shares * px,
                        "reason": f"Signal turned RISK_ON — new position",
                    })

            # Check drift for existing positions
            if len(held_tickers) >= 2 and total_invested > 0:
                for ticker in held_tickers:
                    pos = positions.get(ticker, {})
                    current_weight = pos.get("market_value", 0) / total_invested
                    target_weight = target_weights.get(ticker, 0.5)
                    drift = current_weight - target_weight

                    if abs(drift) > DRIFT_THRESHOLD:
                        px = prices.get(ticker, pos.get("price", 0))
                        target_val = total_invested * target_weight
                        current_val = pos.get("market_value", 0)
                        diff_val = target_val - current_val
                        diff_shares = int(abs(diff_val) / px) if px > 0 else 0

                        if diff_shares > 0:
                            action = "BUY" if diff_val > 0 else "SELL"
                            trades.append({
                                "ticker": ticker, "action": action,
                                "shares": diff_shares, "ref_price": px,
                                "est_value": diff_shares * px,
                                "reason": f"Rebalance: drift {drift:+.1%} "
                                          f"(current {current_weight:.1%}, target {target_weight:.1%})",
                            })

    # Summary
    if not trades:
        print("\nNo trades needed — positions are in line with signals.")
    else:
        total_buy = sum(t["est_value"] for t in trades if t["action"] == "BUY")
        total_sell = sum(t["est_value"] for t in trades if t["action"] == "SELL")
        print(f"\nTrade plan: {len(trades)} trades")
        for t in trades:
            print(f"  {t['action']:>5} {t['shares']:>6} {t['ticker']:<6} "
                  f"@ ${t['ref_price']:.2f} = ${t['est_value']:>10,.2f}  "
                  f"({t['reason']})")
        if total_buy > 0:
            print(f"\n  Total buys:  ${total_buy:,.0f}")
        if total_sell > 0:
            print(f"  Total sells: ${total_sell:,.0f}")

    return trades


def execute_leveraged_trades(
    trades: list,
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 18,
    use_limit_orders: bool = True,
    limit_buffer_pct: float = 1.0,
    confirm: bool = False,
) -> list:
    """Execute leveraged strategy trades on IB.

    NO trailing stops — SMA 200 crossover is the exit signal.
    Uses LIMIT GTC by default for outside-hours submission.
    """
    if not trades:
        print("No trades to execute.")
        return []

    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)

    if not confirm:
        order_mode = "LIMIT GTC" if use_limit_orders else "MARKET"
        print(f"DRY RUN — CONFIRM=False. No orders placed.")
        print(f"Order mode: {order_mode}")
        if use_limit_orders:
            print(f"Limit buffer: {limit_buffer_pct}%")
        print(f"Trailing stops: NONE (SMA exit signal)\n")
        for t in trades:
            ref = t.get("ref_price", 0)
            is_buy = t["action"] == "BUY"
            if use_limit_orders and ref:
                buf = 1 + limit_buffer_pct / 100 if is_buy else 1 - limit_buffer_pct / 100
                lp = f" limit ${ref * buf:.2f}"
                mode = "LIMIT GTC"
            else:
                lp = ""
                mode = "MARKET"
            print(f"  {t['action']:>5} {t['shares']:>6} {t['ticker']:<6} ({mode}){lp}")
        return []

    from ib_insync import IB, Stock, LimitOrder, MarketOrder

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=False, timeout=10)
    account = ib.managedAccounts()[0]
    order_mode = "LIMIT GTC" if use_limit_orders else "MARKET"
    print(f"Connected for trading: {account}")
    print(f"Order mode: {order_mode}")
    print(f"Trailing stops: NONE\n")

    exec_results = []

    for t in trades:
        ticker = t["ticker"]
        ib_action = t["action"]
        shares = t["shares"]

        contract = Stock(ticker, "SMART", "USD")
        try:
            ib.qualifyContracts(contract)
        except Exception as e:
            print(f"  {ticker}: FAILED — {e}")
            exec_results.append({"ticker": ticker, "status": "FAILED", "message": str(e)})
            continue

        ref_price = t.get("ref_price")
        if use_limit_orders and ref_price:
            is_buy = ib_action == "BUY"
            buf = 1 + limit_buffer_pct / 100 if is_buy else 1 - limit_buffer_pct / 100
            limit_px = round(ref_price * buf, 2)
            order = LimitOrder(ib_action, shares, limit_px)
            order.tif = "GTC"
            order.outsideRth = False
            effective_type = "LIMIT GTC"
        else:
            order = MarketOrder(ib_action, shares)
            effective_type = "MARKET"
            limit_px = None

        price_str = f" @ limit ${limit_px:.2f}" if limit_px else ""
        print(f"  {ib_action} {shares} {ticker} ({effective_type}){price_str}...", end="")
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
            "action": ib_action, "shares": shares,
        })

    # Log execution
    log_file = live_dir / "execution_log.csv"
    file_exists = log_file.exists()
    log_ts = datetime.now().isoformat()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "ticker", "action", "shares", "status",
                        "order_id", "fill_price", "limit_price", "order_type"])
        for r in exec_results:
            w.writerow([log_ts, r.get("ticker"), r.get("action", ""),
                        r.get("shares", ""), r.get("status"),
                        r.get("order_id", ""), r.get("fill_price", ""),
                        r.get("limit_price", ""), r.get("order_type", "")])

    # Save signal history
    _save_signal_history(live_dir)

    # Summary
    filled = sum(1 for r in exec_results
                 if r["status"] in ("Filled", "Submitted", "PreSubmitted"))
    failed = sum(1 for r in exec_results if r["status"] in ("ERROR", "FAILED"))
    print(f"\nExecution summary:")
    print(f"  Executed: {filled}  |  Failed: {failed}  |  Total: {len(exec_results)}")
    if use_limit_orders:
        pending = sum(1 for r in exec_results
                      if r["status"] in ("Submitted", "PreSubmitted"))
        if pending:
            print(f"\n  {pending} LIMIT GTC orders pending — will fill at market open")

    ib.disconnect()
    return exec_results


def check_leveraged_status(
    trades: list,
    tickers: List[str],
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 17,
) -> dict:
    """Check order status for leveraged strategy trades.

    Returns dict with filled, pending, missing lists.
    """
    from ib_insync import IB

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=True, timeout=10)
    account = ib.managedAccounts()[0]
    print(f"Connected (read-only): {account}\n")

    open_trades = ib.openTrades()
    positions = ib.positions()
    ib.sleep(1)

    pos_map = {}
    for p in positions:
        sym = p.contract.symbol
        if sym in tickers:
            pos_map[sym] = pos_map.get(sym, 0) + p.position

    order_map = {}
    for t in open_trades:
        sym = t.contract.symbol
        if sym in tickers:
            info = {
                "ticker": sym,
                "action": t.order.action,
                "shares": int(t.order.totalQuantity),
                "order_type": t.order.orderType,
                "limit_price": getattr(t.order, "lmtPrice", None),
                "status": t.orderStatus.status,
                "filled_qty": int(t.orderStatus.filled),
                "remaining": int(t.orderStatus.remaining),
                "avg_fill": t.orderStatus.avgFillPrice,
                "order_id": t.order.orderId,
                "_trade": t,
            }
            order_map.setdefault(sym, []).append(info)

    filled = []
    pending = []
    missing = []

    for t in trades:
        ticker = t["ticker"]
        action = t["action"]
        ticker_orders = order_map.get(ticker, [])
        matching = [o for o in ticker_orders if o["action"] == action]

        if matching:
            o = matching[0]
            if o["status"] == "Filled":
                filled.append({**t, "fill_status": "FILLED", "fill_price": o["avg_fill"]})
            else:
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
            held = pos_map.get(ticker, 0)
            if action == "BUY" and held >= t["shares"]:
                filled.append({**t, "fill_status": "FILLED (position confirms)"})
            elif action == "SELL" and held == 0:
                filled.append({**t, "fill_status": "FILLED (sold, 0 held)"})
            else:
                missing.append({**t, "fill_status": "MISSING", "current_position": held})

    ib.disconnect()

    print(f"Leveraged trades: {len(trades)}")
    print(f"  Filled:  {len(filled)}")
    print(f"  Pending: {len(pending)}")
    print(f"  Missing: {len(missing)}")

    if not pending and not missing:
        print("\nAll leveraged trades completed. No action needed.")
    else:
        if pending:
            print(f"\n{len(pending)} orders pending:")
            for p in pending:
                print(f"  {p['action']:>5} {p['shares']:>6} {p['ticker']:<6} "
                      f"— {p['ib_status']}, limit ${p.get('limit_price', 'N/A')}")
        if missing:
            print(f"\n{len(missing)} trades missing:")
            for m in missing:
                print(f"  {m['action']:>5} {m['shares']:>6} {m['ticker']:<6} "
                      f"— currently hold {m.get('current_position', '?')}")

    return {"filled": filled, "pending": pending, "missing": missing}


def fix_leveraged_orders(
    check_result: dict,
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 18,
    use_limit_orders: bool = True,
    limit_buffer_pct: float = 1.0,
    confirm: bool = False,
) -> list:
    """Cancel unfilled leveraged orders and resubmit with fresh prices."""
    pending = check_result.get("pending", [])
    missing = check_result.get("missing", [])

    to_cancel = [p for p in pending if "_trade" in p]
    to_resubmit = pending + missing

    if not to_cancel and not to_resubmit:
        print("Nothing to fix — all leveraged trades completed.")
        return []

    print(f"Orders to cancel:  {len(to_cancel)}")
    print(f"Trades to resubmit: {len(to_resubmit)}")

    if not confirm:
        print("\nDRY RUN — CONFIRM=False.\n")
        if to_cancel:
            print("Would CANCEL:")
            for t in to_cancel:
                print(f"  {t['ticker']} order #{t.get('order_id', '?')}")
        if to_resubmit:
            mode = "LIMIT GTC" if use_limit_orders else "MARKET"
            print(f"\nWould RESUBMIT ({mode}, fresh prices):")
            for t in to_resubmit:
                print(f"  {t['action']:>5} {t['shares']:>6} {t['ticker']:<6}")
        return []

    from ib_insync import IB, Stock, LimitOrder, MarketOrder

    live_dir = Path(live_dir)
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=False, timeout=10)
    print(f"Connected for trading: {ib.managedAccounts()[0]}\n")

    # Cancel stale orders
    cancelled = 0
    for t in to_cancel:
        trade_obj = t.get("_trade")
        if trade_obj:
            print(f"  Cancelling {t['ticker']} order #{t.get('order_id', '?')}...", end="")
            ib.cancelOrder(trade_obj.order)
            ib.sleep(1)
            print(" done")
            cancelled += 1

    print(f"\nCancelled {cancelled} orders.\n")
    ib.sleep(2)

    # Get fresh prices
    resubmit_contracts = {}
    for t in to_resubmit:
        ticker = t["ticker"]
        if ticker not in resubmit_contracts:
            c = Stock(ticker, "SMART", "USD")
            try:
                ib.qualifyContracts(c)
                resubmit_contracts[ticker] = c
            except Exception as e:
                print(f"  {ticker}: Failed to qualify — {e}")

    fresh_prices = {}
    if resubmit_contracts:
        print("Fetching fresh prices...", end="")
        ticker_data = ib.reqTickers(*resubmit_contracts.values())
        ib.sleep(2)
        for td in ticker_data:
            sym = td.contract.symbol
            price = td.last or td.close
            if not price or price <= 0:
                price = (td.bid + td.ask) / 2 if td.bid and td.ask else None
            if price and price > 0:
                fresh_prices[sym] = price
        print(f" got {len(fresh_prices)} prices")

    # Resubmit
    exec_results = []
    print()
    for t in to_resubmit:
        ticker = t["ticker"]
        action = t["action"]
        shares = t["shares"]
        is_buy = action == "BUY"

        contract = resubmit_contracts.get(ticker)
        if not contract:
            exec_results.append({"ticker": ticker, "status": "FAILED"})
            continue

        ref_price = fresh_prices.get(ticker) or t.get("ref_price")
        if use_limit_orders and ref_price:
            buf = 1 + limit_buffer_pct / 100 if is_buy else 1 - limit_buffer_pct / 100
            limit_px = round(ref_price * buf, 2)
            order = LimitOrder(action, shares, limit_px)
            order.tif = "GTC"
            order.outsideRth = False
            effective_type = "LIMIT GTC"
        else:
            order = MarketOrder(action, shares)
            effective_type = "MARKET"
            limit_px = None

        price_str = f" @ limit ${limit_px:.2f}" if limit_px else ""
        print(f"  {action} {shares} {ticker} ({effective_type}){price_str}...", end="")
        trade_obj = ib.placeOrder(contract, order)
        ib.sleep(2)

        status = trade_obj.orderStatus.status
        fill = trade_obj.orderStatus.avgFillPrice
        print(f" {status}" + (f" @ ${fill:.2f}" if fill else ""))

        exec_results.append({
            "ticker": ticker, "status": status,
            "order_id": trade_obj.order.orderId, "fill_price": fill,
            "limit_price": limit_px, "order_type": effective_type,
        })

    # Log
    log_file = live_dir / "execution_log.csv"
    file_exists = log_file.exists()
    log_ts = datetime.now().isoformat()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "ticker", "action", "shares", "status",
                        "order_id", "fill_price", "limit_price", "order_type"])
        for r in exec_results:
            w.writerow([log_ts, r.get("ticker"), "", "",
                        r.get("status"), r.get("order_id", ""),
                        r.get("fill_price", ""), r.get("limit_price", ""),
                        r.get("order_type", "")])

    resubmitted = sum(1 for r in exec_results
                      if r["status"] in ("Filled", "Submitted", "PreSubmitted"))
    print(f"\nFix summary: cancelled {cancelled}, resubmitted {resubmitted}")

    ib.disconnect()
    return exec_results


def _save_signal_history(live_dir: Path):
    """Append current signal state to signal_history.csv."""
    # This is called after execution to record the signal that triggered the trade
    pass  # Populated by the notebook after compute_live_signals
