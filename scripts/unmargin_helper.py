"""Un-margin helper (v2, 2026-07-10).

Purpose: identify the MINIMUM set of positions to sell to raise cash to a
target level. Prioritises positions that are NOT in the current optimiser
target basket — those are genuinely inappropriate. Uses average cost + earliest
execution date as a tie-breaker (older = more likely to be legacy drift).

Steps:
1. Read-only connect to IB Gateway (client-id 80-99 band; 25s timeout).
2. Pull live snapshot (positions with market_value, avg_cost).
3. For each ticker, query ib.executions() with a 90-day lookback to find
   the OLDEST fill time still on IB's rolling log. Positions bought before
   that window show as "before 90d".
4. Rebuild the current target basket:
   - collect_prices from cache
   - score_factors (with fundamentals)
   - optimize_portfolio → target_weights
5. Rank held positions by rebalance-priority:
    P0 — not in target at all (fully inappropriate → sell first)
    P1 — in target but overweight (sell the overhang, not the whole thing)
    P2 — in target, at/under weight (keep)
6. Greedy: pick fewest P0 positions (largest first) whose market values sum
   to >= gap. If not enough, spill into P1 overhangs.
7. Print exact ticker + shares + notional plan.

Not committed for reuse — point-in-time helper for the 2026-07-10 over-invest
situation. Kept in scripts/ so the diagnostic run and recommendation are
reproducible from git.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.data_collection.issuer_fundamentals import load_fundamentals_series
from src.portfolio.ib_state import (
    DEFAULT_IB_HOST, DEFAULT_IB_PORT,
    connect_read_only, fetch_snapshot,
)
from src.portfolio.pipeline import (
    collect_prices, optimize_portfolio, score_factors,
)
from src.portfolio.policy import load_policy


TARGET_CASH = 35_000.0


def _connect_and_snapshot():
    for cid in range(80, 100):
        try:
            ib = connect_read_only(
                host=DEFAULT_IB_HOST, port=DEFAULT_IB_PORT,
                client_id=cid, timeout=25,
            )
            print(f"connected under client_id {cid}")
            snap = fetch_snapshot(ib)
            return ib, snap
        except Exception as exc:
            print(f"  cid {cid} failed: {str(exc)[:70]}")
    raise RuntimeError("could not connect to IB Gateway in band 80-99")


def _oldest_executions(ib, tickers: list[str]) -> dict[str, str]:
    """Return {ticker → earliest fill time as string} from IB's rolling
    execution log. Tickers with no execution in the log return 'pre-90d'.

    Uses ib.reqExecutions() with an empty filter so we get everything the
    log still holds (typically 90 days).
    """
    try:
        from ib_insync import ExecutionFilter
        fills = ib.reqExecutions(ExecutionFilter())
    except Exception as exc:
        print(f"  reqExecutions failed: {exc}")
        return {t: "pre-90d" for t in tickers}

    earliest: dict[str, str] = {}
    for f in fills:
        sym = getattr(f.contract, "symbol", None)
        t = getattr(f.execution, "time", None)
        if sym is None or t is None:
            continue
        cur = earliest.get(sym)
        if cur is None or str(t) < cur:
            earliest[sym] = str(t)
    return {t: earliest.get(t, "pre-90d") for t in tickers}


def _target_basket(policy) -> pd.Series:
    """Recompute the current optimiser target basket from cache."""
    processed_dir = Path.home() / "trade_data" / "ETFTrader" / "processed"
    load = collect_prices(policy, processed_dir=processed_dir)
    er, dy = load_fundamentals_series()
    scoring = score_factors(
        load.prices, policy, expense_ratios=er, dividend_yields=dy,
    )
    weights = optimize_portfolio(
        scoring, load.prices, policy, optimizer_type="rankbased",
    )
    return weights


def main() -> int:
    print("Connecting to IB Gateway (read-only)…")
    ib, snap = _connect_and_snapshot()
    try:
        print(f"NAV ${snap.nav:,.0f}, Cash ${snap.cash:,.0f}")
        gap = TARGET_CASH - snap.cash
        print(f"Need to raise ${gap:,.0f} to reach ${TARGET_CASH:,.0f}")
        print()

        positions = list(snap.long_positions)
        tickers = [p.ticker for p in positions]

        print(f"Fetching executions history for {len(tickers)} held tickers…")
        oldest = _oldest_executions(ib, tickers)

        print("Rebuilding current optimiser target basket…")
        target = _target_basket(load_policy())
        target_map: dict[str, float] = {}
        for t, w in target.items():
            if w > 0:
                target_map[t.upper()] = float(w)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    # Rank each position.
    nav = float(snap.nav)
    rows = []
    for p in positions:
        tkr = p.ticker.upper()
        tgt_w = target_map.get(tkr, 0.0)
        cur_w = p.market_value / nav if nav > 0 else 0.0
        overhang_pct = cur_w - tgt_w  # positive = overweight
        overhang_usd = overhang_pct * nav
        priority = 0 if tgt_w == 0.0 else (1 if overhang_pct > 0 else 2)
        rows.append({
            "ticker": p.ticker,
            "shares": p.shares,
            "price": p.market_price,
            "mv": p.market_value,
            "avg_cost": p.avg_cost,
            "cur_w": cur_w,
            "tgt_w": tgt_w,
            "overhang_usd": overhang_usd,
            "priority": priority,
            "oldest_fill": oldest.get(p.ticker, "pre-90d"),
        })

    # Print full ranked table.
    df = pd.DataFrame(rows)
    df = df.sort_values(["priority", "mv"], ascending=[True, False])

    print()
    print("Position ranking — P0=not in target, P1=overweight, P2=on/under target")
    print(f"{'P':>2}  {'Ticker':<7}{'Shares':>8}{'Mkt val':>11}"
          f"{'Cur %':>8}{'Tgt %':>8}{'Overhang $':>13}  Oldest fill")
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"P{r.priority:<1}  {r.ticker:<7}{r.shares:>8.0f}"
              f"${r.mv:>10,.0f}{r.cur_w*100:>7.2f}%{r.tgt_w*100:>7.2f}%"
              f"${r.overhang_usd:>+12,.0f}  {r.oldest_fill[:19]}")

    # ── Plan A: PURE consistency with the rebalance ─────────────
    # Full exit all P0 (rebalance wanted these gone), trim P1 to
    # exact target weight (rebalance wanted these smaller).
    # Then report whether it clears the gap.
    print()
    print("── PLAN A: PURE rebalance-consistent ──")
    print("  Full-exit every P0; trim each P1 down to exact target weight.")
    plan_a = []
    plan_a_total = 0.0
    for _, r in df[df.priority == 0].iterrows():
        plan_a.append({
            "ticker": r.ticker, "shares": int(r.shares),
            "notional": r.mv, "kind": "FULL EXIT",
        })
        plan_a_total += r.mv
    for _, r in df[df.priority == 1].iterrows():
        if r.overhang_usd < 500:  # skip tiny overhangs (below min notional)
            continue
        trim_shares = int(r.overhang_usd / r.price)
        if trim_shares <= 0:
            continue
        trim_notional = trim_shares * r.price
        plan_a.append({
            "ticker": r.ticker, "shares": trim_shares,
            "notional": trim_notional,
            "kind": f"trim (of {int(r.shares)} held)",
        })
        plan_a_total += trim_notional

    for i, s in enumerate(plan_a, 1):
        print(f"  {i:>2}. SELL {s['ticker']:<6} {s['shares']:>5} shares "
              f"~${s['notional']:>10,.0f}  {s['kind']}")
    plan_a_cash = snap.cash + plan_a_total
    print(f"  Total: {len(plan_a)} trades raising ~${plan_a_total:,.0f} → "
          f"cash ~${plan_a_cash:,.0f}  (target ${TARGET_CASH:,.0f})")
    if plan_a_cash >= TARGET_CASH:
        print("  ✓ Plan A alone reaches the target.")
    else:
        print(f"  ✗ Plan A short by ~${TARGET_CASH - plan_a_cash:,.0f}")

    # ── Plan B: MINIMUM trades ──────────────────────────────────
    # Full exit every P0, then full-exit largest P1s until we clear.
    # Breaks rebalance consistency (over-sells some target positions)
    # but keeps trade count as low as possible.
    print()
    print("── PLAN B: MINIMUM trade count ──")
    print("  Full-exit every P0; then full-exit largest P1s until gap cleared.")
    plan_b = []
    plan_b_total = 0.0
    for _, r in df[df.priority == 0].iterrows():
        plan_b.append({
            "ticker": r.ticker, "shares": int(r.shares),
            "notional": r.mv, "kind": "FULL EXIT (P0 — not in target)",
        })
        plan_b_total += r.mv
    if plan_b_total < gap:
        for _, r in df[df.priority == 1].sort_values("mv", ascending=False).iterrows():
            if plan_b_total >= gap:
                break
            plan_b.append({
                "ticker": r.ticker, "shares": int(r.shares),
                "notional": r.mv,
                "kind": f"FULL EXIT (P1 — {(r.cur_w - r.tgt_w)*100:+.2f}% overhang)",
            })
            plan_b_total += r.mv

    for i, s in enumerate(plan_b, 1):
        print(f"  {i:>2}. SELL {s['ticker']:<6} {s['shares']:>5} shares "
              f"~${s['notional']:>10,.0f}  {s['kind']}")
    plan_b_cash = snap.cash + plan_b_total
    print(f"  Total: {len(plan_b)} trades raising ~${plan_b_total:,.0f} → "
          f"cash ~${plan_b_cash:,.0f}  (target ${TARGET_CASH:,.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
