"""ETFTrader — portfolio applet (ADR-0001).

Local Streamlit operating surface for the ETF smart-beta strategy.
Replaces the s1-s7 notebook ritual.

Run:
    streamlit run app.py

Architecture (per ADR-0001 and CLAUDE.md):
    - Deterministic core decides. The applet calls into `src/` for all
      numbers — factor scoring, optimizer weights, order payloads.
    - LLM narrates. Steps 5 (Explain) uses an LLM to describe *why*,
      but never sets a size or places an order.
    - Guarded apply. Step 6 (BIG switch) requires an explicit confirm
      and can only run when the IB Gateway Read-Only API flag is off.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.portfolio.ib_state import (
    DEFAULT_IB_CLIENT_ID,
    DEFAULT_IB_HOST,
    DEFAULT_IB_PORT,
    DEFAULT_NAV_HISTORY_PATH,
    append_nav_snapshot,
    connect_read_only,
    fetch_snapshot,
    load_nav_history,
)
from src.portfolio.pipeline import (
    DEFAULT_IB_CACHE_DIR,
    cache_status,
    collect_prices,
    optimize_portfolio,
    portfolio_hhi,
    portfolio_volatility,
    refresh_prices_from_ib,
    score_factors,
)
from src.portfolio.policy import DEFAULT_POLICY_PATH, load_policy

DEFAULT_PROCESSED_DIR = Path.home() / "trade_data" / "ETFTrader" / "processed"


# ────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ETFTrader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────────
# Policy — loaded once per session
# ────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_policy():
    return load_policy()


try:
    policy = _load_policy()
except Exception as exc:
    st.error(f"Failed to load policy from {DEFAULT_POLICY_PATH}: {exc}")
    st.stop()


# ────────────────────────────────────────────────────────────────
# Session state defaults
# ────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "cash_budget": 0.0,
    "ib_snapshot": None,
    "price_load": None,
    "scoring": None,
    "target_weights": None,
    "opt_diagnostics": None,
    "proposed_trades": None,
    "explanation": None,
    "execution_receipt": None,
}
for key, default in _DEFAULTS.items():
    st.session_state.setdefault(key, default)


# ────────────────────────────────────────────────────────────────
# Sidebar — policy summary + step navigator status
# ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Policy")
    st.caption(f"`{DEFAULT_POLICY_PATH.name}` — edit + restart to change")
    st.markdown(
        f"**{policy.name}** (v{policy.version})  \n"
        f"Positions: `{policy.num_positions}` "
        f"({policy.min_weight:.0%}–{policy.max_weight:.0%} each)  \n"
        f"Rebalance: `{policy.rebalance_frequency}` "
        f"@ {policy.drift_threshold:.0%} drift  \n"
        f"Stops: `{policy.entry_stop_loss_pct:.0%}` entry, "
        f"`{policy.trailing_stop_pct:.0%}` trail  \n"
        f"Cash reserve: `${policy.cash_reserve:,.0f}`"
    )

    st.markdown("**Factor blend**")
    fw = policy.factor_weights.as_dict()
    for name, weight in sorted(fw.items(), key=lambda kv: -kv[1]):
        st.markdown(f"- {name.title()}: `{weight:.0%}`")

    st.divider()
    st.caption("Progress")
    _step_status = [
        ("1 · Cash budget", st.session_state["cash_budget"] > 0 or st.session_state["ib_snapshot"] is not None),
        ("2 · Data collected", st.session_state["price_load"] is not None),
        ("3 · Optimised", st.session_state["target_weights"] is not None),
        ("4 · Trades proposed", st.session_state["proposed_trades"] is not None),
        ("5 · Explained", st.session_state["explanation"] is not None),
        ("6 · Sent to IBKR", st.session_state["execution_receipt"] is not None),
        ("7 · Summary", st.session_state["execution_receipt"] is not None),
    ]
    for label, done in _step_status:
        st.markdown(f"{'✅' if done else '⬜️'} {label}")


# ────────────────────────────────────────────────────────────────
# Main — seven tabs, one per step
# ────────────────────────────────────────────────────────────────

st.title("ETFTrader — smart-beta operator")
st.caption(
    "Deterministic pipeline (see `src/`) + LLM narration. "
    "Nothing goes to IBKR until you flip the BIG switch on tab 6."
)

tab_setup, tab_collect, tab_optimize, tab_propose, tab_explain, tab_send, tab_summary = st.tabs(
    [
        "1 · Cash",
        "2 · Collect",
        "3 · Optimise",
        "4 · Propose",
        "5 · Explain",
        "6 · Send",
        "7 · Summary",
    ]
)


with tab_setup:
    st.header("Step 1 — cash budget + live IBKR snapshot")
    st.markdown(
        "Enter the amount of **additional** cash you want to deploy on top of "
        "the current portfolio. Leave at `0` to run a pure rebalance. Then "
        "pull a **read-only** snapshot of your live IB account below."
    )

    cash_col, ib_col = st.columns([1, 2])
    with cash_col:
        st.number_input(
            "Additional cash (USD)",
            min_value=0.0,
            step=1000.0,
            format="%.0f",
            key="cash_budget",
        )

    with ib_col:
        st.markdown("**IB Gateway (read-only)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            ib_host = st.text_input("Host", value=DEFAULT_IB_HOST)
        with c2:
            ib_port = st.number_input(
                "Port", value=DEFAULT_IB_PORT, step=1, format="%d",
                help="4001 live, 4002 paper",
            )
        with c3:
            ib_client_id = st.number_input(
                "Client ID", value=DEFAULT_IB_CLIENT_ID, step=1, format="%d",
                help="Applet reserves 30; script defaults use 1/2/5/15/22.",
            )

        if st.button("Pull snapshot", type="primary"):
            with st.spinner("Connecting to IB Gateway..."):
                try:
                    ib = connect_read_only(
                        host=ib_host,
                        port=int(ib_port),
                        client_id=int(ib_client_id),
                    )
                    snap = fetch_snapshot(ib)
                    ib.disconnect()
                    st.session_state["ib_snapshot"] = snap
                    hist = append_nav_snapshot(snap)
                    st.session_state["nav_history"] = hist
                    st.success(
                        f"Connected to {snap.account} — "
                        f"{len(snap.positions)} positions, "
                        f"{len(snap.open_orders)} open orders."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["ib_snapshot"] = None
                    st.error(f"IB connection failed: {exc}")

    snap = st.session_state.get("ib_snapshot")
    if snap is not None:
        st.divider()

        # Row 1 — top-line financial state (mirrors Client Portal top strip)
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        r1c1.metric("NAV", f"${snap.nav:,.0f}")
        r1c2.metric("Cash", f"${snap.cash:,.0f}")
        r1c3.metric("Buying Power", f"${snap.buying_power:,.0f}")
        r1c4.metric(
            "Daily P&L",
            "—" if snap.daily_pnl != snap.daily_pnl else f"${snap.daily_pnl:,.0f}",
        )
        r1c5.metric("Unrealized P&L", f"${snap.unrealized_pnl_reported:,.0f}")

        # Row 2 — margin / liquidity health
        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
        r2c1.metric("Realized P&L", f"${snap.realized_pnl_reported:,.0f}")
        r2c2.metric("Excess Liquidity", f"${snap.excess_liquidity:,.0f}")
        r2c3.metric("Available Funds", f"${snap.available_funds:,.0f}")
        r2c4.metric("Maintenance", f"${snap.maint_margin:,.0f}")
        r2c5.metric("Initial Margin", f"${snap.init_margin:,.0f}")

        st.caption(
            f"Account **{snap.account}** · "
            f"Snapshot at {snap.timestamp:%Y-%m-%d %H:%M:%S %Z} · "
            f"P&L values are as reported by IB (`Daily` from `reqPnL`, "
            f"`Realized`/`Unrealized` from `accountSummary`)."
        )

        st.subheader("Equity curve (local snapshots)")
        hist = st.session_state.get("nav_history")
        if hist is None:
            hist = load_nav_history()
        if hist.empty or len(hist) < 2:
            st.caption(
                f"Only {len(hist)} snapshot(s) so far — the curve will build "
                f"as you run the applet on subsequent days. Stored at "
                f"`{DEFAULT_NAV_HISTORY_PATH}`."
            )
        else:
            st.line_chart(hist[["nav", "cash"]])

        pos_col, ord_col = st.columns(2)
        with pos_col:
            st.subheader(f"Positions ({len(snap.positions)})")
            st.dataframe(
                snap.positions_df(),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "avg_cost": st.column_config.NumberColumn(format="$%.2f"),
                    "market_price": st.column_config.NumberColumn(format="$%.2f"),
                    "market_value": st.column_config.NumberColumn(format="$%.0f"),
                    "daily_pnl": st.column_config.NumberColumn(format="$%.0f"),
                    "unrealized_pnl": st.column_config.NumberColumn(format="$%.0f"),
                    "unrealized_pct": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )
        with ord_col:
            st.subheader(f"Open orders ({len(snap.open_orders)})")
            n_trails = len(snap.open_trails)
            if n_trails:
                st.caption(
                    f"🛡️ {n_trails} protective TRAIL(s) already in place — "
                    f"the Propose panel will layer new TRAILs only on **new** shares."
                )
            st.dataframe(
                snap.orders_df(),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info(
            "Snapshot not yet loaded. Make sure IB Gateway is running and "
            "**Read-Only API is on** (Global Config → API → Settings), then "
            "click *Pull snapshot*."
        )


with tab_collect:
    st.header("Step 2 — collect ETF prices")
    st.markdown(
        f"Loads the ETF price matrix from "
        f"`{DEFAULT_PROCESSED_DIR}` and applies the quality filter "
        f"(min {policy.factor_lookbacks.momentum} days of history, "
        f"less than 10% missing bars). Priority order: Databento → IB → yfinance. "
        f"The daily cron keeps this current — refresh from IB Gateway "
        f"on-demand if you need fresh data right now."
    )

    # ── Cache status ────────────────────────────────────────
    with st.spinner("Scanning per-ticker cache…"):
        try:
            status = cache_status(cache_dir=DEFAULT_IB_CACHE_DIR)
        except Exception as exc:  # noqa: BLE001
            status = None
            st.warning(f"Cache scan failed: {exc}")
    if status is not None:
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("Universe", status["n_universe"])
        cs2.metric("Current", status["n_current"])
        cs3.metric("Stale", status["n_stale"])
        cs4.metric("Missing", status["n_missing"])
        if status["latest_bar"] is not None:
            st.caption(
                f"Latest cached bar across all tickers: "
                f"**{status['latest_bar']:%Y-%m-%d}** · "
                f"per-ticker files under `{DEFAULT_IB_CACHE_DIR}`"
            )

    # ── Config + actions ─────────────────────────────────────
    ac1, ac2 = st.columns([2, 1])
    with ac1:
        processed_dir = st.text_input(
            "Processed data directory", value=str(DEFAULT_PROCESSED_DIR),
            help="Directory containing etf_prices_{db,ib,filtered}.parquet",
        )
    with ac2:
        refresh_from_ib = st.checkbox(
            "Refresh from IB before load",
            value=False,
            help=(
                "Runs IBDataCollector against IB Gateway (client_id 31). "
                "Skips tickers already current; only stale/missing incur an "
                "IB request. A fully-current cache costs seconds; a stale "
                "cache costs minutes; a missing cache costs hours."
            ),
        )

    if st.button("Load / refresh price matrix", type="primary"):
        # Optional IB refresh first.
        if refresh_from_ib:
            work = (status or {}).get("n_stale", 0) + (status or {}).get("n_missing", 0)
            with st.status(
                f"Refreshing {work} tickers from IB Gateway…",
                expanded=True,
            ) as status_box:
                progress = st.progress(0.0)
                progress_label = st.empty()

                def _cb(i: int, total: int, ticker: str):
                    progress.progress(min(i / max(total, 1), 1.0))
                    progress_label.markdown(f"`{i}/{total}` — {ticker}")

                try:
                    result = refresh_prices_from_ib(
                        processed_dir=Path(processed_dir),
                        progress_callback=_cb,
                    )
                    status_box.update(
                        label=(
                            f"IB refresh done — {result.n_current} current, "
                            f"{result.n_stale} stale, {result.n_missing} missing."
                        ),
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    status_box.update(
                        label=f"IB refresh failed: {exc}",
                        state="error",
                    )
                    st.stop()

        with st.spinner("Loading prices…"):
            try:
                load = collect_prices(policy, processed_dir=Path(processed_dir))
                st.session_state["price_load"] = load
                # Invalidate downstream artefacts.
                for k in ("scoring", "target_weights", "opt_diagnostics",
                          "proposed_trades", "explanation"):
                    st.session_state[k] = None
                st.success(
                    f"Loaded `{load.source}` — {load.n_tickers} tickers, "
                    f"{load.start_date:%Y-%m-%d} to {load.end_date:%Y-%m-%d}."
                )
            except (FileNotFoundError, ValueError) as exc:
                st.session_state["price_load"] = None
                st.error(str(exc))

    load = st.session_state.get("price_load")
    if load is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers", load.n_tickers)
        c2.metric("Trading days", len(load.prices))
        c3.metric("Start", f"{load.start_date:%Y-%m-%d}")
        c4.metric("End", f"{load.end_date:%Y-%m-%d}")
        # Both tz-naive; end_date comes from the parquet index (also tz-naive).
        age_days = (pd.Timestamp.now().normalize() - load.end_date.normalize()).days
        if age_days > 3:
            st.warning(
                f"Latest bar in loaded matrix is {age_days} days old — tick "
                f"*Refresh from IB before load* and re-run if you need current data."
            )
        with st.expander("Latest prices (last 5 rows, first 10 tickers)"):
            st.dataframe(load.prices.iloc[-5:, :10])


with tab_optimize:
    st.header("Step 3 — score factors + optimise")
    st.markdown(
        f"Applies factor blend "
        f"({', '.join(f'{n}={w:.0%}' for n, w in policy.factor_weights.as_dict().items())}) "
        f"and runs the selected optimiser. Weight bounds "
        f"`{policy.min_weight:.0%}`–`{policy.max_weight:.0%}` per position; "
        f"target `{policy.num_positions}` positions."
    )

    load = st.session_state.get("price_load")
    if load is None:
        st.info("Load the price matrix in tab 2 first.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            optimizer_type = st.selectbox(
                "Optimiser",
                ["rankbased", "mvo", "minvar", "simple"],
                help=(
                    "rankbased = exponential rank weights (default in the scripts). "
                    "mvo = Robust Mean-Variance (matches the tech doc). "
                    "minvar = Min-variance. simple = equal-weight top N."
                ),
            )
        with c2:
            st.caption(
                "The tech document's canonical strategy uses `mvo`. The scripts "
                "default to `rankbased`. This selector is intentional — the "
                "diagnostic in `docs/RESEARCH_BACKLOG.md` #1 will resolve which "
                "the applet should default to."
            )

        if st.button("Score + optimise", type="primary"):
            with st.spinner("Scoring factors + running optimiser…"):
                try:
                    scoring = score_factors(load.prices, policy)
                    weights = optimize_portfolio(
                        scoring, load.prices, policy, optimizer_type=optimizer_type,
                    )
                    diagnostics = {
                        "vol": portfolio_volatility(weights, load.prices),
                        "hhi": portfolio_hhi(weights),
                        "max_weight": float(weights.max()),
                        "min_weight": float(weights[weights > 0].min()),
                        "optimizer_type": optimizer_type,
                        "active_weights": scoring.active_weights,
                    }
                    st.session_state["scoring"] = scoring
                    st.session_state["target_weights"] = weights
                    st.session_state["opt_diagnostics"] = diagnostics
                    for k in ("proposed_trades", "explanation"):
                        st.session_state[k] = None
                    st.success(
                        f"Target portfolio built — {len(weights)} positions, "
                        f"HHI {diagnostics['hhi']:.4f}."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Optimisation failed: {exc}")

    weights = st.session_state.get("target_weights")
    diag = st.session_state.get("opt_diagnostics")
    scoring = st.session_state.get("scoring")
    if weights is not None and diag is not None:
        st.divider()
        st.subheader("Target portfolio")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Positions", len(weights))
        m2.metric("Max weight", f"{diag['max_weight']:.2%}")
        m3.metric("Min weight", f"{diag['min_weight']:.2%}")
        m4.metric("HHI", f"{diag['hhi']:.4f}")
        vol_display = "—" if diag["vol"] != diag["vol"] else f"{diag['vol']:.1%}"
        m5.metric("Ex-ante vol", vol_display)

        aw = diag["active_weights"]
        if set(aw.keys()) != set(policy.factor_weights.as_dict().keys()):
            st.caption(
                "⚠️ Value factor skipped (no expense-ratio data). Weights used: "
                + ", ".join(f"{n}={w:.0%}" for n, w in aw.items())
            )

        holdings = pd.DataFrame({
            "ticker": weights.index,
            "target_weight": weights.values,
        }).sort_values("target_weight", ascending=False).reset_index(drop=True)

        if scoring is not None:
            factor_cols = list(scoring.factor_scores.columns)
            for col in factor_cols:
                holdings[f"score_{col}"] = holdings["ticker"].map(
                    scoring.factor_scores[col]
                )

        st.dataframe(
            holdings,
            hide_index=True,
            use_container_width=True,
            column_config={
                "target_weight": st.column_config.NumberColumn(format="%.2f%%"),
                **{
                    f"score_{c}": st.column_config.NumberColumn(format="%.2f")
                    for c in (scoring.factor_scores.columns if scoring else [])
                },
            },
        )


with tab_propose:
    st.header("Step 4 — proposed trades")
    st.markdown(
        "Blotter with **Buy** (fresh position), **Sell** (reduce / exit), and "
        "**Extend** (add to existing). Every BUY / EXTEND gets a "
        f"`{policy.trailing_stop_pct:.0%}` trailing stop; existing TRAILs on "
        "held positions are never cancelled or rebased — Extends layer a new "
        "TRAIL covering only the new shares."
    )
    st.button("Generate proposal", disabled=True, help="Wired in slice 4")


with tab_explain:
    st.header("Step 5 — why these trades?")
    st.markdown(
        "The LLM narrates the deterministic proposal — which factors are "
        "driving each pick, what the portfolio is tilting toward (e.g. "
        "*quality*), and what changed vs. the prior snapshot. "
        "**The LLM does not set sizes or place orders.**"
    )
    st.info("Wired after read-only slices (ADR-0001 action item #4).")


with tab_send:
    st.header("Step 6 — BIG switch → IBKR")
    st.error(
        "⚠️ **This is the only step that talks to your live IB account.**  \n"
        "Requires: IB Gateway Read-Only API **off**, and an explicit confirm."
    )
    st.info("Wired last (ADR-0001 action item #5).")


with tab_summary:
    st.header("Step 7 — post-trade summary")
    st.markdown(
        "After the trades have been sent, this tab shows the resulting "
        "portfolio, the new trailing stops in place, and a diff vs. the "
        "state before the run."
    )
    st.info("Populated automatically once step 6 completes.")


st.divider()
st.caption(
    f"Policy loaded from `{DEFAULT_POLICY_PATH.name}` · "
    f"Deterministic core: `src/` · "
    f"See `docs/ADR-0001-portfolio-applet.md` for the design."
)
