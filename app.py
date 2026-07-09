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
from src.portfolio.policy import DEFAULT_POLICY_PATH, load_policy


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
    "prices": None,
    "factor_scores": None,
    "target_weights": None,
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
        ("2 · Data collected", st.session_state["prices"] is not None),
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
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("NAV", f"${snap.nav:,.0f}")
        m2.metric("Cash", f"${snap.cash:,.0f}")
        m3.metric("Buying Power", f"${snap.buying_power:,.0f}")
        m4.metric("Unrealized P&L", f"${snap.unrealized_pnl_reported:,.0f}")
        m5.metric("Realized P&L", f"${snap.realized_pnl_reported:,.0f}")
        st.caption(
            f"Account **{snap.account}** · "
            f"Snapshot at {snap.timestamp:%Y-%m-%d %H:%M:%S %Z} · "
            f"P&L values are as reported by IB `accountSummary` "
            f"(labelling may reflect day/YTD depending on IB config)."
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
        "Pulls the latest daily bars into `~/trade_data/ETFTrader/processed`. "
        "The pipeline function will live in `src/portfolio/pipeline.py` "
        "(extracted from `notebooks/scripts/s2_collect.py`)."
    )
    st.button("Run collection", disabled=True, help="Wired in slice 3")


with tab_optimize:
    st.header("Step 3 — score factors + optimise")
    st.markdown(
        f"Applies factor blend "
        f"({', '.join(f'{n}={w:.0%}' for n, w in policy.factor_weights.as_dict().items())}) "
        f"then runs the cvxpy optimiser with min/max weight "
        f"`{policy.min_weight:.0%}`/`{policy.max_weight:.0%}` and "
        f"`{policy.num_positions}` target positions."
    )
    st.button("Run optimisation", disabled=True, help="Wired in slice 3")


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
