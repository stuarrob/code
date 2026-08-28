# ADR-0002: Standalone leveraged sleeve applet --- end-to-end

**Status:** Proposed
**Date:** 2026-07-11
**Deciders:** Stuart (owner)

> **Scope:** the leveraged ETF strategy as specified in `docs/LEVERAGED_STRATEGY_DOCUMENT.tex` v4.0. **Standalone applet, separate from the smart-beta applet (ADR-0001).** This single ADR covers architecture, paper trading, recalibration, live operation, alerts, and monitoring --- front to back.

## Context

The v4.0 leveraged strategy lands with a fully-validated recommended stack (TECL + ROM + NAIL 33/33/33, SMA + vol filter, VVIX overlay, SGOV cash). It needs its own operating surface --- separate from the smart-beta applet --- because:

- **Different risk posture.** Leveraged sleeve carries 3x-and-2x exposure with tail-risk features (VVIX overlay, halving triggers) that the smart-beta app doesn't need to reason about. Mixing UIs risks conceptual confusion.
- **Different signal cadence.** Smart-beta rebalances quarterly on factor scoring. Leveraged has weekly signal checks and daily VVIX monitoring. Different mental model.
- **Different failure modes.** Leveraged sleeve can miss a signal transition and lose 10 pp of drawdown protection in a single week; smart-beta's failure modes are much slower. Keeps the operator's alert budget clean.
- **Different lifecycle.** Leveraged is going through paper trading and staged live rollout; smart-beta is already partially live. Different gates, different confidence levels.

The user's explicit instruction: **"I want a separate app for this strategy and this strategy alone --- it needs to sit entirely outside of the smart beta."**

**Constraints:**

- **Live money at 10% NAV cap.** Not 20%. Kelly-quarter (not Kelly-half). The operator is dialling back the sizing recommendation from the tex doc.
- **Long only, cash buffer required.** No margin, no shorts (leveraged ETFs already provide the leverage synthetically).
- **Deterministic core decides, LLM (optional) explains.** Same operating principle as ADR-0001. No LLM in the signal path.
- **Manual monthly rebalance at minimum for 6+ months.** No automation before paper trading + manual live are both clean.
- **Alerts are non-negotiable.** The operator cannot maintain a signal-driven strategy without automated monitoring. Telegram (primary) + email (backup).

## Decision

Build a **standalone Streamlit applet at `leveraged_app/app.py`**, sharing the ETFTrader repo's `src/leveraged/` engine but with its own UI, its own cash budget, its own audit trail, and its own daily/weekly/monthly cron jobs.

The applet operates in three staged phases (paper $\to$ manual live $\to$ semi-auto). A quarterly recalibration cycle keeps the signal parameters honest. Alerts drive daily attention.

## Design

### Repo layout

```
ETFTrader/
├── app.py                          # smart-beta applet (unchanged)
├── src/leveraged/                  # shared engine (unchanged)
│   ├── signals.py                  # SMA + vol + VVIX
│   ├── strategy.py                 # allocation, rebalancing
│   ├── backtest.py                 # recalibration engine
│   └── universe.py                 # TECL, ROM, NAIL, SGOV pairs
├── leveraged_app/                  # NEW --- the standalone applet
│   ├── app.py                      # Streamlit entry point
│   ├── engine.py                   # thin wrapper over src/leveraged
│   ├── proposal.py                 # trade proposals + cash-cap logic
│   ├── execution.py                # IB order placement (LMT+GTC)
│   ├── alerts.py                   # Telegram + email
│   ├── monitoring.py               # signal state, VVIX daily
│   └── recalibrate.py              # quarterly backtest re-run
├── scripts/
│   ├── leveraged_daily.py          # cron: daily VVIX + drawdown check
│   ├── leveraged_weekly.py         # cron: Friday signal check
│   ├── leveraged_monthly.py        # cron: monthly review report
│   └── leveraged_quarterly.py      # cron: recalibration
└── tests/test_leveraged_app/
    ├── test_signal_state.py
    ├── test_proposal.py            # cash-cap invariant (see below)
    ├── test_alerts.py
    └── test_recalibrate.py
```

### Applet UI --- 6-step flow

Simpler than the smart-beta 7-step because there is less to configure:

1. **State panel** --- current sleeve NAV, cash, signal state (RISK\_ON/OFF), VVIX overlay state, next scheduled rebalance.
2. **Refresh data** --- pull latest TECL, ROM, NAIL, XLK, ITB, SGOV, VIX, VVIX prices; show freshness.
3. **Compute signal** --- deterministic. Shows: SMA-200 vs price for XLK and ITB, 20-day vol vs 22\%, VVIX vs 100, resulting signal state, resulting target weights.
4. **Propose trades** --- deterministic. Reads current positions, compares to targets, generates BUY/SELL/EXTEND list. Enforces cash-neutrality cap (see below).
5. **Explain (optional LLM)** --- narrates the proposal in plain English. Skippable.
6. **Guarded Apply** --- big red switch. Confirms all orders, submits LMT+GTC, writes to audit log.

### Cash budget --- fully independent from smart-beta

The leveraged applet has its **own cash budget line item**. It does not share the smart-beta applet's cash pool. On IB, this is enforced by convention (the operator allocates a fixed dollar amount to the leveraged sleeve at the start of each month; the applet works within that budget).

**Sizing rule (updated from tex v4.0):**

$$w_{\text{leveraged}} = \min\left(\text{Kelly-quarter}, \; 0.10 \times \text{NAV}\right)$$

At the recommended stack's Sharpe (1.34) and vol (21.6\%), Kelly-quarter is $\sim$155\% of NAV. The **10\% NAV cap binds**. Because sizing is expressed as a percentage of NAV and the operator invests continuously, this rule adjusts automatically with capital and does not lock in specific dollar amounts.

**Cash-neutrality invariant** (same shape as smart-beta):

```
sum(BUY + EXTEND) - sum(SELL) <= sleeve_cash - reserve_buffer
```

with a \$1000 reserve buffer to prevent zero-cash edge cases. Enforced by `leveraged_app/proposal.py` post-processing. Any proposal that would exceed the cap gets its BUYs scaled proportionally; a warning is added to the proposal.

**Four pinning tests** in `tests/test_leveraged_app/test_proposal.py` (analogous to the smart-beta suite):

1. `test_normal_proposal_within_budget` --- clean proposal is unchanged.
2. `test_overcap_proposal_scaled` --- proposals over the cap scale BUYs down proportionally.
3. `test_scale_preserves_relative_weights` --- when the cap bites, TECL:ROM:NAIL ratio is preserved.
4. `test_warning_emitted_when_capped` --- operator can see the cap engaged.

### Reduction triggers below 10\% NAV

| Trigger | Action |
|:-|:-|
| Smart-beta regime overlay $\to$ risk-off | Scale sleeve to 5\% NAV |
| VVIX at close $>$ 130 | Scale sleeve to 2.5\% NAV or fully out |
| Running sleeve DD $>$ 25\% from HWM | Freeze until DD $<$ 15\% |
| Compliance / life event | Manual override at operator's discretion |

## Operational phases

### Phase 1 --- Paper trading (2 months minimum)

**Setup:**
- IB Gateway configured with paper credentials, port 4002.
- `leveraged_app/config.py` set to `IB_PORT=4002`, `PHASE='paper'`.
- Audit path: `~/trade_data/ETFTrader/paper/leveraged_{YYYYMMDD}.jsonl`.
- Telegram bot registered.

**Kickoff Friday routine (once):**
1. Refresh data: `python scripts/leveraged_daily.py` (populates cache).
2. Launch: `streamlit run leveraged_app/app.py`.
3. Run steps 1--5 in the UI; sanity check signal state matches your read of the market.
4. Press Guarded Apply.
5. Verify orders appear in paper account and fill.

**Weekly Friday routine (every week):**
1. Cron fires `python scripts/leveraged_weekly.py` at Friday 3:30 PM local. Applet computes signal.
2. If no change: log-only, no action, Telegram silent.
3. If signal transition: Telegram alert, operator opens applet within 30 minutes to review + confirm.
4. If VVIX overlay change: Telegram alert, applet auto-proposes halving/restoration.

**Daily monitoring:**
- Cron fires `python scripts/leveraged_daily.py` at 5 PM local.
- Checks VVIX close, sleeve NAV, position drift.
- Telegram alerts per the table below.

**Paper trading exit criteria (all must hold):**
- [ ] $\geq$ 8 weeks elapsed.
- [ ] $\geq$ 1 signal transition observed and handled cleanly (OR 12 weeks elapsed regardless).
- [ ] $\geq$ 1 VVIX overlay trigger observed and handled cleanly (OR 12 weeks elapsed regardless).
- [ ] Audit log reconciles 100\% with paper account activity.
- [ ] Zero applet crashes on signal-firing days.

### Phase 2 --- Manual live (3 months minimum)

**Setup:**
- `IB_PORT=7496` (live) or `4001` (live paper, if paper Gateway kept for testing).
- `PHASE='manual_live'`.
- Confirm sleeve NAV = 10\% of total. On \$400k: \$40k.
- Audit path: `~/trade_data/ETFTrader/audit/leveraged_{YYYYMMDD}.jsonl`.
- Reduce Telegram alert urgency thresholds by 20\% (more sensitive alerts during ramp).

**Weekly and monthly routines:** same as Phase 1, but every trade the operator confirms via Apply is a real-money trade. Slower deliberation on each proposal; no auto-anything.

**Live exit criteria for Phase 3:**
- [ ] $\geq$ 3 months elapsed.
- [ ] $\geq$ 2 signal transitions handled cleanly (OR 6 months elapsed).
- [ ] Zero orders rejected due to preventable operator error.
- [ ] Live drawdown behaviour matches backtest within $\pm$ 10 pp.
- [ ] Live Sharpe within $\pm$ 0.20 of backtest expectation.

### Phase 3 --- Semi-automated (endpoint)

**On signal transition** (from cron):
1. Applet computes proposal.
2. Sends Telegram alert with a summary + one-click confirm URL (localhost tunnel).
3. Waits 30 minutes.
4. If operator confirms: submits LMT+GTC to live IB.
5. If operator vetoes: stands down, logs the veto.
6. If operator doesn't respond: submits anyway (default trust the strategy).

**Monthly rebalance stays manual.** The operator opens the applet on the first Friday of the month and presses Apply. This is a deliberate checkpoint --- keeps operator eyes on the strategy every month.

## Alerts

The alert system is the operational backbone of Phase 2 and Phase 3. All alerts go to Telegram (primary) with email as backup for critical events.

### Alert table

| Trigger | Channel | Urgency | Body |
|:-|:-:|:-:|:-|
| VVIX close $\geq$ 95 (approaching RISK\_MID) | Telegram | Low | Warning: VVIX \{val\}. Approaching RISK\_MID. Overlay may fire on Friday close. |
| VVIX state RISK\_UP $\to$ RISK\_MID (Friday close) | Telegram + Email | Medium | State change: RISK\_MID. Halve leveraged sleeve to 50\% of cap on Monday. |
| VVIX state RISK\_MID $\to$ RISK\_UP | Telegram | Medium | State change: RISK\_UP. Restore leveraged sleeve to 100\% of cap on Monday. |
| VVIX state RISK\_MID $\to$ RISK\_DOWN | Telegram + Email | High | State change: RISK\_DOWN (deep-stress). Reduce leveraged sleeve to 25\% of cap on Monday. |
| VVIX state RISK\_DOWN $\to$ RISK\_MID | Telegram | Medium | State change: RISK\_MID. Restore to 50\% of cap on Monday. |
| Signal state flip (RISK\_ON $\leftrightarrow$ RISK\_OFF) | Telegram + Email | High | \{ticker\}: signal flipped to \{state\}. Applet has proposal ready. |
| Weekly cron didn't fire | Telegram + Email | High | Weekly signal check missed. Investigate. |
| Daily cron didn't fire | Telegram | Medium | Daily VVIX check missed. |
| Running sleeve DD $>$ 15\% | Telegram | Low | Sleeve DD \{val\}. Monitoring; no action. |
| Running sleeve DD $>$ 20\% | Telegram + Email | High | Sleeve DD \{val\}. Approaching freeze trigger. |
| Running sleeve DD $>$ 25\% | Telegram + Email | Critical | FREEZE trigger reached. Sleeve trading suspended until DD $<$ 15\%. |
| Order rejection | Telegram + Email | High | Order \{id\} \{ticker\} rejected: \{reason\}. |
| Order stuck in "unknown" for 30 min | Telegram + Email | High | Order \{id\} status unresolved. Manual check needed. |
| Live Sharpe (rolling 3-month) diverges $>$ 0.30 from backtest | Telegram + Email | High | Live Sharpe \{val\} vs backtest expectation \{exp\}. Review needed. |

### Alert budget

The operator should expect roughly:
- **1-5 alerts/week during Phase 1** (many "state OK" heartbeats for calibration).
- **1-2 alerts/week during Phase 2/3 steady state**.
- **5-15 alerts during a genuine stress event** (2018, 2020, 2022).

If the alert count exceeds these bands, the trigger levels need tuning down (too sensitive) or up (too loud).

## Recalibration

The signal parameters (SMA-200, 22\% vol threshold, VVIX 100) are stable but not immutable. Periodic recalibration catches parameter drift before it hurts live performance.

### Quarterly recalibration (`scripts/leveraged_quarterly.py`)

Runs on the first Monday of each quarter. Actions:

1. **Refresh price cache** to the most recent trading day.
2. **Rerun the extended-window backtest** on the recommended stack (TECL+ROM+NAIL 33/33/33 with VVIX + SGOV overlays).
3. **Compare metrics to the last-quarter and to the v4.0 tex baseline:**
   - CAGR within $\pm$ 5 pp?
   - Sharpe within $\pm$ 0.20?
   - MaxDD no worse than 1.2$\times$ v4.0?
   - Time in market within $\pm$ 10 pp of v4.0?
4. **If any metric drifts outside band:** flag for review, do NOT auto-adjust parameters. Operator inspects.
5. **Emit a quarterly report** (Markdown) to `~/trade_data/ETFTrader/reports/leveraged_recal_{YYYYQx}.md`.
6. **Send Telegram summary** with the top-line numbers.

### Annual walk-forward re-validation (`scripts/leveraged_annual_walkfwd.py`)

Runs annually on the anniversary of v4.0 publication (July 11). Actions:

1. Re-execute the walk-forward test from the v4.0 doc (3-year train, 1-year test rolling windows, threshold grid $\{0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35\}$).
2. Verify static 22\% still beats walk-forward-optimised on the tradeable tickers.
3. If a different threshold now wins by $\geq$ 0.20 Sharpe on OOS, flag for review.
4. Emit an annual report.

### Recalibration triggers (event-driven, not scheduled)

Beyond the scheduled recalibrations, trigger a review if:

- **Live drawdown exceeds 30\%** (10 pp worse than v4.0 backtest MaxDD).
- **Cumulative live vs backtest divergence** (weekly return difference) exceeds 15 pp over any rolling 12-week window.
- **VVIX distribution shifts materially** (median VVIX moves outside 80--100 for a full quarter).

## Interventions --- when to override

| Situation | Intervention | Notes |
|:-|:-|:-|
| Applet fails to run on Friday | Manual `python scripts/leveraged_weekly.py` + manual LMT+GTC | Log in weekly journal |
| IB Gateway not up on cron | Bring Gateway up, refire the daily job | Standard ops |
| TECL/ROM/NAIL LMT rejected (thin market) | Split into 3-4 smaller LMT orders, wider limit | Log the split |
| VVIX prints $>$ 100 briefly intraday, reverts | No action --- overlay uses close, not tick | Design feature |
| Live drawdown $\geq$ 25\% | Freeze via applet freeze switch | Contact author for code review |
| Suspect a code bug in signal path | Stop the sleeve; revert to cash | Higher priority than any live signal |
| Life event / job change / compliance | Manual override at operator's discretion | Log reason for future |

Default posture: **trust the strategy**. Interventions are the exception.

## Audit trail

Every trade goes into an append-only JSONL file:

```json
{"ts":"2026-08-14T20:03:12Z","phase":"manual_live","sleeve":"leveraged",
 "ticker":"TECL","side":"SELL","shares":47,"order_type":"LMT+GTC",
 "limit_price":168.20,"ib_order_id":42,"status":"Submitted",
 "signal_before":"RISK_ON","signal_after":"RISK_OFF",
 "vvix_close":83.4,"notes":"XLK below SMA-200, weekly Friday check"}
```

The audit file is append-only. Never overwrite. Keep 12 months of files locally; older archives to cloud storage.

## Consequences

### Positive

- Clean separation from smart-beta. No cross-contamination of signal logic, cash budgets, or operator mental model.
- One coherent operating surface for the leveraged sleeve --- one app, one audit, one alert stream, one recalibration cycle.
- Staged rollout (paper $\to$ manual $\to$ semi-auto) minimises live-money risk during the ramp.
- Recalibration cadence is explicit and scheduled, not "I should check that sometime."
- Alert budget is bounded (1-5/week steady state) --- operator doesn't drown in notifications.
- 10\% NAV cap + Kelly-quarter is materially more conservative than tex v4.0's 20\% + Kelly-half --- meaningful de-risking during the confidence-building phase.

### Negative

- Two Streamlit apps to run (smart-beta + leveraged). Slightly higher operational overhead vs the previous ADR-0002 integration proposal.
- Alerts are a new infrastructure dependency (Telegram bot). Requires initial setup + monitoring of the bot itself.
- Quarterly recalibration is a recurring 30-minute review commitment.

### Neutral

- Full automation is explicitly not the goal. Semi-auto with monthly manual is the endpoint. Deliberate.
- 10\% NAV cap will feel conservative once the strategy is proven. Cap can be raised in a v4.1 update after 12 months of clean live operation, with the operator's explicit sign-off.

## Sizing summary --- explicit for the reader

| Line item | Value |
|:-|:-:|
| Sizing formula | $w = \min(\text{Kelly-quarter}, 0.10 \times \text{NAV})$ |
| Recommended stack Sharpe (net of TC) | 1.32 |
| Recommended stack vol (annualised) | 22.7\% |
| Kelly-quarter (theoretical) | $\sim$155\% NAV |
| Cap that binds | 10\% NAV |
| Per-leg target within sleeve | one third of the sleeve each (TECL, ROM, NAIL) |
| Reduction: smart-beta risk-off | scale to 5\% NAV |
| Reduction: sleeve DD $>$ 25\% | freeze until DD $<$ 15\% |
| Signal cadence | weekly (Friday close) |
| Estimated transaction cost | $\sim$0.6 pp per year |

## Out of scope

- Full automation without operator confirmation. Not planned.
- Adding a fourth or fifth ticker. Top-3 is the recommendation; expanding requires a new ADR.
- Options overlays. VVIX is used as a signal, not as a tradeable instrument.
- Sharing the audit trail with the smart-beta applet. Kept separate.
- Web/mobile UI. Streamlit local only.

## References

- Strategy document: `docs/LEVERAGED_STRATEGY_DOCUMENT.tex` (v4.0)
- Smart-beta applet: `docs/ADR-0001-portfolio-applet.md`
- Cash-neutrality invariant: `~/.claude/projects/c--Users-stuar-code-ETFTrader/memory/rule_cash_neutrality_invariant.md`
- Rigour/discipline/accuracy: `~/.claude/projects/c--Users-stuar-code-ETFTrader/memory/rule_rigour_discipline_accuracy.md`
