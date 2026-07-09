# ADR-0001: ETF smart-beta portfolio applet — deterministic core + LLM explainer

**Status:** Proposed
**Date:** 2026-07-09
**Deciders:** Stuart (owner)

> **Scope: ETF smart-beta strategy only.** The FX and leveraged strategies are explicitly
> **deferred** to future ADRs (see "Out of scope"). The goal is one super-smooth, automated
> ETF smart-beta operator first; the other two strategies are developed later, on their own.

## Context

ETFTrader is a working, tested (green), published quant stack — **not yet run live**. Today it
is operated by running twelve notebook-scripts in sequence (`s1_universe … s12_leveraged_execute`).
That workflow is fragile, easy to run out of order, and unpleasant to drive. The reusable logic
already lives in the `src/` package: data collection, factor scoring, a cvxpy optimizer, a
threshold rebalancer, a risk manager, a backtest engine with a cost model, and `ib_insync`
execution.

The goal is to operate the **ETF smart-beta** portfolio through **one simple interface**: pull the
live account, enter an optional **additional-cash budget**, run collection → factor scoring →
optimize → propose trades, read a plain-English explanation of what the trades do, and — only by a
deliberate action — apply them. Only the ETF smart-beta path is in scope; the FX (`s8…s10`,
`src/fx_options`, `src/fx_spot`, `signals/`) and leveraged (`s11`/`s12`, `src/leveraged`) surfaces
are left untouched for now.

**Constraints shaping this decision:**
- **Live IBKR account, no paper** — safety is paramount. Default posture is advisory / draft-only;
  IB Gateway runs with **Read-Only API on**.
- **Single user, local, Windows + miniconda** — no multi-tenant/web-scale requirements.
- **Reuse `src/`** — the applet must be a thin layer over audited logic, adding no new trading maths.
- **Redirectable** — the target may evolve; the design should not lock us in.

## Decision

Build a **local Streamlit applet** (`app.py`, run with `streamlit run`) that orchestrates the
existing **ETF smart-beta** `src/` pipeline behind a linear UI, with an **embedded LLM
explanation/chat layer**, and a **guarded apply switch** for execution. Retire the ETF-path
notebook ritual (`s1…s7`) as the operating surface; the FX/leveraged notebooks (`s8…s12`) are left
as-is for their own later projects.

**Core architectural principle — deterministic decides, the LLM explains.** All numbers (selections,
weights, sizes, limits, order payloads) come from the deterministic `src/` code — cvxpy optimizer +
`risk_manager` + cost model. The LLM only *narrates* the structured output and answers questions.
The LLM cannot move a weight or place an order. This keeps trust with auditable code and satisfies
the `CLAUDE.md` "no magic numbers / fail-loud execution" rules.

## Options Considered

### Option A: Local Streamlit applet + deterministic core + LLM explainer (recommended)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low–Med — Streamlit already in `environment.yml`; wraps existing `src/` |
| Cost | Low — local; LLM calls only for narration/Q&A |
| Scalability | Sufficient — single user, single machine |
| Team familiarity | High — Python/pandas/Streamlit; your own codebase |

**Pros:** simple one-command launch; reuses audited logic; clear human-in-the-loop; safe by default;
explanation makes the "why" legible; easy to iterate.
**Cons:** Streamlit reruns model needs care for long-running steps (cache/session state); LLM layer
adds an API dependency for the narration (not the decisions).

### Option B: Keep the s1…s12 notebooks
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low to keep, high to operate |
| Cost | Low |
| Scalability | Poor operability |
| Team familiarity | High |

**Pros:** zero new work; full transparency of each step.
**Cons:** the exact pain point being removed — twelve manual steps, order-dependent, error-prone,
no guarded apply, no synthesized explanation.

### Option C: Full web service (FastAPI + React, scheduler, DB)
| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost | Higher (hosting, auth, ops) |
| Scalability | Overkill for one user |
| Team familiarity | Medium |

**Pros:** most flexible; remote access; robust scheduling.
**Cons:** large build and maintenance burden; more attack surface for a live-trading app; unjustified
for a single local operator.

### Sub-decision: LLM authority
| Model | Verdict |
|-------|---------|
| LLM **explains only**, deterministic core decides | ✅ Chosen — auditable, safe |
| LLM **proposes sizes**, human approves | ✗ Rejected — non-deterministic sizing in a live-money path |
| No LLM | ✗ Rejected — loses the "tell me what I'm getting" value |

## Trade-off Analysis

The central trade-off is **operability vs. build cost vs. safety**. Option B is free but fails the
operability goal. Option C maximizes flexibility at a cost and risk profile that a single local
operator cannot justify — and a bigger surface is worse for a live-trading tool. Option A hits the
sweet spot: minimal new code (glue, not maths), a genuinely simple UI, and — crucially — it keeps the
**decision logic deterministic and auditable** while still delivering the agentic explanation. The
LLM-authority sub-decision is the safety linchpin: narration is additive and reversible; letting a
model size trades in a live account is not.

## Consequences

**Easier:** operating the portfolio (one screen, one run button); understanding each proposal (plain
English + interactive Q&A); deploying new cash (budget → cash-deployment mode: mostly buys, low
turnover, no tax churn); enforcing policy (limits live in the optimizer/risk-manager, surfaced in UI).

**Harder / to watch:** Streamlit's rerun model requires caching long steps and holding pipeline state
in `st.session_state`; the applet is only as trustworthy as the `src/` internals — **so the Claude
Code review/refactor is a prerequisite**; the LLM layer needs guardrails so its text can never be
mistaken for an instruction to the broker.

**To revisit:** tax-lot / wash-sale handling depth; whether to add scheduling (a weekly auto-run that
still stops at *proposed*, never *applied*); multi-account support if ever needed.

## Out of scope (deferred to future ADRs)

Explicitly **not** part of this applet, to keep the ETF smart-beta path super-smooth first:

- **FX strategy** — `src/fx_options`, `src/fx_spot`, `signals/`, notebook steps `s8…s10`,
  the Databento FX collectors. Harder to model and validate; its own ADR later.
- **Leveraged strategy** — `src/leveraged`, notebook steps `s11`/`s12`, decay/rebalance
  assumptions. Highest-risk execution; deliberately deferred.

These modules stay in the repo untouched. The Claude Code refactor and audit should therefore
**skip the FX/leveraged code paths** — which also means the risky `s11`/`s12` execution scripts
are out of the refactor blast radius entirely.

## Applet flow (ETF smart-beta path only)

1. **Setup** — live IBKR read-only snapshot (NAV, cash, positions) + policy targets; input: *additional budget*.
2. **Run** — `s2` collect (ETF prices) → `s3` factors (value/quality/momentum/low-vol) → `s4` optimize (cash-deployment mode when budget > 0) → `s6` proposed trades.
3. **Proposed trades** — blotter (side/qty/est-cost/weight vs target), turnover, est. commission/slippage (cost model), factor exposures before→after.
4. **Explain (agentic)** — Claude narrates the structured proposal and answers "why sell X?" / "what if budget = Y?".
5. **Apply (guarded)** — default OFF; two-key action (Read-Only API off **and** explicit confirm) → shows exact payloads → transmits via `ib_insync` → writes audit log to `trade_data`.

## Action Items

1. [ ] **Prereq:** Claude Code review + Option-A refactor scoped to the **ETF-path modules only**
   (`data_collection` for ETFs, `factors`, `portfolio`, `backtesting`, `utils`) — skip FX/leveraged;
   `pytest` green in `etftrader`.
2. [ ] Lock the ETF smart-beta **policy file** (factor targets, bands, cash buffer, tax status).
3. [ ] Build `app.py` steps 1–3 (setup + run + proposed trades) — **read-only, no apply.**
4. [ ] Add the LLM explanation/chat layer over the structured proposal (step 4).
5. [ ] Add the guarded **Apply** switch with two-key safety + audit log (step 5) — last.
6. [ ] Retire `s1…s7` (ETF path) as the operating surface; FX/leveraged (`s8…s12`) stay untouched
   under `notebooks/` for their own later projects.
