# Research backlog

Standing list of research questions that need periodic investigation to keep
the ETF smart-beta strategy evidence-based rather than intuition-based.
Priority order below is a suggestion, not a lock — the current top item is
the calibration revisit.

## Governing rule

**No magical thinking on the knobs.** Every numeric parameter in
[configs/etf_smart_beta.toml](../configs/etf_smart_beta.toml) — factor
weights, position count, weight bounds, drift threshold, stop distances,
lookback windows — must ultimately trace to either published research or a
backtest recorded in this repo. If a number's provenance is *"seemed
reasonable"*, it belongs on this backlog.

Target cadence: **semi-annual** re-run of the top item, with results
committed as a dated note under `docs/research/YYYY-MM_<topic>.md`.

---

## 1. Factor-weight calibration + regime awareness  (P0, next semi-annual slot)

**Why:** Current weights (Mom 35% / Qual 30% / Vol 20% / Val 15%) are
academic-plus-intuition. The tech document says "optimized weights" but no
walk-forward or cross-validation record exists in the repo. The current
blend is ~65% pro-cyclical (Mom + Qual), ~20% counter-cyclical (Vol), ~0%
real value protection (Val is proxied by expense ratio, not P/E or P/B) —
so it is potentially fragile in a valuation-driven downturn.

**Belief the exercise is testing:** smarter regime-aware calibration can
meaningfully lift the strategy's Sharpe. Owner: user.

**Level 1 — Diagnostic (~1 evening):**
- Backtest a grid of factor-weight vectors over the last 5 years.
- Bucket the period by regime (options: VIX quintile / SPY vs 200dma /
  valuation percentile of the universe).
- For each (weight vector, regime bucket) produce: annualised return, vol,
  Sharpe, max drawdown, turnover.
- Deliverable: a table + short note. Answers *"was our chosen blend
  actually best, and was it best in downturns specifically?"*
- Output location: `docs/research/YYYY-MM_factor_weight_diagnostic.md`.

**Level 2 — Real value factor (~1-2 evenings):**
- Replace `Value = -expense_ratio` with a composite (P/E, P/B, dividend
  yield — pulled via `yfinance` info dict for ETFs).
- Re-run level 1.
- Output: same location, section 2.

**Level 3 — Regime-conditional weights (bigger, ADR-0002 candidate):**
- Detect regime at each rebalance (VIX bucket, breadth, etc.).
- Switch to a regime-specific weight vector chosen ahead of time from
  level 1.
- Requires a design decision on regime-switching stability (hysteresis,
  minimum dwell time) to avoid whipsaw.

## 2. Config drift between tech doc, code, and policy TOML  (P1)

The [technical investment document](TECHNICAL_INVESTMENT_DOCUMENT.tex)
describes v4.2 with 30 positions, 3–8% weight bounds, quarterly rebalance,
robust MVO. The current
[policy TOML](../configs/etf_smart_beta.toml) has 20 positions, 2–15%
bounds, bimonthly rebalance. The published backtest numbers (12.1% CAGR,
0.64 Sharpe) belong to the tech-doc config, not the policy-TOML config.

Before the applet goes live it must run *one canonical config*. Options:
- (a) adopt tech-doc numbers (30 / 3–8% / quarterly), regenerate any tests
  that pin the old values;
- (b) keep 20 / 2–15% / bimonthly and rerun the backtest, updating the tech
  doc with the fresh numbers.

Decision blocked on level-1 diagnostic (above) — that exercise will
naturally surface which config performs better.

## 3. Cost model realism  (P2)

The transaction-cost model in
[src/backtesting/costs.py](../src/backtesting/costs.py) uses fixed bps
assumptions for spread + slippage. Real IB fills on our universe
(especially thin international ETFs) may be materially different. Worth
sampling: pull the last N fills from IB and compute realised vs modelled
cost, and update the model if the gap is large enough to affect optimiser
choices.

## 4. Stop-loss parameterisation  (P2)

12% entry stop and 10% trailing stop are round numbers — CLAUDE.md flags
this as the class of magic constant to avoid. Rerun the backtest across a
grid of (entry_stop, trailing_stop) pairs and see whether the current
values are on the Sharpe / drawdown frontier or well off it.

## 5. Rebalance frequency vs drift-threshold trade-off  (P3)

Bimonthly + 5% drift produces ~6 rebalances/year; quarterly + 5% produces
~0.6 (per tech doc). Turnover cost sensitivity — verify the "fewer
rebalances is better" claim across a realistic cost model.

---

## Completed research

_(empty — populate as items ship)_
