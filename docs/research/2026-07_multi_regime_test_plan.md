# Multi-Regime Test Plan — 2026-07

**Author:** Stuart (via AI-assisted drafting)
**Date:** 2026-07-10
**Status:** Awaiting user greenlight
**Blocker cleared:** FMP price backfill for ~5000 tickers 2010→2026 (in progress at write time)

## The two questions this suite answers

**A. Are the heuristic 35 / 30 / 20 / 15 factor weights defensible?**
Given the current values were an "informed prior" from AQR literature + non-ergodicity reasoning, does multi-regime data support them, or does it point at a different vector? *Small note on regime impact only — this is a sanity check, not a re-optimisation.*

**B. Where can real risk-adjusted-return gains come from?**
Which single or combined interventions (regime overlay, correlation clustering, magnitude weighting, weight refinement) produce the largest Sharpe / Sortino uplift? **Preference: higher return, provided the drawdown remains acceptable.**

## Design constraints (do not violate)

- **Slowly-varying only** — every proposed change passes the whipsaw filter (long lookbacks, hysteresis on any regime signal, high drift thresholds).
- **No factor timing** — dynamic factor weights based on recent factor performance are rejected on principle.
- **No look-ahead** — every calculation at time $t$ uses only data at index $< t$. Existing regime and clustering modules already carry a look-ahead test.
- **Report the CAGR cost of any protective addition** — drawdown improvements that cost more than they save must be flagged even if Sharpe improves marginally.
- **Multi-window robustness** — no result is quoted from a single window. Every metric is reported for at least the full period plus a trailing 3-year and 5-year window.

## The suite (five tests, ordered by expected value-per-hour)

### Test 1 — Baseline restatement on the deeper data

Rerun the current 35/30/20/15 policy on the 2010-2026 backfill. This is the anchor for every subsequent comparison and is genuinely new — the previous baseline was on 2021-2026 only.

**Metrics reported:** CAGR, Vol, Sharpe, Sortino, MaxDD, monthly hit-rate, turnover. Two windows: full 2011-01-01 → 2026-07-09 and trailing 5-year.
**Time:** ~10 minutes runtime, once the backfill is in.

### Test 2 — Factor weight grid (question A)

Grid search over the 3-factor simplex (momentum, quality, low-vol; value included via the new real yield+ER blend):

- **Grid resolution:** 10% steps on each factor. Weights sum to 1.0. Value fixed at {0.10, 0.15, 0.20} to keep the grid tractable.
- Approximately 45 configurations.
- Full-period + 5-year + 3-year windows.
- **Robustness score:** min-Sharpe across the three windows (a config that wins one window and loses two is not a robust choice).

Deliverables:
- CSV of all configurations with metrics per window
- Heatmap plots (momentum × quality) for each value weight
- Explicit comparison: does the empirical grid winner beat the 35/30/20/15 prior on min-Sharpe? If yes by more than 0.10 Sharpe, propose a weight change with a research note. If no, the prior stands and this is the note.

### Test 3 — Regime overlay on the smart-beta portfolio (question B)

The SPY-only regime diagnostic showed −1.79% CAGR / +8.44% MaxDD trade-off. The strategy-level effect will be different (momentum already provides some implicit regime response). Test:

- Baseline: no overlay
- Overlay with default `RegimeConfig` (200d SMA, VIX<25, 10d hysteresis, 30d dwell, 0.60 multiplier)
- Sensitivity: VIX threshold ∈ {20, 25, 30}, multiplier ∈ {0.40, 0.60, 0.80}

**Verdict criterion:** overlay enabled live only if it improves Sortino on the multi-regime full window AND its worst-window CAGR loss vs baseline is ≤ 2.0%. Otherwise it stays as a defensive-only mode toggled by hand during acknowledged stress.

### Test 4 — Correlation clustering with real depth (question B)

Rerun T2.2 with multi-regime data. Same grid as before (τ ∈ {0.60, 0.70, 0.80, 0.90, 1.0}, N=30 only — the 20/30 policy question is settled). Add the sector-aware variant using the FMP `etf/info` sector map that Premium unlocks.

**Verdict criterion:** clustering enabled live only if it improves MaxDD by ≥ 2.0% AND the CAGR cost is ≤ 1.5% across at least two independent windows.

### Test 5 — Combined "best-of" (question B)

Take the winners from Tests 2, 3, 4 and run them together. Combined effects can be non-linear:
- Regime overlay + clustering: does clustering compound regime protection or duplicate it?
- Weight refinement + overlay: does a different weight vector amplify or blunt the overlay?

Report the "kitchen sink" combination against the baseline restated in Test 1.

### Test 6 (optional — T2.3) — Score magnitude as expected return

Currently the optimiser uses rank order; the magnitude of the integrated score is discarded. Test replacing exponential rank weighting with score-magnitude-proportional weighting (with the same 2% / 15% bounds). Simple A/B.

Report only whether it changes the answer materially — if <10 bps of CAGR movement, note and move on.

## Non-goals — what this suite does NOT do

- **Not a live-trading validation.** Backtest results ≠ live results. Even after this suite, the "put trades on" decision remains an operator judgement, not an automated one.
- **Not an alpha search across new factors.** No search for exotic factors (skewness, kurtosis, term-spread). Adding factors is a design-principle change and requires its own note.
- **Not a rebalance-frequency sensitivity.** The bimonthly cadence is a policy input, tested separately if T3.3 comes up in the queue.
- **Not a survivorship-corrected backtest.** T1.2 accepted as small; not fixing.

## Execution plan

Once the FMP backfill notification lands:

1. Recompute rolling factor scores on the extended universe (`scripts/factor_weight_diagnostic.py` refresh cache).
2. Run Test 1 (baseline restatement) — 10 min.
3. Run Test 2 (weight grid) — ~2–3 hours.
4. Run Test 3 (regime overlay) — ~1 hour.
5. Run Test 4 (clustering) — ~30 min.
6. Run Test 5 (combined) — ~30 min.
7. Optional Test 6 (magnitude) — 30 min.
8. Publish `docs/research/2026-07_multi_regime_results.md` consolidating all findings, with explicit recommendations against the two questions (A/B) and per-verdict-criterion decisions.

Total: an evening of runtime + a morning of interpretation and doc write-up.

## Reporting shape

Each test result carries:

- The specific config tested
- Metrics on at least two windows
- Comparison delta vs baseline
- A one-line verdict against the verdict criterion (accept / reject / borderline)

Consolidated report ends with:

- **Recommended factor weights** (unchanged 35/30/20/15 unless a robust winner is found)
- **Regime overlay decision** (enable / defer / decline)
- **Clustering decision** (enable / defer / decline)
- **Magnitude weighting decision** (adopt / decline)
- **Concrete config change to `configs/etf_smart_beta.toml`** if any
- **Sanity note:** would the live 28-position book have performed materially differently under the recommended changes over the last 12 months? If yes, is that a good thing or a bad thing?

## Decision gate before live enablement

None of the above interventions get enabled on the live book until:

1. Multi-regime backtest verdict is positive per its criterion
2. Change is committed to `configs/etf_smart_beta.toml` with a dated research-note link
3. Operator explicitly approves via the applet's BIG switch on the next rebalance
