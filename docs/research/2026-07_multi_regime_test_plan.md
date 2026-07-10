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

### Test 2 — Factor weight defence (question A)

**Reframed 2026-07-10 per operator directive: this is a validation of the 35/30/20/15 prior, not a search for a replacement.** The goal is to demonstrate that the prior is defensible, understand where and why it might be sub-optimal, and produce a written defence that a serious reviewer would accept.

**Structure (four sub-tests, each answers a specific critique of the prior):**

**2a. Local stability around the prior.** Vary each factor's weight by $\pm 5$\% and $\pm 10$\% relative, holding the others proportional. Question answered: "if the prior is a knife-edge, small perturbations should collapse the performance." A defensible prior shows a broad, gentle plateau — small perturbations produce small metric changes across all three windows.

**2b. Concentration stress-test.** Test the three single-factor concentrations (momentum 70\% / quality 70\% / low-vol 70\%, others residual). Question answered: "am I under-weighting the strongest single factor?" A defensible multi-factor prior beats each concentration on drawdown-adjusted return, even if a concentration wins raw CAGR in one window.

**2c. Empirical-winner-vs-prior contrast.** Identify the in-sample-optimal weight vector on each of the three windows independently. Question answered: "would tuning to any one window meaningfully help?" A defensible prior sits near the average of the three winners AND survives the min-Sharpe robustness test against them.

**2d. Non-ergodicity check.** Compute the time-average log-return (geometric mean) rather than the arithmetic-mean CAGR for each of {prior, single-factor concentrations, in-sample winners}. Question answered: "does the ergodicity-based defence of the prior actually hold?" A defensible prior wins the time-average metric more often than the arithmetic-mean metric.

**Deliverables:**
- Table showing the prior's metrics on 3 windows next to nearby-variant metrics
- Plot showing the local performance plateau around 35/30/20/15
- A written defence section in the results doc explaining, in prose, WHY these weights are appropriate given: (i) AQR canonical evidence, (ii) non-ergodicity of log-space compounding, (iii) insurance-premium interpretation of quality + low-vol, (iv) slowly-varying design principle.
- Explicit statement: "the prior is retained" is the default outcome. A weight change is proposed only if a robust alternative beats the prior on min-Sharpe by $\geq 0.10$ across all three windows AND survives the time-average check.

**Runtime with multiprocessing:** 45 configs × 3 windows over the cached rolling scores $\approx$ 135 evaluations. With 4-worker pool: 5--8 minutes. Sequential would be 20--30 minutes.

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

## Execution plan (parallelised)

Once the FMP backfill notification lands:

**Phase 1 — sequential prerequisite (~5 min)**

Recompute rolling factor scores on the extended universe once. This is the shared input to every subsequent test and cannot be parallelised without duplicating work. Cached to `~/trade_data/ETFTrader/processed/rolling_factor_scores.parquet` under a fresh source-hash so downstream tests reuse it.

**Phase 2 — parallel test bank (~15 min wall clock)**

Tests 2, 3, 4, 6 all consume the same cached rolling scores and produce independent outputs. Run each as its own process, each using a `multiprocessing.Pool` internally to parallelise its own grid over the local CPU pool. Split as follows on a typical 8-core machine:

| Test | Own process | Internal workers | Est. wall-clock |
|---|---|---|---|
| Test 1 (baseline) | 1 | 1 (single config) | 30 sec |
| Test 2 (defence grid) | 1 | 4 workers × 45 configs × 3 windows | 5-8 min |
| Test 3 (regime sensitivity) | 1 | 3 workers × 9 configs | 3-4 min |
| Test 4 (clustering) | 1 | 3 workers × 5 configs × 2 variants | 3-4 min |
| Test 6 (magnitude, if run) | 1 | 1 (single A/B) | 1-2 min |

Phase 2 uses ~11 concurrent workers total; on an 8-core box this is fine because the workload is CPU-bound but each config completes in seconds — over-subscription is not a bottleneck.

**Phase 3 — sequential synthesis (~3 min)**

Test 5 (combined best-of) depends on outputs from tests 2/3/4. Run after Phase 2 completes.

**Phase 4 — write-up (~30-60 min)**

Publish `docs/research/2026-07_multi_regime_results.md` — the polished deliverable. Structure documented under "Reporting shape" below. Written as a serious research note, not a data dump: headline stats prominent, prose explains WHY each result matters, explicit constructive critique of the prior in Test 2's section.

**Total wall-clock:** ~25-30 minutes of compute + write-up time.

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
