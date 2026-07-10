# Correlation-Clustering Experiment — Design Note (T2.2)

**Date:** 2026-07-10
**Author:** Stuart (via AI-assisted drafting)
**Status:** Design; awaiting data catch-up before first run

## The question the user asked

> "I definitely see the *same* ETFs in my portfolio — but does this come at a
> cost to the returns? We will need to do this experiment."

Restated: the current top-N-by-integrated-score selection is
correlation-blind. It routinely picks several tech ETFs (QQQ, XLK, VGT, SMH,
IYW …) that move together. Diversification is nominal, not real. The proposal
is to add a decorrelation filter on top of the ranking step. The question is:
**does the decorrelation help, hurt, or wash out?**

## The honest expectation before running anything

Not neutral, either way:

- **Return cost is likely in a single-regime backtest.** If the backtest period
  is a bull run led by the correlated cluster (e.g. 2021–2026, tech-led), the
  clustering constraint forces us out of the winning cluster in favour of
  lower-scored uncorrelated ETFs. In-sample Sharpe falls.
- **Drawdown protection is likely in a multi-regime backtest.** In a regime
  reversal — 2000 dotcom, 2008, 2022 tech unwind — a correlation-blind
  portfolio takes the full hit. A clustered portfolio takes less. Max drawdown
  and Sortino improve.
- **The user's non-ergodicity argument favours the second effect.** Time-average
  growth compounds negatively on drawdowns; the arithmetic-mean gain from the
  bull-market cluster is a mirage in log-space.

So: if the experiment only runs on 2021–2026, we will conclude clustering
hurts, and that conclusion will be wrong. **The experiment is only meaningful
once we have the 2010–2020 backfill** (T2, backlog item).

## Design

### Data plane

- `prices` daily frame (2021–2026 available now, 2010–2026 target once T2
  backfill lands).
- `rolling_factor_scores.parquet` (already cached — 12 rebalances/yr × 5y for
  the current window).
- **New:** 126-day rolling correlation matrix, evaluated at each rebalance date
  on the return series of eligible tickers. 126 days ≈ 6 months, matching the
  slowly-varying doctrine (short windows overfit noise, long windows lag
  regime shifts).

### Clustering step

At each rebalance date, given the ranked candidate list `C` (fractional rank
scores), pick positions greedily:

```
picked = []
for candidate in C sorted by score descending:
    if any(|corr(candidate, p)| > tau for p in picked):
        continue  # skip — too correlated with something we already picked
    picked.append(candidate)
    if len(picked) == num_positions:
        break
```

Correlations use the trailing 126-day return series ending at `rebalance_date`
(no look-ahead). Correlation matrix is refreshed at **quarterly** cadence, not
per rebalance — the slowly-varying doctrine again. Between refreshes the
matrix is held constant.

### Parameter sweep

| Parameter          | Values                     | Rationale |
|--------------------|----------------------------|-----------|
| `tau` (corr cap)   | 0.60, 0.70, 0.80, 0.90     | 0.90 ≈ near-identity; 0.60 forces aggressive spread |
| `num_positions`    | 20, 30                     | Current pre-change and post-change baskets |
| `corr_window`      | 126d                       | Fixed; sensitivity check later if warranted |
| `refresh_cadence`  | quarterly                  | Fixed; per slowly-varying doctrine |

Baseline: `tau = 1.0` (no clustering) = the current behaviour, for direct
comparison.

### Metrics reported

The same set as `factor_weight_diagnostic.py::StrategyMetrics`:

- CAGR
- Realised annualised vol
- Sharpe, Sortino
- **Max drawdown** (this is the metric that should benefit from clustering)
- Monthly hit rate
- Turnover — clustering may raise turnover as the eligible set churns; needs monitoring
- **New: realised portfolio correlation summary** — average pairwise correlation of
  the picked basket, quarter by quarter. This is the diagnostic that says
  "yes, clustering actually did what it claimed."

### Reporting shape

Mirror `2026-07_factor_weight_diagnostic.md`: one markdown report with
tables + PNG plots of CAGR-vs-MDD frontier across `(tau, num_positions,
window)`, plus the realised-correlation trace over time.

## What this will *not* do

- Not a regime-conditional experiment. That is T2.1. Combining regime overlay
  and clustering is a strictly separate exercise once both work alone.
- Not a factor-weight re-optimisation. Factor weights stay at 35/30/20/15 —
  the non-ergodicity argument holds and clustering is orthogonal.
- Not a cluster-count target. The greedy-drop above is the simplest defensible
  choice; k-means / hierarchical clustering with a fixed k adds hyperparameters
  and whipsaw risk without obvious upside.

## Blocker before first run

Two-part gate:

1. **2010–2020 backfill in the cache.** Without it the experiment answers a
   question about a single tech-led bull run, which is not the question the
   user asked.
2. **Value factor real (T1.1).** Not strictly required — but if we're
   evaluating clustering under a fake-value baseline, the winners will be
   momentum-and-quality-heavy tech names, biasing the "no clustering wins"
   conclusion. Real value pulls in banks, energy, defensives — a natively more
   diversified cohort where clustering may show a smaller marginal benefit.

## Implementation size estimate

- New file: `scripts/correlation_diagnostic.py`, ≈ 250 lines, imports
  `RollingScores`, `_integrated_scores`, and metric helpers from
  `factor_weight_diagnostic.py`. Adds `apply_correlation_filter(candidates,
  scores, corr_matrix, tau)` and a `simulate_strategy_clustered` wrapper.
- Estimated runtime: 4–5 minutes per `(tau, num_positions)` combo × 8 combos
  ≈ 35 minutes. Cheaper than the factor-weight diagnostic because we reuse
  the cached rolling scores.

## Decision gate

Do not build the script until the 2010–2020 backfill is in the cache.
Otherwise we will run it once, get an in-sample-favouring result, and either
draw the wrong conclusion or waste the effort a second time.
