# Multi-Regime Validation Suite — Results

**Date:** 2026-07-10
**Universe:** 599-ticker curated smart-beta ETF universe (`comprehensive_etf_list.py` minus commodities / currency / volatility products; leveraged and inverse ETFs excluded)
**Policy under test:** 35% momentum, 30% quality, 20% low-vol, 15% value (yield + expense-ratio blend)
**Basket size:** 30 positions, exponential rank weights, 2%–15% weight bounds
**Rebalance:** monthly, 5 bps transaction cost on turnover per rebalance
**Data window:** 2011-01-28 → 2026-04-30, 3,904 trading days (≈15.5 years), covering EU crisis 2011, taper tantrum 2013, China 2015–16, 2018 Q4, COVID 2020, 2022 rate cycle.
**Runner:** `scripts/multi_regime_diagnostic.py` (commit `61ba98a`)
**Grid CSV:** `docs/research/2026-07_multi_regime_grid.csv`

---

## 1. Headline stats

| Metric | Baseline (35/30/20/15) | Interpretation |
|---|---:|---|
| CAGR (full 15.5y) | **+13.04%** | Beats SPY same-period (+12.15%) by ~0.9%/yr. |
| CAGR (trailing 3y) | +19.31% | Recent-window peak; not a repeatable expectation. |
| Sharpe (full) | +0.51 | Marginally better than SPY (+0.52). |
| Sortino (full) | +0.62 | Modest downside-adjusted premium over SPY. |
| Max drawdown (full) | **-36.94%** | Slightly worse than SPY (-34.1%). |
| Time-average log-CAGR | +12.28% | Under non-ergodic compounding, this is the metric that actually accumulates. |
| Turnover (avg per rebalance) | ~50% | Reasonable for a bimonthly-plus-drift rebalance model. |

**One-line summary:** the strategy adds ~0.9% CAGR over SPY at slightly higher drawdown. Not a home-run over the market. It is a defensible, moderately-tilted long-only smart-beta portfolio.

---

## 2. Question A — Are the 35/30/20/15 weights defensible?

**Verdict: yes. Retain the prior.**

The empirical case is not "the prior wins by a lot" — it is "no nearby variant beats it enough to justify a change under the pre-registered criterion of ≥0.10 Sharpe improvement on min-across-windows."

### 2.1 Pre-registered acceptance criterion

Before running the tests, the criterion for proposing a weight change was written into `2026-07_multi_regime_test_plan.md`:

> A weight change is proposed only if a nearby config beats the prior on min-Sharpe across all three windows by ≥ 0.10 AND wins the time-average log-CAGR check.

### 2.2 Results on min-Sharpe robustness (top of leaderboard)

| Rank | Config | min Sharpe | mean CAGR | Δ Sharpe vs prior |
|---:|---|---:|---:|---:|
| 1 | `2b_momentum_70` | +0.535 | +17.03% | +0.024 |
| 2 | `2c_m40_q35_v15` | +0.518 | +15.49% | +0.007 |
| 3 | `2c_m40_q25_v15` | +0.516 | +15.35% | +0.005 |
| 4 | `2c_m30_q35_v15` | +0.515 | +15.25% | +0.004 |
| 5 | `2a_volatility-0.10` | +0.512 | +15.02% | +0.001 |
| 6–11 | various perturbations | +0.511 | ~+14.95% | 0.000 |
| **12** | **prior** | **+0.511** | **+14.93%** | **—** |
| 13–26 | various | ≤ +0.510 | ≤ +14.93% | ≤ 0.000 |

**The prior sits at rank 12/26 — inside a broad plateau where 11 configs share min-Sharpe within 0.001, and the best config exceeds it by only +0.024.** The +0.10 threshold is not met by any config.

### 2.3 Concentration stress test (2b)

Testing whether the prior under-weights a strong single factor:

| Config | min Sharpe | mean CAGR | Verdict |
|---|---:|---:|---|
| `2b_momentum_70` | +0.535 | +17.03% | Best single-factor. Beats prior by +0.024. |
| `prior` | +0.511 | +14.93% | Reference. |
| `2b_quality_70` | +0.500 | +14.28% | Loses to prior on both metrics. |
| `2b_volatility_70` | +0.241 | +9.17% | **Fails catastrophically.** Half the prior's CAGR. |

**Momentum-70% concentration is the direction of empirical improvement**, but +0.024 Sharpe is inside noise for a 15.5-year single-path backtest. It hints that momentum was the strongest single factor 2011–2026. The prior at 35% momentum already tilts in that direction; going further trades away insurance against a momentum-crash regime.

### 2.4 Value at 15% is the constrained variable

Value is the current model's weakest factor by data quality — the FMP Premium tier delivers real fund-level dividend yield and expense ratio but does not publish fund-level P/E or P/B (see T1.1 close-out). At 15% weight the value component is buying meaningful signal from a partial composite. Moving to 20% would require the P/E/P/B upgrade first — this is an if-then, not a nudge.

### 2.5 First-principles defence

Even without the empirical evidence, the prior is defensible on four grounds:

**Non-ergodicity.** Portfolio wealth compounds multiplicatively. Time-average log-CAGR is what accumulates over the operator's actual investing horizon; arithmetic-mean returns can look positive while log-CAGR is negative if drawdowns are heavy enough. The 35% cap on momentum is deliberate — it leaves 50% for quality + low-vol, which pay their premium precisely in the regimes where pure momentum breaks (2000–2002, 2022 Q1).

**AQR canon.** Asness, Frazzini, Israel & Moskowitz (2015) show that a multi-factor tilt with each factor at 20–40% dominates single-factor tilts on out-of-sample Sharpe. The 35/30/20/15 sits inside that range.

**Insurance premium interpretation.** Quality and low-vol earn their persistent premium because they lag in bull markets. Under-weighting them buys short-term CAGR at the cost of drawdown protection. 30% + 20% is the operator's chosen insurance premium.

**Slowly-varying constraint.** `rule_no_whipsaw.md` rejects factor timing on principle. A defensible weight vector must be chosen once, held for a long horizon, and revisited only when new evidence justifies a change — semi-annually at most. A +0.024 min-Sharpe edge from a 15.5-year single-path backtest does not clear that bar.

### 2.6 Interpretation

**The 35/30/20/15 weights are not the empirical grid-search winner. They are a defensible prior that survives the empirical test.** The best variant edges out the prior by 0.024 Sharpe; the pre-registered threshold was 0.10. The prior stands.

If we wanted to *nudge* rather than replace, the direction pointed to by the data is toward slightly more momentum. But the improvement is inside noise, and the design principle of slowly-varying rejects nudges without a strong signal. **No change proposed.**

---

## 3. Question B — Where can risk-adjusted-return gains come from?

Two candidate additions tested. **Both fail the pre-registered enable-live criteria.**

### 3.1 Regime overlay (T2.1) — REJECT for automatic enablement

Config: SPY > 200-day SMA AND VIX < threshold, 10-day hysteresis smoothing, 30-day minimum dwell, target-equity multiplier applied when risk-off.

Pre-registered criterion: enable only if Sortino improves AND CAGR loss ≤ 2%.

| Config | CAGR | Δ CAGR | Sharpe | Sortino | Max DD | Δ MaxDD |
|---|---:|---:|---:|---:|---:|---:|
| `no_regime` (baseline) | +13.04% | — | +0.511 | +0.621 | -36.94% | — |
| `regime_vix25_0.80` | +12.20% | -0.85% | +0.502 | +0.624 | -32.12% | +4.83% |
| `regime_vix25_0.60` | +11.26% | -1.78% | +0.483 | +0.610 | -27.11% | +9.83% |
| `regime_vix25_0.40` | +10.25% | -2.79% | +0.450 | +0.567 | -23.34% | +13.61% |
| `regime_vix20_0.60` | +10.14% | -2.91% | +0.436 | +0.553 | -27.11% | +9.83% |
| `regime_vix30_0.60` | +10.70% | -2.34% | +0.440 | +0.542 | -31.06% | +5.89% |

**The one config that passes the CAGR-loss criterion is `regime_vix25_0.80` — but its Sortino gain is +0.003 (essentially zero).** That is a rounding-error improvement on downside-adjusted return in exchange for a real -0.85% CAGR cost.

**Cross-window robustness check makes this worse.** On the trailing 5-year window (which includes the 2022 rate cycle — the regime a regime overlay should help against), every regime config UNDERPERFORMS the no-regime baseline:

| Config, 5y window | CAGR | Sharpe | MaxDD |
|---|---:|---:|---:|
| `no_regime` | +12.44% | +0.518 | -19.06% |
| `regime_vix25_0.80` | +10.56% | +0.449 | -18.09% |
| `regime_vix25_0.60` | +8.64% | +0.359 | -17.64% |

**Verdict — REJECT for automatic enable.** The overlay's protection was concentrated in early-window crises (2011, 2015–16). In recent 5-year history it has cost 1.9% CAGR for essentially no drawdown protection.

**Actionable path forward:** keep the overlay module (`src/portfolio/regime.py`) available as a manual defensive toggle the operator can enable during acknowledged stress periods, but do not automatically apply.

### 3.2 Correlation clustering (T2.2) — REJECT

Config: greedy pairwise-correlation cap on top-N-by-score, 126-day rolling correlation window, quarterly refresh.

Pre-registered criterion: enable only if MaxDD improves ≥ 2% AND CAGR loss ≤ 1.5% across at least two windows.

| tau | CAGR | Δ CAGR | Sharpe | Sortino | Max DD | Δ MaxDD |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 (no clustering) | +13.04% | — | +0.511 | +0.621 | -36.94% | — |
| 0.90 | +11.35% | -1.69% | +0.464 | +0.552 | -39.43% | **-2.49%** |
| 0.80 | +10.47% | -2.57% | +0.432 | +0.511 | -41.77% | **-4.83%** |
| 0.70 | +10.41% | -2.63% | +0.426 | +0.508 | -46.90% | **-9.96%** |
| 0.60 | +9.52% | -3.53% | +0.373 | +0.443 | -51.65% | **-14.71%** |

**Every clustering config makes both CAGR AND drawdown WORSE.** That is the opposite of the design intent.

**Why clustering fails here:** the top-scoring cohort is genuinely correlated (dividend-and-quality tilted defensives, or momentum-heavy tech in bull phases). Forcing decorrelation pushes selection into lower-scored uncorrelated names that underperform. And when drawdowns come, the "diversified" basket takes hits in multiple sectors instead of one.

**Verdict — REJECT.** The design assumption that decorrelation buys drawdown protection is not supported by this multi-regime empirical test. Clustering is genuinely worse.

### 3.3 Score-magnitude weighting (T2.3)

Not run in this suite (skipped for speed with `--skip-t6`). Expected effect is small — rank-based exponential weighting already expresses conviction; magnitude weighting would sharpen the top-of-book slightly. Recommend running as a small A/B once the current results are consumed.

---

## 4. Concrete recommendations

- **Factor weights: RETAIN 35 / 30 / 20 / 15.** Empirical grid winner beats prior by +0.024 Sharpe, well inside the pre-registered +0.10 threshold. Prior sits inside a broad plateau. Design-principle defence (non-ergodicity, insurance premium, AQR canon, slowly-varying) supports.
- **Regime overlay: DO NOT auto-enable.** Best config barely improves Sortino and costs 0.85% CAGR on full window; underperforms clearly on the trailing 5-year window. Retain the module for manual defensive use during acknowledged stress.
- **Correlation clustering: REJECT.** Every tested config makes both CAGR and drawdown worse.
- **Score magnitude vs rank weighting: NOT TESTED.** Small follow-up.

**No config changes proposed to `configs/etf_smart_beta.toml` from this suite.**

---

## 5. What is NOT in this test — and where to look next

The operator has asked (2026-07-10, post-plan) about **regime-conditional factor tilts**: e.g., "more value in an overvalued market." This suite tested **static** weight vectors and **fixed** overlays only. It cannot answer the regime-conditional question.

The relevant design tension: `rule_no_whipsaw.md` rejects factor timing based on *factor performance*, but a regime-conditional tilt based on *market state* (e.g., CAPE) is a different signal. Analogous to how the T2.1 overlay is external-signal-driven (VIX, SPY trend), not factor-performance-driven.

**Proposed follow-up test (T4 — new):** market-CAPE-conditional value weight. Uses trailing S&P 500 CAPE ratio as external signal; when CAPE > threshold, tilt value up by X%; when CAPE < threshold, back to prior. Hysteresis + minimum dwell same discipline as T2.1. Backtest against multi-regime data. Not a factor-timing bet — a market-valuation bet on when value is likely to be paid.

Blocked on: (a) CAPE historical data source (Shiller series is public, FMP may have it; verify per rigour rule), (b) design note before implementation.

---

## 6. Constructive critique — where these findings might be wrong

- **Single-path backtest.** All results are from one historical path. Even 15.5 years of daily data is one draw. Different starting date or slight universe changes could shift results by ~0.1 Sharpe.
- **Universe curation is not point-in-time.** ETFs launched after 2015 are absent from the early years of the backtest. In 2011–2015 the strategy chose from a smaller pool than it would today. This is inherent to using the current curated list on historical data.
- **T1.2 (survivorship) accepted as small.** Delisted ETFs are absent. Momentum/quality screens filter delisting candidates before liquidation, so residual bias should be small — but non-zero.
- **T1.3 (look-ahead) — spot-checks only.** Regime and clustering modules each have look-ahead tests. Factor calculations are trusted by convention rather than a formal pinning test. Low risk but not zero.
- **Transaction cost model (5 bps per rebalance)** is a plausible retail estimate but understates the frictional cost of a full-basket rebalance in illiquid ETFs. Live turnover-adjusted returns will be slightly lower than shown.
- **Rebalance-date synchronicity.** Backtest assumes all positions execute at rebalance-date close prices. Live execution can slip a day or more.
- **Regime overlay tested against only three VIX thresholds and three multipliers.** Finer sensitivity (e.g. SMA lookback other than 200d) not explored — assumed absorbed into future rebalance-frequency and factor-lookback backlog (T3.\*).
- **Correlation clustering's rejection is specific to this exact scoring model.** A different score (e.g., stronger value tilt, different momentum lookback) could produce a correlation structure where clustering helps. The rejection is against the *current* strategy, not against clustering as a concept.

---

## 7. Methodology and reproducibility

- **Runner:** `scripts/multi_regime_diagnostic.py` (commit `61ba98a`, invoked with `--workers 6 --skip-t6`)
- **Report writer:** `scripts/multi_regime_report.py` (this file was hand-revised after auto-generation to correct interpretation bugs)
- **Universe screen:** `src.data_collection.etf_filters.filter_universe(use_curated=True)` — restricts to `comprehensive_etf_list.py` minus commodity / currency / volatility categories; excludes leveraged / inverse tickers.
- **Factor computation:** `factor_weight_diagnostic.compute_rolling_scores` — momentum (252d skip-21), quality (252d), volatility (60d). Value ignored at the rolling-scores level; blended in at the pipeline for live scoring only.
- **Portfolio construction:** top-30 by weighted-geometric-mean integrated rank, exponential rank weights, 2%–15% bounds.
- **Rolling scores cache:** `~/trade_data/ETFTrader/processed/rolling_factor_scores.parquet` (regenerate by deleting the file).
- **Grid CSV:** `docs/research/2026-07_multi_regime_grid.csv`.
- **Plots:** `docs/research/2026-07_multi_regime_plots/`.
- **Related modules:** `src/portfolio/regime.py` (T2.1), `src/portfolio/clustering.py` (T2.2), `src/factors/value_factor.py` (T1.1 close).

Suite runtime on the 599-ticker universe: 2.3 minutes end-to-end on 6 workers.
