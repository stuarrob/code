"""Value Factor for ETFs.

Two levels of implementation currently supported:

**Preferred — yield + expense ratio blend** (post-2026-07-10, closes T1.1):
    Consumes real per-fund dividend yield from `stable/profile` + real
    per-fund expense ratio from `stable/etf/info`, both cached to
    `~/trade_data/ETFTrader/processed/etf_fundamentals.parquet`. Blend
    is 60% yield + 40% expense-ratio.

    Rationale: trailing 12-month distribution yield is the canonical
    fund-level value tilt (SCHD, VYM, HDV rank high; VUG, QQQ, ARK
    rank low). Expense ratio is the cost-efficiency component. Together
    they express "am I paying up for growth or being paid to hold value".

    FMP fund-level P/E and P/B are not available on any current FMP
    tier (verified 2026-07-10 against key-metrics-ttm, ratios-ttm,
    etf-info, etf-holdings on VOO/SPY/QQQ/VYM/SCHD). When a P/E/P/B
    source is plumbed in later, extend `_compute_blended_value` with
    additional components — the ranking logic is monotonic.

**Fallback — expense ratio only** (backward compatible):
    When only `expense_ratios` is provided. Kept so notebooks and legacy
    tests that pre-date the fundamentals cache still function.
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base_factor import BaseFactor
try:
    from src.utils.logging_config import get_logger
except ModuleNotFoundError:
    import logging
    get_logger = logging.getLogger

logger = get_logger(__name__)


# Blend weights for the two-component value factor. Yield weighted higher
# because it's the more direct academic-value proxy for ETFs; ER is a
# cost-efficiency tilt that matters more for buy-and-hold than for
# smart-beta rotation. Keep as module-level constants so a future tune
# lands in one place, not scattered across the class.
_YIELD_BLEND_WEIGHT = 0.60
_EXPENSE_BLEND_WEIGHT = 0.40


class ValueFactor(BaseFactor):
    """Value factor for ETFs.

    Preferred mode: blended dividend yield + expense ratio.
    Fallback mode: expense ratio only (when yields unavailable).
    """

    def __init__(self):
        super().__init__("value", lookback_period=60)

    def calculate(self,
                  prices: pd.DataFrame,
                  expense_ratios: pd.Series = None,
                  dividend_yields: pd.Series = None,
                  benchmarks: pd.DataFrame = None,
                  **kwargs) -> pd.Series:
        """Calculate value scores for ETFs.

        Args:
            prices: DataFrame (dates × tickers). Used for tracking-error
                fallback and to define the output universe.
            expense_ratios: Series (ticker → decimal expense ratio).
                Required — the fallback mode uses this alone.
            dividend_yields: Series (ticker → decimal yield, e.g. 0.025).
                Optional but strongly preferred. When present the value
                factor is a real yield-based signal, not a cost proxy.
            benchmarks: DataFrame of benchmark prices. Optional; used only
                in the expense-ratio-only mode for a tracking-error tilt.

        Returns:
            pd.Series indexed by ticker with normalised value scores
            (higher = better value). NaN for tickers with no data.
        """
        if expense_ratios is None:
            raise ValueError("ValueFactor requires expense_ratios parameter")

        expense_ratios = expense_ratios.reindex(prices.columns)

        # Preferred path: real fund-level yield + expense ratio blend.
        if dividend_yields is not None and dividend_yields.notna().sum() > 0:
            dividend_yields = dividend_yields.reindex(prices.columns)
            score = self._compute_blended_value(dividend_yields, expense_ratios)
            n_y = int(dividend_yields.notna().sum())
            n_e = int(expense_ratios.notna().sum())
            logger.info(
                "Value factor: blended yield (%.0f%%, %d tickers) + "
                "expense ratio (%.0f%%, %d tickers)",
                100 * _YIELD_BLEND_WEIGHT, n_y,
                100 * _EXPENSE_BLEND_WEIGHT, n_e,
            )
            return score

        # Fallback: expense ratio only.
        er_scores = -1 * expense_ratios  # lower ER = better
        if benchmarks is not None:
            te_scores = self._calculate_tracking_error_scores(prices, benchmarks)
            value_score = 0.6 * self.normalize(er_scores) + 0.4 * self.normalize(te_scores)
            logger.info("Value factor: expense ratio (60%%) + tracking error (40%%) — no yield")
        else:
            value_score = self.normalize(er_scores)
            logger.info("Value factor: expense ratio only — no yield")
        return value_score

    def _compute_blended_value(self,
                                dividend_yields: pd.Series,
                                expense_ratios: pd.Series) -> pd.Series:
        """Blend yield + expense-ratio into a single value score.

        Higher score = better value:
          - Higher yield is better (positive contribution).
          - Lower expense ratio is better (negated).

        Both components are z-score normalised on the current
        cross-section before blending. NaN inputs are preserved (no
        silent imputation) — the caller decides whether to fill or drop.
        """
        yield_score = self.normalize(dividend_yields)          # higher yield → higher score
        er_score = self.normalize(-1.0 * expense_ratios)       # lower ER → higher score
        # Blend with NaN-safe addition: if either component is NaN, the
        # result is NaN. Downstream ranking skips NaNs (BaseFactor
        # convention).
        blended = _YIELD_BLEND_WEIGHT * yield_score + _EXPENSE_BLEND_WEIGHT * er_score
        return blended

    def _calculate_tracking_error_scores(self,
                                        prices: pd.DataFrame,
                                        benchmarks: pd.DataFrame) -> pd.Series:
        """Legacy tracking-error component for the fallback path.

        Kept for backward compatibility with tests that predate the
        fundamentals cache. Not used when `dividend_yields` is provided.
        """
        etf_returns = prices.pct_change().dropna()
        bench_returns = benchmarks.pct_change().dropna()
        common_dates = etf_returns.index.intersection(bench_returns.index)
        etf_returns = etf_returns.loc[common_dates]
        bench_returns = bench_returns.loc[common_dates]
        tracking_diff = etf_returns.sub(bench_returns, axis=0)
        tracking_error = tracking_diff.std() * np.sqrt(252)
        return -1 * tracking_error  # lower TE → higher score


class SimplifiedValueFactor(ValueFactor):
    """Legacy expense-ratio-only value factor.

    Kept so `pipeline.score_factors` continues to work when no yield data
    is available (e.g. fundamentals cache missing). The main
    `ValueFactor` handles both modes now.
    """

    def __init__(self):
        super().__init__()
        self.name = "simplified_value"

    def calculate(self, prices: pd.DataFrame,
                  expense_ratios: pd.Series, **kwargs) -> pd.Series:
        return super().calculate(prices, expense_ratios,
                                  dividend_yields=None, benchmarks=None)
