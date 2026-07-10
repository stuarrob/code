"""Compare current portfolio to target and produce a trade blotter.

This is the deterministic bridge between:
  - `pipeline.optimize_portfolio` output (target weights) and
  - `ib_state.IBSnapshot` (current positions + cash + NAV) and
  - operator-supplied `cash_budget` (extra dollars to deploy this cycle)

into a concrete list of BUY / SELL / EXTEND trades, with turnover,
estimated costs, and factor-exposure deltas.

**No orders are placed here.** This module is pure computation; the
downstream applet step handles the guarded live-order path.

Design principles applied
-------------------------
- **Deterministic.** Same inputs → same outputs. No LLM. No random.
- **Slowly-varying.** Small drifts inside the policy's drift threshold
  are ignored — avoids whipsaw from tiny rebalances.
- **Cash-deployment mode.** When `cash_budget > 0`, drift thresholds are
  relaxed for BUY / EXTEND (we WANT to deploy new cash) but preserved
  for SELL (no forced churn).
- **Whole shares.** Integer share quantities; fractional shares are not
  supported on IB retail equity.
- **Minimum trade notional.** Trades below the minimum are dropped to
  avoid IB commission-drag on micro-orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

from src.portfolio.ib_state import IBSnapshot, Position
from src.portfolio.policy import SmartBetaPolicy
from src.backtesting.costs import TransactionCostModel


DEFAULT_MIN_TRADE_NOTIONAL = 500.0  # USD

# Action labels used in blotter and downstream execution.
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_EXTEND = "EXTEND"


@dataclass(frozen=True)
class Trade:
    """One proposed trade.

    All monetary values are USD. Share quantities are integer for equity
    ETFs (IB retail does not support fractional shares).

    Attributes:
        ticker: symbol
        action: BUY (new position), SELL (reduce or exit), EXTEND (add)
        current_shares: shares held before the trade
        target_shares: shares held after the trade
        delta_shares: signed integer (positive = buy, negative = sell)
        market_price: reference price from the snapshot (USD/share)
        delta_notional: |delta_shares| * market_price. Positive.
        est_cost: estimated commission + spread + slippage for this trade
        current_weight_pct: current market_value / NAV, before
        target_weight_pct: target dollar / NAV_after
        weight_gap_pct: target - current (informational)
    """
    ticker: str
    action: str
    current_shares: int
    target_shares: int
    delta_shares: int
    market_price: float
    delta_notional: float
    est_cost: float
    current_weight_pct: float
    target_weight_pct: float
    weight_gap_pct: float


@dataclass(frozen=True)
class FactorExposure:
    """One factor's portfolio-weighted score, before vs after."""
    factor: str
    before: float
    after: float
    delta: float


@dataclass(frozen=True)
class TradeProposal:
    """The complete blotter for a rebalance run.

    Attributes:
        trades: tuple of Trade rows (BUY / SELL / EXTEND). May be empty
            if no meaningful trade emerged from the comparison.
        turnover_notional: sum of |delta_notional| over all trades.
        turnover_pct_of_nav: turnover_notional / nav_before.
        total_est_cost: sum of per-trade est_cost.
        factor_exposures: portfolio-weighted factor scores before/after.
            Empty tuple if scores not supplied.
        investable_nav: NAV + cash_budget - cash_reserve. This is the
            dollar amount that target_weights sum to.
        cash_after: expected cash balance after all trades settle,
            excluding the reserve.
        n_positions_after: number of positions with non-zero target
            after applying all trades.
        warnings: list of soft warnings (e.g. "held-but-not-in-target",
            "cash reserve breached", "trade below min notional dropped").
    """
    trades: tuple[Trade, ...]
    turnover_notional: float
    turnover_pct_of_nav: float
    total_est_cost: float
    factor_exposures: tuple[FactorExposure, ...]
    investable_nav: float
    cash_after: float
    n_positions_after: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────

def propose_trades(
    snapshot: IBSnapshot,
    target_weights: pd.Series,
    cash_budget: float,
    policy: SmartBetaPolicy,
    factor_scores: Optional[pd.DataFrame] = None,
    cost_model: Optional[TransactionCostModel] = None,
    min_trade_notional: float = DEFAULT_MIN_TRADE_NOTIONAL,
    prices: Optional[pd.DataFrame] = None,
) -> TradeProposal:
    """Compute the trade blotter that gets from `snapshot` to `target_weights`.

    Args:
        snapshot: read-only IB account snapshot (positions, cash, NAV).
        target_weights: pd.Series indexed by ticker. Non-negative,
            typically sums to 1.0. Zero-weight tickers are treated as
            "not held after this rebalance".
        cash_budget: additional USD the operator wants to deploy this
            cycle. May be 0.
        policy: SmartBetaPolicy — provides cash_reserve, drift_threshold,
            min_weight, max_weight bounds.
        factor_scores: optional DataFrame indexed by ticker, columns are
            factor names (momentum / quality / volatility / value).
            Used to compute factor-exposure before/after. Optional.
        cost_model: optional TransactionCostModel; defaults to the module
            default (2 bps spread + 2 bps slippage, $0 commission).
        min_trade_notional: drop any proposed trade below this dollar
            value. Prevents micro-orders that lose money to commission
            drag or minimum-lot issues.
        prices: optional wide DataFrame (dates × tickers) providing a
            fallback market price for tickers NOT held in the snapshot.
            The snapshot only carries prices for currently-held names;
            without this fallback, fresh BUYs get skipped with a
            "no market price" warning. When provided, uses the LAST
            available close per ticker.

    Returns:
        TradeProposal.

    Preconditions checked:
        - target_weights >= 0 elementwise (raises)
        - snapshot.nav > 0 (raises)
        - cash_budget >= 0 (raises)
    """
    _validate(snapshot, target_weights, cash_budget)

    cost_model = cost_model or TransactionCostModel()

    # Investable NAV = current NAV + additional budget - cash reserve.
    # This is the dollar total that target_weights sum to.
    investable_nav = float(snapshot.nav) + float(cash_budget) - float(policy.cash_reserve)
    if investable_nav <= 0:
        return _empty_proposal(
            snapshot, cash_budget, warnings=(
                f"Investable NAV ({investable_nav:.2f}) after reserve "
                f"({policy.cash_reserve}) is not positive — no trades proposed.",
            ),
        )

    current = _current_dollar_map(snapshot)  # ticker -> (shares, price, mkt_value)
    warnings: list[str] = []
    trades: list[Trade] = []

    # Build a fallback price lookup for tickers NOT currently held. The
    # snapshot only carries prices for held names; without this fallback
    # a rebalance that proposes 25 fresh BUYs on the top-30 would emit
    # 25 "no market price" warnings and no BUY trades.
    fallback_price: dict[str, float] = {}
    if prices is not None and not prices.empty:
        last_row = prices.ffill().iloc[-1]
        fallback_price = {
            t: float(p) for t, p in last_row.items()
            if pd.notna(p) and p > 0
        }

    # Full universe of tickers to consider: union of current holdings and
    # target weights. Anything with target > 0 might be BUY/EXTEND;
    # anything held but with target = 0 is a SELL candidate.
    all_tickers = set(target_weights.index) | set(current.keys())

    # Cash-deployment mode: relax drift threshold for BUY/EXTEND when
    # the operator has explicitly asked to deploy new capital. Still
    # respect drift for SELL — no forced churn on the way in.
    cash_mode = cash_budget > 0
    drift = float(policy.drift_threshold) * (0.5 if cash_mode else 1.0)

    for ticker in sorted(all_tickers):
        target_pct = float(target_weights.get(ticker, 0.0))
        current_shares, market_price, current_value = current.get(
            ticker, (0, 0.0, 0.0),
        )
        current_pct = current_value / float(snapshot.nav) if snapshot.nav > 0 else 0.0
        target_value = target_pct * investable_nav
        delta_value = target_value - current_value

        # Decide action + skip if inside drift threshold.
        # ASYMMETRIC drift (2026-07-10 fix following live over-invest incident):
        #   - Under-weight retained within drift → SKIP (whipsaw guard)
        #   - Over-weight retained within drift → PROCEED (cash discipline)
        #   - Retained beyond drift (either side) → PROCEED
        #   - Dropped from target (target_pct == 0) → ALWAYS PROCEED (full exit)
        # This closes the loophole where appreciated retained positions
        # accumulated overhang and pushed the total invested past the
        # available cash budget.
        if abs(delta_value) < 1e-6:
            continue
        if current_shares > 0 and target_pct > 0.0:
            gap_pct = target_pct - current_pct  # positive = under-weight
            if 0 <= gap_pct < drift:
                # Under-weight retained, within drift — skip (whipsaw guard).
                continue

        # Below-min notional filter (drop micro-trades regardless of side).
        if abs(delta_value) < min_trade_notional and current_shares > 0:
            warnings.append(
                f"{ticker}: delta {abs(delta_value):.0f} < min notional "
                f"{min_trade_notional:.0f} — skipped"
            )
            continue

        # Market price is required. If we don't have one from the snapshot,
        # fall back to the last close in the price cache (if provided).
        # If still no price, skip and warn — this is the final barrier.
        if market_price <= 0 and target_pct > 0:
            fallback = fallback_price.get(ticker, 0.0)
            if fallback > 0:
                market_price = fallback
            else:
                warnings.append(
                    f"{ticker}: no market price in snapshot or price cache — "
                    f"skipped BUY"
                )
                continue

        # Direction + share count.
        if target_pct == 0.0:
            # SELL full position.
            delta_shares = -int(current_shares)
            action = ACTION_SELL
            target_shares = 0
        elif current_shares == 0:
            # BUY fresh position. Round shares down so we do not overshoot.
            delta_shares = int(target_value / market_price)
            if delta_shares == 0:
                warnings.append(f"{ticker}: BUY sized 0 shares (price too high) — skipped")
                continue
            action = ACTION_BUY
            target_shares = delta_shares
        else:
            # Existing position. Adjust.
            new_shares = int(target_value / market_price)
            delta_shares = new_shares - int(current_shares)
            if delta_shares == 0:
                continue
            if delta_shares > 0:
                action = ACTION_EXTEND
            else:
                action = ACTION_SELL
            target_shares = new_shares

        delta_notional = abs(delta_shares) * market_price
        if delta_notional < min_trade_notional:
            # Post-rounding size fell below min — skip.
            warnings.append(
                f"{ticker}: post-rounding notional {delta_notional:.0f} < "
                f"min {min_trade_notional:.0f} — skipped"
            )
            continue

        est_cost = cost_model.calculate_trade_cost(
            delta_notional, is_buy=(delta_shares > 0),
        )

        trades.append(Trade(
            ticker=ticker,
            action=action,
            current_shares=int(current_shares),
            target_shares=int(target_shares),
            delta_shares=int(delta_shares),
            market_price=float(market_price),
            delta_notional=float(delta_notional),
            est_cost=float(est_cost),
            current_weight_pct=float(current_pct),
            target_weight_pct=float(target_pct),
            weight_gap_pct=float(target_pct - current_pct),
        ))

    # ────────────────────────────────────────────────────────────
    # CASH-NEUTRALITY INVARIANT (2026-07-10 fix)
    # ────────────────────────────────────────────────────────────
    # After Change 2 (asymmetric drift) the raw trade list may still
    # over-invest if the operator has a large ratio of new BUYs to
    # SELLs (e.g. a full universe rotation). Enforce a hard cap:
    #     buys - sells  <=  available_cash
    # When violated, scale every BUY / EXTEND down proportionally.
    # SELLs are never touched (do not force churn).
    available_cash = (
        float(snapshot.cash) + float(cash_budget) - float(policy.cash_reserve)
    )
    buys_notional = sum(t.delta_notional for t in trades if t.delta_shares > 0)
    sells_notional = sum(t.delta_notional for t in trades if t.delta_shares < 0)
    net_cash_out = buys_notional - sells_notional

    if net_cash_out > available_cash + 1e-6 and buys_notional > 0:
        max_allowed_buys = max(available_cash + sells_notional, 0.0)
        scale = max_allowed_buys / buys_notional
        rebuilt: list[Trade] = []
        for t in trades:
            if t.delta_shares <= 0:
                rebuilt.append(t)
                continue
            new_shares = int(t.delta_shares * scale)
            if new_shares == 0:
                # Scaled below one share — drop this BUY entirely and note.
                warnings.append(
                    f"{t.ticker}: BUY scaled to 0 shares by cash cap — dropped"
                )
                continue
            new_notional = new_shares * t.market_price
            if new_notional < min_trade_notional:
                warnings.append(
                    f"{t.ticker}: BUY scaled below min notional "
                    f"{min_trade_notional:.0f} — dropped"
                )
                continue
            new_cost = cost_model.calculate_trade_cost(new_notional, is_buy=True)
            new_target_shares = int(t.current_shares) + new_shares
            rebuilt.append(Trade(
                ticker=t.ticker, action=t.action,
                current_shares=int(t.current_shares),
                target_shares=new_target_shares,
                delta_shares=new_shares,
                market_price=t.market_price,
                delta_notional=float(new_notional),
                est_cost=float(new_cost),
                current_weight_pct=t.current_weight_pct,
                target_weight_pct=t.target_weight_pct,
                weight_gap_pct=t.weight_gap_pct,
            ))
        trades = rebuilt
        warnings.append(
            f"Proposal cash-constrained — BUY notionals scaled to "
            f"{scale:.1%} of intended (available cash "
            f"${available_cash:,.0f}, raw BUYs ${buys_notional:,.0f}, "
            f"SELLs ${sells_notional:,.0f}). Reduces exposure but "
            f"preserves relative BUY weights."
        )

    # Aggregate metrics.
    turnover_notional = sum(t.delta_notional for t in trades)
    turnover_pct = turnover_notional / float(snapshot.nav) if snapshot.nav > 0 else 0.0
    total_cost = sum(t.est_cost for t in trades)

    # Cash after: current cash + sells - buys - costs. Reserve stays put.
    net_buy = sum(t.delta_notional for t in trades if t.delta_shares > 0)
    net_sell = sum(t.delta_notional for t in trades if t.delta_shares < 0)
    cash_after = float(snapshot.cash) + float(cash_budget) + net_sell - net_buy - total_cost

    if cash_after < policy.cash_reserve - 1.0:  # tiny tolerance
        warnings.append(
            f"Post-trade cash {cash_after:.0f} would fall below the "
            f"cash reserve {policy.cash_reserve:.0f} — review sizing."
        )

    # Positions after: anything with target_pct > 0 OR (current > 0 AND not fully sold).
    positions_after = _count_positions_after(current, target_weights, trades)

    exposures = _factor_exposures(
        current, target_weights, factor_scores,
        current_nav=snapshot.nav, investable_nav=investable_nav,
    ) if factor_scores is not None and not factor_scores.empty else ()

    return TradeProposal(
        trades=tuple(trades),
        turnover_notional=float(turnover_notional),
        turnover_pct_of_nav=float(turnover_pct),
        total_est_cost=float(total_cost),
        factor_exposures=exposures,
        investable_nav=float(investable_nav),
        cash_after=float(cash_after),
        n_positions_after=int(positions_after),
        warnings=tuple(warnings),
    )


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _validate(snapshot: IBSnapshot, target_weights: pd.Series,
              cash_budget: float) -> None:
    if snapshot.nav <= 0:
        raise ValueError(f"snapshot.nav must be positive; got {snapshot.nav}")
    if cash_budget < 0:
        raise ValueError(f"cash_budget must be >= 0; got {cash_budget}")
    if (target_weights < -1e-9).any():
        bad = target_weights[target_weights < -1e-9]
        raise ValueError(f"target_weights must be non-negative; got: {bad.to_dict()}")


def _current_dollar_map(snapshot: IBSnapshot) -> dict[str, tuple[float, float, float]]:
    """ticker -> (shares, market_price, market_value)."""
    out: dict[str, tuple[float, float, float]] = {}
    for p in snapshot.long_positions:
        out[p.ticker] = (p.shares, p.market_price, p.market_value)
    return out


def _count_positions_after(current: dict, target: pd.Series,
                            trades: list[Trade]) -> int:
    """Count non-zero positions after applying trades."""
    after: dict[str, float] = {t: shares for t, (shares, _, _) in current.items()}
    for tr in trades:
        after[tr.ticker] = tr.target_shares
    return sum(1 for shares in after.values() if shares > 0)


def _factor_exposures(
    current: dict[str, tuple[float, float, float]],
    target_weights: pd.Series,
    factor_scores: pd.DataFrame,
    current_nav: float,
    investable_nav: float,
) -> tuple[FactorExposure, ...]:
    """Compute portfolio-weighted factor scores before vs after."""
    # Before: current dollar weights on the current NAV.
    before_weights = pd.Series({
        t: mv / current_nav for t, (_, _, mv) in current.items()
    }, dtype=float) if current_nav > 0 else pd.Series(dtype=float)
    # After: target weights on the investable NAV (as fraction of NAV,
    # so cash residual dilutes the exposure).
    after_weights = target_weights.copy() * (investable_nav / current_nav) \
        if current_nav > 0 else target_weights.copy()

    out: list[FactorExposure] = []
    for factor in factor_scores.columns:
        scores = factor_scores[factor]
        b = _weighted_mean(before_weights, scores)
        a = _weighted_mean(after_weights, scores)
        out.append(FactorExposure(factor=factor, before=b, after=a, delta=a - b))
    return tuple(out)


def _weighted_mean(weights: pd.Series, scores: pd.Series) -> float:
    """Dollar-weighted mean of scores over holdings. NaN scores skipped
    but their weight is not renormalised — a missing score reduces
    coverage rather than distorts the mean."""
    if weights.empty:
        return float("nan")
    aligned = weights.reindex(scores.index).fillna(0.0)
    valid = ~scores.isna()
    aligned = aligned[valid]
    scores = scores[valid]
    total_weight = aligned.sum()
    if total_weight <= 0:
        return float("nan")
    return float((aligned * scores).sum() / total_weight)


def _empty_proposal(snapshot: IBSnapshot, cash_budget: float,
                     warnings: tuple[str, ...]) -> TradeProposal:
    return TradeProposal(
        trades=tuple(),
        turnover_notional=0.0,
        turnover_pct_of_nav=0.0,
        total_est_cost=0.0,
        factor_exposures=tuple(),
        investable_nav=float(snapshot.nav) + float(cash_budget),
        cash_after=float(snapshot.cash) + float(cash_budget),
        n_positions_after=len(tuple(p for p in snapshot.long_positions)),
        warnings=warnings,
    )
