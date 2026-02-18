"""
Leveraged ETF Strategy

Implements the full strategy logic:
- Allocation: 55/45 equity/bond within a vol-sized, capped envelope
- Position sizing: volatility-targeted with 20% hard cap
- Trailing stop loss with VIX-based adjustment
- Cash ratchet to lock in profits
- Re-entry rules after stop-outs
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .signals import SignalState, compute_signals, compute_vol_scalar


@dataclass
class StrategyConfig:
    """Configuration for the leveraged strategy."""

    # Position sizing
    max_allocation: float = 0.20       # 20% hard cap of total portfolio
    target_vol: float = 0.15           # target annualised vol for sizing
    vol_lookback: int = 60             # days for realised vol calculation

    # Allocation splits (within the leveraged envelope)
    equity_split: float = 0.55         # 55% equity leg
    bond_split: float = 0.45           # 45% bond leg
    caution_scale: float = 0.50        # reduce to 50% in CAUTION state

    # Stop loss
    position_stop: float = 0.15        # 15% trailing stop on each position
    portfolio_stop: float = 0.20       # 20% total leveraged drawdown limit

    # Cash ratchet
    ratchet_pct: float = 0.25          # extract 25% of gains above HWM

    # Re-entry
    cooldown_days: int = 5             # wait 5 trading days after stop
    reentry_scale: float = 0.50        # re-enter at 50% size
    reentry_ramp_days: int = 10        # ramp to full over 10 trading days

    # Transaction costs
    spread_bps: float = 5.0            # wider spreads for leveraged ETFs

    # Signal parameters
    momentum_lookback: int = 252
    momentum_skip: int = 21
    sma_period: int = 200
    signal_mode: str = "dual"
    vol_filter_threshold: float = 0.0
    vol_filter_lookback: int = 20

    # VIX integration (optional)
    vix_weight: float = 0.0
    vix_exit_threshold: float = 0.0


@dataclass
class StrategyState:
    """Mutable state tracked during backtest."""

    portfolio_value: float = 0.0
    cash: float = 0.0
    cash_reserve: float = 0.0          # ratcheted profits (locked)
    equity_units: float = 0.0
    bond_units: float = 0.0
    high_water_mark: float = 0.0
    last_signal: str = SignalState.RISK_OFF.value
    stopped_out: bool = False
    stop_date: Optional[pd.Timestamp] = None
    reentry_scale_current: float = 1.0
    equity_high: float = 0.0           # trailing stop reference
    bond_high: float = 0.0


class LeveragedStrategy:
    """Runs the leveraged ETF momentum strategy day by day."""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

    def run(
        self,
        equity_prices: pd.Series,
        bond_prices: pd.Series,
        reference_prices: pd.Series,
        initial_capital: float = 1_000_000,
        use_timing: bool = True,
        use_stops: bool = True,
        use_vol_sizing: bool = True,
        use_ratchet: bool = True,
        vix_prices: pd.Series = None,
    ) -> Dict:
        """Run the strategy over historical data.

        Args:
            equity_prices: Daily closes for equity leg.
            bond_prices: Daily closes for bond leg.
            reference_prices: Daily closes for reference index.
            initial_capital: Starting capital.
            use_timing: Enable momentum timing signals.
            use_stops: Enable trailing stop loss.
            use_vol_sizing: Enable vol-based position sizing.
            use_ratchet: Enable cash ratchet.
            vix_prices: Optional VIX index daily closes
                for composite vol filter / emergency exit.
        """
        cfg = self.config

        # Align all series to common dates
        prices = pd.DataFrame({
            "equity": equity_prices,
            "bond": bond_prices,
            "reference": reference_prices,
        }).dropna()

        min_days = cfg.sma_period + 10
        if cfg.signal_mode != "sma_only":
            min_days = max(min_days, cfg.momentum_lookback + 10)
        if len(prices) < min_days:
            raise ValueError(f"Need at least {min_days} days of data, got {len(prices)}")

        # Compute signals on the reference index
        signals = compute_signals(
            prices["reference"],
            cfg.momentum_lookback,
            cfg.momentum_skip,
            cfg.sma_period,
            signal_mode=cfg.signal_mode,
            vol_filter_threshold=cfg.vol_filter_threshold,
            vol_filter_lookback=cfg.vol_filter_lookback,
            vix_prices=vix_prices,
            vix_weight=cfg.vix_weight,
            vix_exit_threshold=cfg.vix_exit_threshold,
        )

        # Compute vol scalar on the equity leg
        vol_scalar = compute_vol_scalar(prices["equity"], cfg.target_vol, cfg.vol_lookback)

        # Initialise state
        state = StrategyState()
        state.portfolio_value = initial_capital
        state.cash = initial_capital
        state.high_water_mark = initial_capital

        # History tracking
        history = {
            "date": [],
            "portfolio_value": [],
            "invested_value": [],
            "cash": [],
            "cash_reserve": [],
            "signal_state": [],
            "vol_scalar": [],
            "equity_weight": [],
            "bond_weight": [],
            "allocation_pct": [],
        }
        trades = []
        last_rebalance_month = None

        for date in prices.index:
            equity_px = prices.loc[date, "equity"]
            bond_px = prices.loc[date, "bond"]
            signal_row = signals.loc[date] if date in signals.index else None
            signal_state = signal_row["signal_state"] if signal_row is not None else SignalState.RISK_OFF.value
            vs = vol_scalar.get(date, 0.5) if use_vol_sizing else 1.0

            # Mark to market
            invested = state.equity_units * equity_px + state.bond_units * bond_px
            state.portfolio_value = invested + state.cash

            # --- Trailing stop check (daily) ---
            if use_stops and (state.equity_units > 0 or state.bond_units > 0):
                stop_triggered = self._check_stops(state, equity_px, bond_px, date)
                if stop_triggered:
                    # Exit all positions
                    proceeds = state.equity_units * equity_px + state.bond_units * bond_px
                    cost = proceeds * cfg.spread_bps / 10000
                    trades.append({"date": date, "action": "stop_exit", "value": proceeds, "cost": cost})
                    state.cash += proceeds - cost
                    state.equity_units = 0
                    state.bond_units = 0
                    state.stopped_out = True
                    state.stop_date = date
                    state.equity_high = 0
                    state.bond_high = 0

            # --- Weekly signal check + monthly rebalance ---
            is_friday = date.weekday() == 4
            is_new_month = last_rebalance_month is None or date.month != last_rebalance_month
            should_rebalance = (is_friday and is_new_month) or (is_friday and self._signal_flipped(state, signal_state))

            if should_rebalance and not pd.isna(signal_state):
                # Check cooldown after stop-out
                if state.stopped_out:
                    if state.stop_date is not None and (date - state.stop_date).days < cfg.cooldown_days:
                        pass  # still in cooldown
                    else:
                        state.stopped_out = False
                        state.reentry_scale_current = cfg.reentry_scale
                        # Reset HWM to current portfolio value so the
                        # portfolio-level stop measures drawdown from
                        # re-entry, not from the old peak.
                        state.high_water_mark = state.portfolio_value

                if not state.stopped_out:
                    # Determine target allocation
                    target_alloc = self._compute_target_allocation(
                        signal_state, vs, use_timing,
                    )

                    # Scale for re-entry ramp
                    if state.reentry_scale_current < 1.0:
                        target_alloc *= state.reentry_scale_current
                        # Ramp up
                        ramp_step = (1.0 - cfg.reentry_scale) / max(cfg.reentry_ramp_days / 5, 1)
                        state.reentry_scale_current = min(1.0, state.reentry_scale_current + ramp_step)

                    # Execute rebalance
                    total_investable = state.portfolio_value  # cash + invested
                    target_invested = total_investable * target_alloc
                    target_equity_val = target_invested * cfg.equity_split
                    target_bond_val = target_invested * cfg.bond_split

                    new_equity_units = target_equity_val / equity_px if equity_px > 0 else 0
                    new_bond_units = target_bond_val / bond_px if bond_px > 0 else 0

                    # Transaction costs
                    equity_trade_val = abs(new_equity_units - state.equity_units) * equity_px
                    bond_trade_val = abs(new_bond_units - state.bond_units) * bond_px
                    cost = (equity_trade_val + bond_trade_val) * cfg.spread_bps / 10000

                    state.equity_units = new_equity_units
                    state.bond_units = new_bond_units
                    invested = state.equity_units * equity_px + state.bond_units * bond_px
                    state.cash = total_investable - invested - cost

                    # Update trailing stop references
                    state.equity_high = equity_px
                    state.bond_high = bond_px

                    trades.append({
                        "date": date, "action": "rebalance",
                        "signal": signal_state, "allocation": target_alloc,
                        "equity_val": target_equity_val, "bond_val": target_bond_val,
                        "cost": cost,
                    })

                    # Cash ratchet
                    if use_ratchet:
                        self._apply_ratchet(state)

                    last_rebalance_month = date.month
                    state.last_signal = signal_state

            # Update trailing highs
            if state.equity_units > 0:
                state.equity_high = max(state.equity_high, equity_px)
            if state.bond_units > 0:
                state.bond_high = max(state.bond_high, bond_px)

            # Final mark to market
            invested = state.equity_units * equity_px + state.bond_units * bond_px
            state.portfolio_value = invested + state.cash
            total_value = state.portfolio_value + state.cash_reserve

            # Record history
            total_invested = state.equity_units * equity_px + state.bond_units * bond_px
            alloc_pct = total_invested / total_value if total_value > 0 else 0

            history["date"].append(date)
            history["portfolio_value"].append(total_value)
            history["invested_value"].append(invested)
            history["cash"].append(state.cash)
            history["cash_reserve"].append(state.cash_reserve)
            history["signal_state"].append(signal_state)
            history["vol_scalar"].append(vs)
            history["equity_weight"].append(
                state.equity_units * equity_px / total_value if total_value > 0 else 0
            )
            history["bond_weight"].append(
                state.bond_units * bond_px / total_value if total_value > 0 else 0
            )
            history["allocation_pct"].append(alloc_pct)

        history_df = pd.DataFrame(history).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return {
            "history": history_df,
            "trades": trades_df,
            "final_value": state.portfolio_value + state.cash_reserve,
            "final_cash_reserve": state.cash_reserve,
            "final_invested": state.equity_units * prices.iloc[-1]["equity"] + state.bond_units * prices.iloc[-1]["bond"],
        }

    def _compute_target_allocation(
        self,
        signal_state: str,
        vol_scalar: float,
        use_timing: bool,
    ) -> float:
        """Compute target allocation as fraction of total portfolio."""
        cfg = self.config

        if not use_timing:
            # Always risk-on if timing disabled
            base = cfg.max_allocation
        elif signal_state == SignalState.RISK_ON.value:
            base = cfg.max_allocation
        elif signal_state == SignalState.CAUTION.value:
            base = cfg.max_allocation * cfg.caution_scale
        else:
            return 0.0  # risk-off → all cash

        # Apply vol scaling
        return base * min(vol_scalar, 1.0)

    def _check_stops(
        self,
        state: StrategyState,
        equity_px: float,
        bond_px: float,
        date: pd.Timestamp,
    ) -> bool:
        """Check trailing stop loss on positions. Returns True if triggered."""
        cfg = self.config

        # Position-level trailing stops
        if state.equity_high > 0 and equity_px > 0:
            equity_loss = (state.equity_high - equity_px) / state.equity_high
            if equity_loss > cfg.position_stop:
                return True

        if state.bond_high > 0 and bond_px > 0:
            bond_loss = (state.bond_high - bond_px) / state.bond_high
            if bond_loss > cfg.position_stop:
                return True

        # Portfolio-level drawdown
        invested = state.equity_units * equity_px + state.bond_units * bond_px
        total = invested + state.cash
        if state.high_water_mark > 0:
            portfolio_dd = (state.high_water_mark - total) / state.high_water_mark
            if portfolio_dd > cfg.portfolio_stop:
                return True

        return False

    def _signal_flipped(self, state: StrategyState, new_signal: str) -> bool:
        """Check if signal state has changed from last rebalance."""
        return new_signal != state.last_signal

    def _apply_ratchet(self, state: StrategyState):
        """Extract a percentage of gains above HWM to cash reserve."""
        cfg = self.config
        if state.portfolio_value > state.high_water_mark:
            gains = state.portfolio_value - state.high_water_mark
            extract = gains * cfg.ratchet_pct
            state.cash_reserve += extract
            state.cash -= extract
            state.high_water_mark = state.portfolio_value - extract
