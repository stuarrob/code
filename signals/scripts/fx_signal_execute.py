"""
FX Skew Divergence Signal — Trade Execution
=============================================
Detects active signals, scores quality, constructs risk reversals,
and executes via IB Gateway on CME FX futures options.

IB Client IDs: 19 (write), 21 (read-only). Port 4001.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --- Signal detection ---


def detect_active_signals(data_dir: Optional[Path] = None) -> List[dict]:
    """Run the divergence signal module on latest data.

    Returns list of active signal dicts with quality scores.
    Only returns signals that fired on the most recent data date.
    """
    import sys

    if data_dir is None:
        data_dir = Path.home() / "trade_data" / "ETFTrader"
    data_dir = Path(data_dir)

    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from fx_skew_divergence import (
        load_pair_data,
        build_divergence_signal,
        build_detection_metrics,
        load_volume_data,
        apply_volume_filter,
        PAIR_CONFIGS,
        SPOT_SIGN,
        CURRENCIES,
        PAIR_MAP,
    )

    active = []
    for ccy in CURRENCIES:
        config = PAIR_CONFIGS[ccy]
        df, mask_both, info = load_pair_data(ccy, data_dir)
        bull_mask, bear_mask, _, _, sig_info = build_divergence_signal(
            df, mask_both, config
        )

        direction = config["trade_direction"]
        signal_mask = bull_mask if direction == "bull" else bear_mask

        # Apply volume filter (P/C ratio)
        volume_df = load_volume_data(ccy, data_dir)
        if not volume_df.empty:
            signal_mask, vol_info = apply_volume_filter(
                df, signal_mask, ccy, volume_df
            )
            print(
                f"  {PAIR_MAP[ccy]}: volume filter "
                f"pass={vol_info['n_pass']}/{vol_info['n_pass']+vol_info['n_fail']}"
            )

        # Check if latest date has a signal
        latest_idx = df.index[-1]
        if not signal_mask.get(latest_idx, False):
            if len(df) > 1:
                prev_idx = df.index[-2]
                if not signal_mask.get(prev_idx, False):
                    continue
                latest_idx = prev_idx
            else:
                continue

        quality = score_signal_quality(df, latest_idx, sig_info, config, ccy)

        # Load SABR params for option construction
        sabr_raw = pd.read_parquet(
            data_dir / "fx_sabr" / "sabr_params_historical.parquet"
        )
        sabr_raw["date"] = pd.to_datetime(sabr_raw["timestamp"]).dt.normalize()
        signal_date = df.loc[latest_idx, "date"]

        sabr_1m = sabr_raw[
            (sabr_raw["currency"] == ccy)
            & (sabr_raw["tenor_bucket"] == "1M")
            & (sabr_raw["date"] == signal_date)
        ]

        sabr_params = None
        if len(sabr_1m) > 0:
            row = sabr_1m.iloc[0]
            sabr_params = {
                "forward": row["forward"],
                "alpha": row["alpha"],
                "rho": row["rho"],
                "nu": row["nu"],
                "T": row["T"],
            }

        active.append(
            {
                "currency": ccy,
                "pair": PAIR_MAP[ccy],
                "direction": direction,
                "signal_date": signal_date,
                "hold_days": config["hold_days"],
                "quality_score": quality["quality"],
                "quality_tercile": quality["tercile"],
                "quality_features": quality,
                "sabr_params": sabr_params,
                "spot_sign": SPOT_SIGN[ccy],
            }
        )

    return active


def score_signal_quality(df, signal_idx, sig_info, config, ccy):
    """Score a single signal using features available at signal time.

    Uses expanding-window percentile ranks (no lookahead).
    """
    from fx_skew_divergence import build_detection_metrics

    fast_tenor = config["fast_tenor"]
    slow_tenor = config["slow_tenor"]

    fast_cols = build_detection_metrics(df, fast_tenor, [config["fast_window"]])
    slow_cols = build_detection_metrics(df, slow_tenor, [config["slow_window"]])
    fast_col = fast_cols[(config["fast_method"], config["fast_window"])]
    slow_col = slow_cols[(config["slow_method"], config["slow_window"])]

    fast_mag = abs(df.loc[signal_idx, fast_col])
    slow_quiet = abs(df.loc[signal_idx, slow_col])
    alpha = df.loc[signal_idx, f"alpha_{fast_tenor}"]
    rho_abs = abs(df.loc[signal_idx, f"rho_{fast_tenor}"])
    nu = df.loc[signal_idx, f"nu_{fast_tenor}"]

    hist = df.loc[:signal_idx]

    def pctile_in_history(value, col):
        vals = hist[col].dropna()
        if len(vals) < 5:
            return 0.5
        return (vals < value).sum() / len(vals)

    fast_mag_pct = pctile_in_history(fast_mag, fast_col)
    slow_quiet_pct = 1.0 - pctile_in_history(slow_quiet, slow_col)
    alpha_pct = pctile_in_history(alpha, f"alpha_{fast_tenor}")
    rho_mid_pct = 1.0 - rho_abs / 0.98
    nu_pct = pctile_in_history(nu, f"nu_{fast_tenor}")

    quality = (
        0.30 * fast_mag_pct
        + 0.20 * slow_quiet_pct
        + 0.15 * alpha_pct
        + 0.15 * rho_mid_pct
        + 0.20 * nu_pct
    )

    if quality > 0.60:
        tercile = "strong"
    elif quality > 0.40:
        tercile = "medium"
    else:
        tercile = "weak"

    return {
        "quality": quality,
        "tercile": tercile,
        "fast_mag": fast_mag,
        "slow_quiet": slow_quiet,
        "alpha": alpha,
        "rho_abs": rho_abs,
        "nu": nu,
    }


# --- Position sizing ---

MARGIN_PER_CONTRACT = {
    "EUR": 1200,  # ~$1,200 for 6E options
    "GBP": 1000,  # ~$1,000 for 6B options
    "JPY": 800,  # ~$800 for 6J options
}

SIZING_MAP = {"strong": 3, "medium": 2, "weak": 1}


def compute_position_size(
    signals: List[dict],
    total_budget: float = 5000.0,
) -> List[dict]:
    """Map quality scores to contract counts within budget.

    Highest quality signals get priority for capital allocation.
    """
    if not signals:
        return signals

    signals = sorted(signals, key=lambda s: s["quality_score"], reverse=True)

    remaining = total_budget
    for sig in signals:
        ccy = sig["currency"]
        margin = MARGIN_PER_CONTRACT.get(ccy, 1000)
        base_contracts = SIZING_MAP.get(sig["quality_tercile"], 1)

        max_affordable = int(remaining / margin) if margin > 0 else 0
        contracts = min(base_contracts, max_affordable)

        if contracts <= 0:
            sig["contracts"] = 0
            sig["margin_estimate"] = 0
            sig["skip_reason"] = "budget_exhausted"
        else:
            sig["contracts"] = contracts
            sig["margin_estimate"] = contracts * margin
            remaining -= sig["margin_estimate"]

    return signals


# --- Risk reversal construction ---

CME_FX_CONFIG = {
    "EUR": {"symbol": "EUR", "exchange": "CME", "multiplier": "125000"},
    "GBP": {"symbol": "GBP", "exchange": "CME", "multiplier": "62500"},
    "JPY": {"symbol": "JPY", "exchange": "CME", "multiplier": "12500000"},
}


def find_25delta_strikes(F, sigma, T):
    """Approximate 25-delta call and put strikes."""
    sqrt_T = np.sqrt(max(T, 1 / 365))
    K_call = F * np.exp(0.6745 * sigma * sqrt_T + 0.5 * sigma**2 * T)
    K_put = F * np.exp(-0.6745 * sigma * sqrt_T + 0.5 * sigma**2 * T)
    return K_call, K_put


def construct_risk_reversal(
    ccy: str,
    direction: str,
    sabr_params: dict,
    ib=None,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 19,
) -> Optional[dict]:
    """Construct a risk reversal using SABR 25-delta strikes,
    snapped to nearest available CME option chain strikes.

    Returns dict with contract details, or None on failure.
    """
    if sabr_params is None:
        return None

    from ib_insync import IB, Future, FuturesOption

    F = sabr_params["forward"]
    sigma = sabr_params["alpha"]
    T = sabr_params["T"]

    K_call, K_put = find_25delta_strikes(F, sigma, T)

    own_connection = False
    if ib is None:
        ib = IB()
        ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=False, timeout=10)
        own_connection = True

    try:
        cme_cfg = CME_FX_CONFIG[ccy]

        fut = Future(symbol=cme_cfg["symbol"], exchange="CME")
        ib.qualifyContracts(fut)

        opt_params = ib.reqSecDefOptParams(fut.symbol, "", fut.secType, fut.conId)

        if not opt_params:
            print(f"  WARNING: No option params found for {ccy}")
            return None

        from datetime import date

        target_expiry = date.today() + timedelta(days=30)

        best_params = None
        best_diff = 999
        best_expiry = None
        best_strikes = []
        for params in opt_params:
            if params.exchange != "CME":
                continue
            for exp_str in params.expirations:
                exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
                diff = abs((exp_date - target_expiry).days)
                if diff < best_diff:
                    best_diff = diff
                    best_params = params
                    best_expiry = exp_str
                    best_strikes = sorted(params.strikes)

        if best_params is None:
            print(f"  WARNING: No suitable expiry found for {ccy}")
            return None

        def snap_strike(target, available):
            return min(available, key=lambda s: abs(s - target))

        call_strike = snap_strike(K_call, best_strikes)
        put_strike = snap_strike(K_put, best_strikes)

        if direction == "bull":
            long_contract = FuturesOption(
                symbol=cme_cfg["symbol"],
                lastTradeDateOrContractMonth=best_expiry,
                strike=call_strike,
                right="C",
                exchange="CME",
                multiplier=cme_cfg["multiplier"],
            )
            short_contract = FuturesOption(
                symbol=cme_cfg["symbol"],
                lastTradeDateOrContractMonth=best_expiry,
                strike=put_strike,
                right="P",
                exchange="CME",
                multiplier=cme_cfg["multiplier"],
            )
        else:
            long_contract = FuturesOption(
                symbol=cme_cfg["symbol"],
                lastTradeDateOrContractMonth=best_expiry,
                strike=put_strike,
                right="P",
                exchange="CME",
                multiplier=cme_cfg["multiplier"],
            )
            short_contract = FuturesOption(
                symbol=cme_cfg["symbol"],
                lastTradeDateOrContractMonth=best_expiry,
                strike=call_strike,
                right="C",
                exchange="CME",
                multiplier=cme_cfg["multiplier"],
            )

        ib.qualifyContracts(long_contract)
        ib.qualifyContracts(short_contract)

        return {
            "long_contract": long_contract,
            "short_contract": short_contract,
            "long_strike": long_contract.strike,
            "short_strike": short_contract.strike,
            "expiry": best_expiry,
            "call_strike_25d": call_strike,
            "put_strike_25d": put_strike,
            "theoretical_call": K_call,
            "theoretical_put": K_put,
            "forward": F,
        }

    finally:
        if own_connection:
            ib.disconnect()


# --- Execution ---


def execute_fx_signals(
    signals: List[dict],
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 19,
    confirm: bool = False,
) -> List[dict]:
    """Execute risk reversal trades for active signals.

    Uses individual LIMIT GTC orders for each leg.
    Safety: confirm=False (default) for dry run.
    """
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if not confirm:
        print("DRY RUN — no orders will be placed.")
        for sig in signals:
            contracts = sig.get("contracts", 0)
            if contracts <= 0:
                results.append(
                    {
                        **sig,
                        "status": "SKIPPED",
                        "reason": sig.get("skip_reason", "no_contracts"),
                    }
                )
                continue
            results.append(
                {
                    **sig,
                    "status": "DRY_RUN",
                    "contracts": contracts,
                    "margin_estimate": sig.get("margin_estimate", 0),
                }
            )
            print(
                f"  {sig['pair']}: Would place {contracts} RR contracts "
                f"({sig['direction']}, quality={sig['quality_tercile']})"
            )
        return results

    from ib_insync import IB, LimitOrder

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=ib_client_id, readonly=False, timeout=10)

    try:
        # Cancel any existing FX option orders
        for trade in ib.openTrades():
            sym = trade.contract.symbol
            if sym in CME_FX_CONFIG:
                print(f"  Cancelling existing order for {sym}: {trade.order.orderId}")
                ib.cancelOrder(trade.order)
        ib.sleep(1)

        for sig in signals:
            contracts = sig.get("contracts", 0)
            if contracts <= 0:
                results.append({**sig, "status": "SKIPPED"})
                continue

            rr = sig.get("rr")
            if rr is None:
                results.append({**sig, "status": "NO_RR_CONSTRUCTED"})
                continue

            long_order = LimitOrder("BUY", contracts, 0)
            long_order.tif = "GTC"
            long_trade = ib.placeOrder(rr["long_contract"], long_order)

            short_order = LimitOrder("SELL", contracts, 0)
            short_order.tif = "GTC"
            short_trade = ib.placeOrder(rr["short_contract"], short_order)

            ib.sleep(2)

            result = {
                **sig,
                "status": "SUBMITTED",
                "long_order_id": long_trade.order.orderId,
                "short_order_id": short_trade.order.orderId,
                "long_status": long_trade.orderStatus.status,
                "short_status": short_trade.orderStatus.status,
            }
            results.append(result)

            print(f"  {sig['pair']}: Submitted {contracts} RR contracts")
            print(
                f"    Long  {rr['long_contract'].right} @ {rr['long_strike']}: "
                f"order {long_trade.order.orderId}"
            )
            print(
                f"    Short {rr['short_contract'].right} @ {rr['short_strike']}: "
                f"order {short_trade.order.orderId}"
            )

        _save_state(signals, results, live_dir)
        _log_trades(results, live_dir)

    finally:
        ib.disconnect()

    return results


def check_fx_positions(
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 21,
) -> dict:
    """Check current FX option positions and pending exits."""
    live_dir = Path(live_dir)
    state_file = live_dir / "signal_state.json"

    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    open_positions = []
    for entry in state.get("open_signals", []):
        exit_date = pd.Timestamp(entry["exit_date"])
        days_left = (exit_date - pd.Timestamp.now()).days
        open_positions.append(
            {
                "pair": entry["pair"],
                "direction": entry["direction"],
                "contracts": entry["contracts"],
                "entry_date": pd.Timestamp(entry["entry_date"]),
                "exit_date": exit_date,
                "days_left": days_left,
                "quality": entry.get("quality_tercile", "unknown"),
                "expired": days_left <= 0,
            }
        )

    return {
        "open_positions": open_positions,
        "n_open": len(open_positions),
        "n_expired": sum(1 for p in open_positions if p["expired"]),
    }


def close_expired_signals(
    live_dir: Path,
    ib_host: str = "127.0.0.1",
    ib_port: int = 4001,
    ib_client_id: int = 19,
    confirm: bool = False,
) -> List[dict]:
    """Close positions where the hold period has expired."""
    status = check_fx_positions(live_dir, ib_host, ib_port, 21)
    expired = [p for p in status["open_positions"] if p["expired"]]

    if not expired:
        print("No expired positions to close.")
        return []

    if not confirm:
        print(f"DRY RUN — {len(expired)} expired positions would be closed:")
        for p in expired:
            print(
                f"  {p['pair']}: {p['contracts']} contracts, "
                f"expired {-p['days_left']} days ago"
            )
        return expired

    # TODO: Implement actual closing logic via IB
    print(f"Closing {len(expired)} expired positions...")
    return expired


# --- State management ---


def _save_state(signals, results, live_dir):
    """Save signal state for position tracking."""
    state_file = live_dir / "signal_state.json"

    state = {"open_signals": []}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    for sig, res in zip(signals, results):
        if res.get("status") not in ("SUBMITTED", "FILLED"):
            continue
        state["open_signals"].append(
            {
                "pair": sig["pair"],
                "currency": sig["currency"],
                "direction": sig["direction"],
                "contracts": sig.get("contracts", 0),
                "entry_date": str(sig["signal_date"]),
                "exit_date": str(
                    sig["signal_date"] + pd.Timedelta(days=sig["hold_days"])
                ),
                "quality_tercile": sig.get("quality_tercile", "unknown"),
                "quality_score": sig.get("quality_score", 0),
                "long_order_id": res.get("long_order_id"),
                "short_order_id": res.get("short_order_id"),
            }
        )

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _log_trades(results, live_dir):
    """Append trades to CSV log."""
    log_file = live_dir / "execution_log.csv"
    file_exists = log_file.exists()

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "pair",
                "direction",
                "contracts",
                "quality",
                "status",
                "long_order_id",
                "short_order_id",
            ],
        )
        if not file_exists:
            writer.writeheader()

        for res in results:
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "pair": res.get("pair", ""),
                    "direction": res.get("direction", ""),
                    "contracts": res.get("contracts", 0),
                    "quality": res.get("quality_tercile", ""),
                    "status": res.get("status", ""),
                    "long_order_id": res.get("long_order_id", ""),
                    "short_order_id": res.get("short_order_id", ""),
                }
            )
