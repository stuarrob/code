"""
Leveraged Strategy Backtesting

Runs the leveraged ETF strategy across multiple pair/config combinations
and computes performance metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .strategy import LeveragedStrategy, StrategyConfig
from .universe import PAIRS, LeveragedPair, load_pair_data


# Predefined backtest configurations
CONFIGS = {
    "A": {
        "name": "Buy-and-hold (no timing)",
        "use_timing": False,
        "use_stops": False,
        "use_vol_sizing": False,
        "use_ratchet": False,
        "config_overrides": {
            "max_allocation": 1.0,
            "equity_split": 1.0, "bond_split": 0.0,
        },
    },
    "B": {
        "name": "SMA 200 timing",
        "use_timing": True,
        "use_stops": False,
        "use_vol_sizing": False,
        "use_ratchet": False,
        "config_overrides": {
            "max_allocation": 1.0,
            "equity_split": 1.0, "bond_split": 0.0,
            "signal_mode": "sma_only",
        },
    },
    "C": {
        "name": "SMA 200 + vol filter",
        "use_timing": True,
        "use_stops": False,
        "use_vol_sizing": False,
        "use_ratchet": False,
        "config_overrides": {
            "max_allocation": 1.0,
            "equity_split": 1.0, "bond_split": 0.0,
            "signal_mode": "sma_only",
            "vol_filter_threshold": 0.20,
        },
    },
    "D": {
        "name": "SMA 200 + vol filter + vol sizing",
        "use_timing": True,
        "use_stops": False,
        "use_vol_sizing": True,
        "use_ratchet": False,
        "config_overrides": {
            "max_allocation": 1.0,
            "equity_split": 1.0, "bond_split": 0.0,
            "signal_mode": "sma_only",
            "vol_filter_threshold": 0.20,
            "target_vol": 0.35,
        },
    },
    "E": {
        "name": "Full strategy (timing + vol + sizing + ratchet)",
        "use_timing": True,
        "use_stops": False,
        "use_vol_sizing": True,
        "use_ratchet": True,
        "config_overrides": {
            "max_allocation": 1.0,
            "equity_split": 1.0, "bond_split": 0.0,
            "signal_mode": "sma_only",
            "vol_filter_threshold": 0.20,
            "target_vol": 0.35,
        },
    },
    "F": {
        "name": "SMA + composite vol (30% VIX blend)",
        "use_timing": True,
        "use_stops": False,
        "use_vol_sizing": False,
        "use_ratchet": False,
        "use_vix": True,
        "config_overrides": {
            "max_allocation": 1.0,
            "equity_split": 1.0, "bond_split": 0.0,
            "signal_mode": "sma_only",
            "vol_filter_threshold": 0.20,
            "vix_weight": 0.3,
        },
    },
    "G": {
        "name": "SMA + refVol + VIX emergency exit (>35)",
        "use_timing": True,
        "use_stops": False,
        "use_vol_sizing": False,
        "use_ratchet": False,
        "use_vix": True,
        "config_overrides": {
            "max_allocation": 1.0,
            "equity_split": 1.0, "bond_split": 0.0,
            "signal_mode": "sma_only",
            "vol_filter_threshold": 0.20,
            "vix_exit_threshold": 35.0,
        },
    },
}


def compute_metrics(history: pd.DataFrame, initial_capital: float = 1_000_000) -> Dict:
    """Compute performance metrics from strategy history DataFrame."""
    values = history["portfolio_value"]
    returns = values.pct_change().dropna()

    if len(returns) < 2:
        return {"cagr": 0, "sharpe": 0, "sortino": 0, "max_drawdown": 0, "calmar": 0}

    # CAGR
    years = (values.index[-1] - values.index[0]).days / 365.25
    total_return = values.iloc[-1] / values.iloc[0]
    cagr = total_return ** (1 / years) - 1 if years > 0 else 0

    # Sharpe
    excess_returns = returns - 0.02 / 252  # risk-free ~2%
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-10
    sortino = (returns.mean() * 252 - 0.02) / downside_std

    # Max drawdown
    cummax = values.cummax()
    drawdowns = (values - cummax) / cummax
    max_dd = drawdowns.min()

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # Time in market
    in_market = (history["allocation_pct"] > 0.01).mean()

    # Trade count
    n_trades = 0
    if "trades" in history.attrs:
        n_trades = len(history.attrs["trades"])

    return {
        "cagr": cagr,
        "total_return": total_return - 1,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "volatility": returns.std() * np.sqrt(252),
        "time_in_market": in_market,
        "final_value": values.iloc[-1],
        "cash_reserve": history["cash_reserve"].iloc[-1],
        "years": years,
    }


def load_vix() -> Optional[pd.Series]:
    """Load VIX index data via yfinance. Cached per session."""
    if not hasattr(load_vix, "_cache"):
        try:
            import yfinance as yf
            raw = yf.download(
                "^VIX",
                start="2010-01-01",
                end="2030-01-01",
                progress=False,
            )
            close = raw[("Close", "^VIX")].squeeze()
            close.index = pd.to_datetime(close.index)
            close.name = "VIX"
            load_vix._cache = close
        except Exception:
            load_vix._cache = None
    return load_vix._cache


def run_single_backtest(
    pair: LeveragedPair,
    data_dir: Path,
    config_key: str = "D",
    initial_capital: float = 1_000_000,
) -> Optional[Dict]:
    """Run a single backtest for one pair/config combo.

    Returns dict with results and metrics, or None if
    data is missing.
    """
    prices = load_pair_data(pair, data_dir)
    if prices is None:
        return None

    config_def = CONFIGS[config_key]

    # Build strategy config with overrides
    cfg = StrategyConfig(
        equity_split=pair.equity_split,
        bond_split=pair.bond_split,
    )
    for k, v in config_def.get("config_overrides", {}).items():
        setattr(cfg, k, v)

    strategy = LeveragedStrategy(cfg)

    # Load VIX if this config needs it
    vix = None
    if config_def.get("use_vix", False):
        vix = load_vix()

    result = strategy.run(
        equity_prices=prices[pair.equity_ticker],
        bond_prices=prices[pair.bond_ticker],
        reference_prices=prices[pair.reference_ticker],
        initial_capital=initial_capital,
        use_timing=config_def["use_timing"],
        use_stops=config_def["use_stops"],
        use_vol_sizing=config_def["use_vol_sizing"],
        use_ratchet=config_def["use_ratchet"],
        vix_prices=vix,
    )

    metrics = compute_metrics(result["history"], initial_capital)

    return {
        "pair": pair.name,
        "config": config_key,
        "config_name": config_def["name"],
        "metrics": metrics,
        "history": result["history"],
        "trades": result["trades"],
        "final_value": result["final_value"],
        "cash_reserve": result["final_cash_reserve"],
    }


def run_all_backtests(
    data_dir: Path,
    pairs: Dict[str, LeveragedPair] = None,
    configs: List[str] = None,
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """Run backtests across all pair/config combinations.

    Returns summary DataFrame with one row per (pair, config).
    """
    if pairs is None:
        pairs = PAIRS
    if configs is None:
        configs = list(CONFIGS.keys())

    data_dir = Path(data_dir)
    results = []

    for pair_key, pair in pairs.items():
        for config_key in configs:
            result = run_single_backtest(pair, data_dir, config_key, initial_capital)
            if result is None:
                print(f"  {pair.name} / Config {config_key}: SKIPPED (missing data)")
                continue

            m = result["metrics"]
            results.append({
                "pair": pair.name,
                "config": config_key,
                "config_name": CONFIGS[config_key]["name"],
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"],
                "calmar": m["calmar"],
                "volatility": m["volatility"],
                "time_in_market": m["time_in_market"],
                "final_value": m["final_value"],
                "cash_reserve": m["cash_reserve"],
                "years": m["years"],
            })

            print(
                f"  {pair.name} / Config {config_key}: "
                f"CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, "
                f"MaxDD={m['max_drawdown']:.1%}, Final=${m['final_value']:,.0f}"
            )

    return pd.DataFrame(results)
