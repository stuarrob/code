"""ETF smart-beta strategy policy.

Single source of truth for the tunable parameters that shape the ETF
smart-beta portfolio — factor weights, position sizing, rebalancing
cadence, stop-loss thresholds, cash reserve, optimizer priors.

Loaded once at applet start-up from a TOML file (default:
`configs/etf_smart_beta.toml`) and passed as a frozen dataclass into
the deterministic `src/` pipeline. The applet's UI reads and displays
these values; changing them means editing the TOML and re-loading —
never editing individual scripts.

Per ADR-0001: the deterministic core decides; the LLM only narrates.
This file is the deterministic core's input contract.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


RebalanceFrequency = Literal["weekly", "monthly", "bimonthly", "quarterly"]
TaxStatus = Literal["taxable", "tax_advantaged", "retirement"]

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "etf_smart_beta.toml"

_FACTOR_WEIGHT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class FactorWeights:
    """Factor blend for the integrator. Weights must sum to 1.0."""

    momentum: float
    quality: float
    value: float
    volatility: float

    def __post_init__(self) -> None:
        total = self.momentum + self.quality + self.value + self.volatility
        if abs(total - 1.0) > _FACTOR_WEIGHT_TOLERANCE:
            raise ValueError(
                f"Factor weights must sum to 1.0 (got {total:.6f}). "
                f"Check momentum/quality/value/volatility in the policy file."
            )
        for name, value in asdict(self).items():
            if value < 0.0:
                raise ValueError(f"Factor weight `{name}` must be >= 0 (got {value})")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FactorLookbacks:
    """Trading-day windows for each factor calculation (AQR convention)."""

    momentum: int = 252
    momentum_skip_recent: int = 21
    quality: int = 252
    value: int = 252
    volatility: int = 60

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 1:
                raise ValueError(f"Lookback `{name}` must be positive (got {value})")


@dataclass(frozen=True)
class SmartBetaPolicy:
    """ETF smart-beta policy — the applet's tunable knob set.

    Attributes:
        name: Free-text label (e.g. 'aqr-multifactor-2026').
        version: Policy schema version, bumped on breaking changes.
        num_positions: Target holding count.
        min_weight: Floor on any single position (fraction of NAV).
        max_weight: Ceiling on any single position (fraction of NAV).
        factor_weights: Blend across the four factors.
        rebalance_frequency: Cadence for scheduled rebalances.
        drift_threshold: Fractional deviation that triggers a rebalance.
        entry_stop_loss_pct: Initial stop distance from entry (0.12 = 12%).
        trailing_stop_pct: Trailing stop distance from high (0.10 = 10%).
        cash_reserve: Minimum USD cash to keep unallocated.
        risk_aversion: MVO risk-aversion parameter.
        robustness_penalty: Axioma robustness weight in the optimizer.
        turnover_penalty: Penalty on portfolio turnover.
        factor_lookbacks: Per-factor trading-day windows.
        tax_status: Governs whether tax-lot / wash-sale logic applies.
    """

    name: str
    version: int
    num_positions: int
    min_weight: float
    max_weight: float
    factor_weights: FactorWeights
    rebalance_frequency: RebalanceFrequency
    drift_threshold: float
    entry_stop_loss_pct: float
    trailing_stop_pct: float
    cash_reserve: float
    risk_aversion: float
    robustness_penalty: float
    turnover_penalty: float
    factor_lookbacks: FactorLookbacks = field(default_factory=FactorLookbacks)
    tax_status: TaxStatus = "taxable"

    def __post_init__(self) -> None:
        if self.num_positions < 1:
            raise ValueError(f"num_positions must be >= 1 (got {self.num_positions})")
        if not 0.0 < self.min_weight <= self.max_weight <= 1.0:
            raise ValueError(
                f"Require 0 < min_weight <= max_weight <= 1 "
                f"(got min={self.min_weight}, max={self.max_weight})"
            )
        if self.num_positions * self.min_weight > 1.0 + _FACTOR_WEIGHT_TOLERANCE:
            raise ValueError(
                f"Infeasible sizing: {self.num_positions} positions * "
                f"min_weight {self.min_weight} exceeds 1.0"
            )
        for name, value in [
            ("drift_threshold", self.drift_threshold),
            ("entry_stop_loss_pct", self.entry_stop_loss_pct),
            ("trailing_stop_pct", self.trailing_stop_pct),
        ]:
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1) (got {value})")
        if self.cash_reserve < 0:
            raise ValueError(f"cash_reserve must be >= 0 (got {self.cash_reserve})")


def load_policy(path: Path | str | None = None) -> SmartBetaPolicy:
    """Load a SmartBetaPolicy from a TOML file.

    Args:
        path: Optional override; defaults to `configs/etf_smart_beta.toml`
            at the repo root.

    Returns:
        Frozen, validated SmartBetaPolicy.

    Raises:
        FileNotFoundError: If the policy file does not exist.
        ValueError: If any policy field fails validation.
        KeyError: If a required field is missing from the TOML.
    """
    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    with open(policy_path, "rb") as f:
        raw = tomllib.load(f)

    factor_weights = FactorWeights(**raw["factor_weights"])
    factor_lookbacks = FactorLookbacks(**raw.get("factor_lookbacks", {}))

    portfolio = raw["portfolio"]
    rebalance = raw["rebalance"]
    risk = raw["risk"]
    optimizer = raw["optimizer"]
    meta = raw["meta"]

    return SmartBetaPolicy(
        name=meta["name"],
        version=meta["version"],
        num_positions=portfolio["num_positions"],
        min_weight=portfolio["min_weight"],
        max_weight=portfolio["max_weight"],
        factor_weights=factor_weights,
        rebalance_frequency=rebalance["frequency"],
        drift_threshold=rebalance["drift_threshold"],
        entry_stop_loss_pct=risk["entry_stop_loss_pct"],
        trailing_stop_pct=risk["trailing_stop_pct"],
        cash_reserve=risk["cash_reserve"],
        risk_aversion=optimizer["risk_aversion"],
        robustness_penalty=optimizer["robustness_penalty"],
        turnover_penalty=optimizer["turnover_penalty"],
        factor_lookbacks=factor_lookbacks,
        tax_status=meta.get("tax_status", "taxable"),
    )
