"""Unit tests for the ETF smart-beta policy loader.

The policy is the applet's tunable knob set. These tests pin its schema
and validation so a bad TOML file cannot silently drive a live trade.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.portfolio.policy import (
    DEFAULT_POLICY_PATH,
    FactorLookbacks,
    FactorWeights,
    SmartBetaPolicy,
    load_policy,
)


VALID_TOML = """
[meta]
name = "test-policy"
version = 1
tax_status = "taxable"

[portfolio]
num_positions = 20
min_weight = 0.02
max_weight = 0.15

[factor_weights]
momentum = 0.35
quality = 0.30
value = 0.15
volatility = 0.20

[factor_lookbacks]
momentum = 252
momentum_skip_recent = 21
quality = 252
value = 252
volatility = 60

[rebalance]
frequency = "bimonthly"
drift_threshold = 0.05

[risk]
entry_stop_loss_pct = 0.12
trailing_stop_pct = 0.10
cash_reserve = 70000

[optimizer]
risk_aversion = 1.5
robustness_penalty = 0.7
turnover_penalty = 0.2
"""


def _write(tmp_path: Path, toml_text: str) -> Path:
    path = tmp_path / "policy.toml"
    path.write_text(textwrap.dedent(toml_text).strip(), encoding="utf-8")
    return path


@pytest.mark.unit
def test_default_policy_ships_and_loads():
    """The checked-in configs/etf_smart_beta.toml must always load cleanly."""
    assert DEFAULT_POLICY_PATH.exists(), (
        f"Default policy file missing at {DEFAULT_POLICY_PATH} — "
        f"the applet cannot start without it."
    )
    policy = load_policy()
    assert policy.name
    assert policy.version >= 1
    assert 1 <= policy.num_positions <= 100


@pytest.mark.unit
def test_load_from_explicit_path(tmp_path):
    path = _write(tmp_path, VALID_TOML)
    policy = load_policy(path)
    assert policy.name == "test-policy"
    assert policy.num_positions == 20
    assert policy.factor_weights.momentum == 0.35
    assert policy.trailing_stop_pct == 0.10
    assert policy.cash_reserve == 70000


@pytest.mark.unit
def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Policy file not found"):
        load_policy("/no/such/policy.toml")


@pytest.mark.unit
def test_factor_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        FactorWeights(momentum=0.5, quality=0.5, value=0.5, volatility=0.5)


@pytest.mark.unit
def test_factor_weights_reject_negative():
    with pytest.raises(ValueError, match="must be >= 0"):
        FactorWeights(momentum=1.1, quality=0.0, value=-0.1, volatility=0.0)


@pytest.mark.unit
def test_factor_weights_tolerance_allows_rounding(tmp_path):
    """Weights that sum to 1.0000001 due to rounding must be accepted."""
    weights = FactorWeights(momentum=0.3333, quality=0.3333, value=0.3334, volatility=0.0)
    assert abs(sum(weights.as_dict().values()) - 1.0) < 1e-3


@pytest.mark.unit
def test_min_max_weight_ordering(tmp_path):
    """min_weight > max_weight is invalid."""
    bad = VALID_TOML.replace("min_weight = 0.02", "min_weight = 0.20")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="min_weight <= max_weight"):
        load_policy(path)


@pytest.mark.unit
def test_infeasible_sizing_rejected(tmp_path):
    """20 positions * min_weight 0.10 = 2.0 exceeds 1.0 — must fail."""
    bad = VALID_TOML.replace("min_weight = 0.02", "min_weight = 0.10")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="Infeasible sizing"):
        load_policy(path)


@pytest.mark.unit
def test_stop_loss_bounds(tmp_path):
    """Stop losses must be strictly in (0, 1)."""
    bad = VALID_TOML.replace("entry_stop_loss_pct = 0.12", "entry_stop_loss_pct = 1.5")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="entry_stop_loss_pct"):
        load_policy(path)


@pytest.mark.unit
def test_negative_cash_reserve_rejected(tmp_path):
    bad = VALID_TOML.replace("cash_reserve = 70000", "cash_reserve = -1000")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="cash_reserve"):
        load_policy(path)


@pytest.mark.unit
def test_num_positions_must_be_positive(tmp_path):
    bad = VALID_TOML.replace("num_positions = 20", "num_positions = 0")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="num_positions"):
        load_policy(path)


@pytest.mark.unit
def test_factor_lookbacks_reject_zero():
    with pytest.raises(ValueError, match="must be positive"):
        FactorLookbacks(momentum=0)


@pytest.mark.unit
def test_policy_is_frozen(tmp_path):
    """Frozen dataclass — mutation must raise so no code can silently
    change a policy value between load and use."""
    path = _write(tmp_path, VALID_TOML)
    policy = load_policy(path)
    with pytest.raises((AttributeError, TypeError)):
        policy.num_positions = 5  # type: ignore[misc]


@pytest.mark.unit
def test_missing_required_field_raises(tmp_path):
    """A TOML missing a required section must raise cleanly rather than
    silently substituting a default."""
    bad = VALID_TOML.replace("num_positions = 20\n", "")
    path = _write(tmp_path, bad)
    with pytest.raises(KeyError):
        load_policy(path)
