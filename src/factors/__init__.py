"""
Factor Library - AQR Multi-Factor Strategy

All factor implementations for ETF selection.
"""

from .base_factor import BaseFactor
from .momentum_factor import MomentumFactor, DualMomentumFactor
from .quality_factor import QualityFactor, DefensiveQualityFactor
from .value_factor import ValueFactor, SimplifiedValueFactor
from .volatility_factor import VolatilityFactor
from .factor_integrator import FactorIntegrator, AdaptiveFactorIntegrator

__all__ = [
    'BaseFactor',
    'MomentumFactor',
    'DualMomentumFactor',
    'QualityFactor',
    'DefensiveQualityFactor',
    'ValueFactor',
    'SimplifiedValueFactor',
    'VolatilityFactor',
    'FactorIntegrator',
    'AdaptiveFactorIntegrator',
]
