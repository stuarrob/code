"""
Factor Library - AQR Multi-Factor Strategy

All factor implementations for ETF selection.
"""

from src.factors.base_factor import BaseFactor
from src.factors.momentum_factor import MomentumFactor, DualMomentumFactor
from src.factors.quality_factor import QualityFactor, DefensiveQualityFactor
from src.factors.value_factor import ValueFactor, SimplifiedValueFactor
from src.factors.volatility_factor import VolatilityFactor
from src.factors.factor_integrator import FactorIntegrator, AdaptiveFactorIntegrator

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
