"""
Factor Integrator

KEY INNOVATION: Geometric mean integration (AQR approach)

Why geometric mean vs arithmetic mean?
- Arithmetic: Rewards ETFs good on AVERAGE (can be mediocre on all factors)
- Geometric: Rewards ETFs good on ALL factors (penalizes weakness)

Example:
ETF A: Momentum=0.9, Quality=0.9, Value=0.9, Vol=0.9
ETF B: Momentum=1.0, Quality=1.0, Value=0.5, Vol=0.5

Arithmetic mean: A=0.9, B=0.75 → Choose A
Geometric mean: A=0.90, B=0.71 → Choose A (penalizes B's weakness more)

Reference: "Fact, Fiction, and Factor Investing" - AQR (2016)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
try:
    from src.utils.logging_config import get_logger
except ModuleNotFoundError:
    import logging
    get_logger = logging.getLogger

logger = get_logger(__name__)


class FactorIntegrator:
    """
    Integrate multiple factors using weighted geometric mean.

    This is the CORE of the AQR multi-factor approach.
    """

    def __init__(self, factor_weights: Dict[str, float]):
        """
        Initialize integrator with factor weights.

        Args:
            factor_weights: {'momentum': 0.35, 'quality': 0.30, ...}
                           Weights must sum to 1.0

        Example:
            integrator = FactorIntegrator({
                'momentum': 0.35,  # Slightly favor momentum
                'quality': 0.30,
                'value': 0.15,
                'low_volatility': 0.20
            })
        """
        self.factor_weights = factor_weights

        # Validate weights sum to 1.0
        weight_sum = sum(factor_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Factor weights must sum to 1.0, got {weight_sum:.3f}"
            )

        logger.info(
            f"FactorIntegrator initialized with weights: "
            f"{', '.join(f'{k}={v:.2%}' for k, v in factor_weights.items())}"
        )

    def integrate(self, factor_scores: pd.DataFrame) -> pd.Series:
        """
        Integrate factor scores using weighted geometric mean.

        Args:
            factor_scores: DataFrame where:
                          - Columns = factor names (e.g., 'momentum', 'quality')
                          - Rows = ETF tickers
                          - Values = normalized z-scores

        Returns:
            pd.Series: Integrated scores for each ETF

        Mathematical Formula:
            score = ∏(rank_i ^ weight_i) for all factors i
            where rank_i is the percentile rank (0-1) of factor i
        """
        # Validate factor names
        missing_factors = set(self.factor_weights.keys()) - set(factor_scores.columns)
        if missing_factors:
            raise ValueError(
                f"Missing factor scores for: {missing_factors}"
            )

        # Convert z-scores to ranks (0-1) to ensure positive values
        # This is critical for geometric mean
        factor_ranks = pd.DataFrame()
        for factor_name in self.factor_weights.keys():
            # Rank transform: converts to percentile (0-1)
            factor_ranks[factor_name] = factor_scores[factor_name].rank(pct=True)

        # Handle NaN values - set to median (0.5 in percentile space)
        factor_ranks = factor_ranks.fillna(0.5)

        # Weighted geometric mean
        # Formula: ∏(x_i ^ w_i) = exp(∑ w_i * log(x_i))
        integrated = pd.Series(1.0, index=factor_ranks.index)

        for factor_name, weight in self.factor_weights.items():
            # Raise each factor's rank to its weight power
            integrated *= factor_ranks[factor_name] ** weight

        # Log distribution
        logger.debug(
            f"Integrated scores: mean={integrated.mean():.3f}, "
            f"std={integrated.std():.3f}, range=[{integrated.min():.3f}, {integrated.max():.3f}]"
        )

        return integrated

    def get_top_etfs(self, factor_scores: pd.DataFrame,
                     n: int = 20,
                     min_score: float = None) -> pd.DataFrame:
        """
        Get top N ETFs by integrated score.

        Args:
            factor_scores: DataFrame of factor scores
            n: Number of top ETFs to return
            min_score: Minimum integrated score threshold (optional)

        Returns:
            pd.DataFrame: Top ETFs with their integrated scores and ranks
        """
        integrated = self.integrate(factor_scores)

        # Apply minimum score filter if provided
        if min_score is not None:
            integrated = integrated[integrated >= min_score]

        # Get top N
        top_etfs = integrated.nlargest(n)

        # Create detailed DataFrame
        result = pd.DataFrame({
            'integrated_score': top_etfs,
            'rank': range(1, len(top_etfs) + 1)
        })

        # Add individual factor scores for top ETFs
        for factor_name in self.factor_weights.keys():
            result[f'{factor_name}_score'] = factor_scores.loc[top_etfs.index, factor_name]

        logger.info(
            f"Selected top {len(result)} ETFs: "
            f"score range [{result['integrated_score'].min():.3f}, "
            f"{result['integrated_score'].max():.3f}]"
        )

        return result

    def analyze_factor_contributions(self,
                                    factor_scores: pd.DataFrame,
                                    ticker: str) -> Dict:
        """
        Analyze factor contributions for a specific ETF.

        Useful for understanding WHY an ETF was selected/rejected.

        Args:
            factor_scores: DataFrame of factor scores
            ticker: ETF ticker to analyze

        Returns:
            dict: Detailed factor breakdown
        """
        if ticker not in factor_scores.index:
            raise ValueError(f"Ticker {ticker} not found in factor scores")

        # Get this ETF's scores
        etf_scores = factor_scores.loc[ticker]

        # Calculate percentile ranks
        etf_ranks = {}
        for factor_name in self.factor_weights.keys():
            rank_pct = factor_scores[factor_name].rank(pct=True)[ticker]
            etf_ranks[factor_name] = rank_pct

        # Calculate weighted contribution
        contributions = {}
        for factor_name, weight in self.factor_weights.items():
            # Contribution = weight * log(rank)
            # This shows how much each factor "pulls" the integrated score
            rank = etf_ranks[factor_name]
            contribution = weight * np.log(rank) if rank > 0 else -np.inf
            contributions[factor_name] = contribution

        # Get integrated score
        integrated_score = self.integrate(factor_scores.loc[[ticker]])[ticker]

        analysis = {
            'ticker': ticker,
            'integrated_score': integrated_score,
            'integrated_rank': factor_scores.apply(
                lambda col: col.rank(pct=True)
            ).loc[ticker].mean(),
            'factor_scores': etf_scores.to_dict(),
            'factor_ranks': etf_ranks,
            'factor_contributions': contributions,
            'weakest_factor': min(etf_ranks.items(), key=lambda x: x[1]),
            'strongest_factor': max(etf_ranks.items(), key=lambda x: x[1])
        }

        return analysis


class AdaptiveFactorIntegrator(FactorIntegrator):
    """
    Adaptive factor integrator that adjusts weights based on recent performance.

    Advanced feature: Can shift weights dynamically based on which factors
    are working in current market regime.
    """

    def __init__(self,
                 base_weights: Dict[str, float],
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive integrator.

        Args:
            base_weights: Starting factor weights
            adaptation_rate: How quickly to adapt (0-1), 0.1 = 10% adjustment
        """
        super().__init__(base_weights)
        self.base_weights = base_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.performance_history = []

    def update_weights(self, factor_performance: Dict[str, float]):
        """
        Update weights based on recent factor performance.

        Args:
            factor_performance: {'momentum': 0.12, 'quality': 0.08, ...}
                               (returns of portfolios built from each factor)
        """
        # Calculate performance-based adjustment
        total_perf = sum(factor_performance.values())

        if total_perf > 0:
            # Shift weight toward better-performing factors
            for factor_name in self.factor_weights.keys():
                perf = factor_performance.get(factor_name, 0)
                perf_weight = perf / total_perf

                # Adaptive update: move toward performance weight
                current = self.factor_weights[factor_name]
                target = self.base_weights[factor_name] * (1 - self.adaptation_rate) + \
                        perf_weight * self.adaptation_rate

                self.factor_weights[factor_name] = target

            # Renormalize to sum to 1.0
            total = sum(self.factor_weights.values())
            self.factor_weights = {k: v/total for k, v in self.factor_weights.items()}

            logger.info(
                f"Adapted weights: "
                f"{', '.join(f'{k}={v:.2%}' for k, v in self.factor_weights.items())}"
            )
