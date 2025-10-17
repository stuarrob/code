"""
Portfolio Optimizer

Simple optimizer that constructs portfolios from factor scores.
Uses equal-weight approach with optional constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class SimpleOptimizer:
    """
    Simple portfolio optimizer using equal-weighting.

    Selects top N ETFs based on integrated factor scores and
    assigns equal weights to each position.

    Parameters
    ----------
    num_positions : int
        Number of ETFs to hold in portfolio
    min_score : float, optional
        Minimum factor score to be eligible for selection
    max_weight : float, optional
        Maximum weight per position (default: 1/num_positions)
    min_weight : float, optional
        Minimum weight per position (default: 0)
    """

    def __init__(self,
                 num_positions: int = 20,
                 min_score: Optional[float] = None,
                 max_weight: Optional[float] = None,
                 min_weight: float = 0.0):

        self.num_positions = num_positions
        self.min_score = min_score
        self.max_weight = max_weight or (1.0 / num_positions)
        self.min_weight = min_weight

        # Validation
        if num_positions <= 0:
            raise ValueError("num_positions must be positive")

        if self.max_weight < self.min_weight:
            raise ValueError("max_weight must be >= min_weight")

        if self.max_weight > 1.0:
            raise ValueError("max_weight cannot exceed 1.0")

        if self.min_weight * num_positions > 1.0:
            raise ValueError("min_weight * num_positions cannot exceed 1.0")

    def optimize(self, factor_scores: pd.Series) -> pd.Series:
        """
        Construct optimal portfolio from factor scores.

        Parameters
        ----------
        factor_scores : pd.Series
            Factor scores for each ETF (ticker -> score)

        Returns
        -------
        pd.Series
            Portfolio weights (ticker -> weight), sums to 1.0
        """
        if len(factor_scores) == 0:
            raise ValueError("No factor scores provided")

        # Filter by minimum score
        if self.min_score is not None:
            eligible = factor_scores[factor_scores >= self.min_score]
            logger.info(f"After min_score filter: {len(eligible)}/{len(factor_scores)} ETFs eligible")
        else:
            eligible = factor_scores

        if len(eligible) == 0:
            raise ValueError("No ETFs meet minimum score requirement")

        # Select top N
        num_to_select = min(self.num_positions, len(eligible))
        selected = eligible.nlargest(num_to_select)

        logger.info(f"Selected {len(selected)} ETFs for portfolio")

        # Equal weight
        weights = pd.Series(1.0 / len(selected), index=selected.index)

        # Apply weight constraints
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)

        # Renormalize to sum to 1.0
        weights = weights / weights.sum()

        return weights

    def optimize_with_constraints(self,
                                  factor_scores: pd.Series,
                                  sector_map: Optional[Dict[str, str]] = None,
                                  max_sector_weight: float = 0.30) -> pd.Series:
        """
        Optimize with sector concentration constraints.

        Parameters
        ----------
        factor_scores : pd.Series
            Factor scores for each ETF
        sector_map : dict, optional
            Mapping of ticker -> sector
        max_sector_weight : float
            Maximum weight per sector

        Returns
        -------
        pd.Series
            Portfolio weights
        """
        # Start with base optimization
        weights = self.optimize(factor_scores)

        if sector_map is None:
            return weights

        # Check sector constraints
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        # If any sector exceeds limit, adjust
        violations = {s: w for s, w in sector_weights.items() if w > max_sector_weight}

        if violations:
            logger.warning(f"Sector violations: {violations}")
            # Simple approach: reduce weights proportionally
            # More sophisticated approaches could use optimization

            for ticker, weight in weights.items():
                sector = sector_map.get(ticker, 'Unknown')
                if sector in violations:
                    scale_factor = max_sector_weight / violations[sector]
                    weights[ticker] = weight * scale_factor

            # Renormalize
            weights = weights / weights.sum()

        return weights

    def get_position_info(self, weights: pd.Series, factor_scores: pd.Series) -> pd.DataFrame:
        """
        Get detailed information about portfolio positions.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
        factor_scores : pd.Series
            Factor scores

        Returns
        -------
        pd.DataFrame
            Position info with weights and scores
        """
        info = pd.DataFrame({
            'weight': weights,
            'factor_score': factor_scores[weights.index]
        })

        info = info.sort_values('weight', ascending=False)

        return info


class RankBasedOptimizer(SimpleOptimizer):
    """
    Rank-based portfolio optimizer.

    Weights positions based on their factor score rank rather than
    equal-weighting. Higher ranked ETFs get higher weights.
    """

    def __init__(self,
                 num_positions: int = 20,
                 min_score: Optional[float] = None,
                 weighting_scheme: str = 'linear'):
        """
        Parameters
        ----------
        num_positions : int
            Number of positions
        min_score : float, optional
            Minimum score threshold
        weighting_scheme : str
            'linear' or 'exponential' weighting
        """
        super().__init__(num_positions=num_positions, min_score=min_score)
        self.weighting_scheme = weighting_scheme

    def optimize(self, factor_scores: pd.Series) -> pd.Series:
        """Optimize using rank-based weights."""
        # Filter by minimum score
        if self.min_score is not None:
            eligible = factor_scores[factor_scores >= self.min_score]
        else:
            eligible = factor_scores

        if len(eligible) == 0:
            raise ValueError("No ETFs meet minimum score requirement")

        # Select top N
        num_to_select = min(self.num_positions, len(eligible))
        selected = eligible.nlargest(num_to_select)

        # Calculate rank-based weights
        ranks = selected.rank(ascending=False)

        if self.weighting_scheme == 'linear':
            # Weight = (N - rank + 1)
            weights = self.num_positions - ranks + 1
        elif self.weighting_scheme == 'exponential':
            # Weight = exp(-rank / N)
            weights = np.exp(-ranks / self.num_positions)
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")

        # Normalize to sum to 1.0
        weights = weights / weights.sum()

        logger.info(f"Rank-based weights: top={weights.max():.3f}, bottom={weights.min():.3f}")

        return weights


class MinVarianceOptimizer:
    """
    Minimum variance optimizer with Axioma risk adjustment.

    Constructs portfolio that minimizes variance subject to constraints.
    Includes Axioma-style risk penalty for robustness under uncertain returns.

    The Axioma adjustment adds portfolio weights × risk to the objective,
    making the optimal portfolio robust to changes in expected returns.

    **Recommended Configuration**:
    - Use with drift_threshold=0.075 (7.5%) in ThresholdRebalancer
    - This reduces excessive turnover compared to default 5% threshold
    - Real data showed MinVar rebalanced 101 times vs 12 for MVO with 5% threshold
    - Higher threshold reduces transaction costs while maintaining performance

    References:
    - Axioma Portfolio Optimization
    - Black-Litterman with robust optimization
    """

    def __init__(self,
                 num_positions: int = 20,
                 min_score: Optional[float] = None,
                 lookback: int = 60,
                 target_return: Optional[float] = None,
                 risk_penalty: float = 0.01):
        """
        Parameters
        ----------
        num_positions : int
            Number of positions to select (default: 20, per AQR plan)
        min_score : float, optional
            Minimum factor score threshold
        lookback : int
            Days of returns history for covariance estimation
        target_return : float, optional
            Target portfolio return (for efficient frontier)
        risk_penalty : float
            Axioma risk penalty parameter (default: 0.01)
            Higher values = more robust to return uncertainty
        """
        self.num_positions = num_positions
        self.min_score = min_score
        self.lookback = lookback
        self.target_return = target_return
        self.risk_penalty = risk_penalty

    def optimize(self,
                 factor_scores: pd.Series,
                 prices: pd.DataFrame) -> pd.Series:
        """
        Optimize for minimum variance.

        Parameters
        ----------
        factor_scores : pd.Series
            Factor scores for each ETF
        prices : pd.DataFrame
            Historical prices for covariance estimation

        Returns
        -------
        pd.Series
            Portfolio weights
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy required for MinVarianceOptimizer. Install with: pip install cvxpy")

        # Filter by score
        if self.min_score is not None:
            eligible = factor_scores[factor_scores >= self.min_score]
        else:
            eligible = factor_scores

        # Select top candidates
        num_to_select = min(self.num_positions, len(eligible))
        selected_tickers = eligible.nlargest(num_to_select).index.tolist()

        # Calculate returns and covariance
        returns = prices[selected_tickers].pct_change().dropna().tail(self.lookback)

        if len(returns) < self.lookback:
            logger.warning(f"Insufficient data for covariance: {len(returns)}/{self.lookback} days")

        cov_matrix = returns.cov().values

        # Calculate portfolio risk (volatility)
        # This is used in the Axioma adjustment
        portfolio_volatility = returns.std().values

        # Optimization
        n = len(selected_tickers)
        w = cp.Variable(n)

        # Objective: minimize variance with Axioma risk adjustment
        # Standard: min w' Σ w
        # Axioma:   min w' Σ w + λ * w' σ
        # where σ is the volatility vector and λ is the risk penalty
        portfolio_variance = cp.quad_form(w, cov_matrix)

        # Axioma adjustment: add weighted sum of individual risks
        # This makes the portfolio robust to uncertainty in expected returns
        axioma_penalty = self.risk_penalty * (w @ portfolio_volatility)

        objective = cp.Minimize(portfolio_variance + axioma_penalty)

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0,          # Long-only
        ]

        # Optional: target return constraint
        if self.target_return is not None:
            mean_returns = returns.mean().values
            constraints.append(w @ mean_returns >= self.target_return)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            logger.error(f"Optimization failed: {problem.status}")
            # Fall back to equal weight
            weights = pd.Series(1.0 / n, index=selected_tickers)
        else:
            weights = pd.Series(w.value, index=selected_tickers)
            # Clean up tiny weights
            weights[weights < 1e-6] = 0
            weights = weights / weights.sum()

        logger.info(f"Min variance portfolio: {len(weights[weights > 0.01])} non-zero positions")
        logger.info(f"  Axioma risk penalty: {self.risk_penalty}")

        return weights


class MeanVarianceOptimizer:
    """
    Mean-Variance Optimizer (Markowitz) with Axioma adjustment.

    Maximizes: Expected Return - λ * Risk
    With Axioma adjustment for robustness under uncertain returns.

    This is the classic Markowitz portfolio optimization, enhanced with
    the Axioma risk penalty to make it more robust to estimation error
    in expected returns.
    """

    def __init__(self,
                 num_positions: int = 20,
                 min_score: Optional[float] = None,
                 lookback: int = 60,
                 risk_aversion: float = 1.0,
                 axioma_penalty: float = 0.01,
                 use_factor_scores_as_alpha: bool = True):
        """
        Parameters
        ----------
        num_positions : int
            Number of positions to select
        min_score : float, optional
            Minimum factor score threshold
        lookback : int
            Days of returns history for covariance estimation
        risk_aversion : float
            Risk aversion parameter λ (default: 1.0)
            Higher = more risk averse
        axioma_penalty : float
            Axioma risk penalty (default: 0.01)
            Makes optimizer robust to return estimation error
        use_factor_scores_as_alpha : bool
            Use factor scores as expected return signal (default: True)
            If False, uses historical mean returns
        """
        self.num_positions = num_positions
        self.min_score = min_score
        self.lookback = lookback
        self.risk_aversion = risk_aversion
        self.axioma_penalty = axioma_penalty
        self.use_factor_scores_as_alpha = use_factor_scores_as_alpha

    def optimize(self,
                 factor_scores: pd.Series,
                 prices: pd.DataFrame) -> pd.Series:
        """
        Optimize using mean-variance framework with Axioma adjustment.

        Parameters
        ----------
        factor_scores : pd.Series
            Factor scores for each ETF (used as expected return signal)
        prices : pd.DataFrame
            Historical prices for covariance estimation

        Returns
        -------
        pd.Series
            Portfolio weights
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy required for MeanVarianceOptimizer. Install with: pip install cvxpy")

        # Filter by score
        if self.min_score is not None:
            eligible = factor_scores[factor_scores >= self.min_score]
        else:
            eligible = factor_scores

        # Select top candidates
        num_to_select = min(self.num_positions, len(eligible))
        selected_tickers = eligible.nlargest(num_to_select).index.tolist()

        # Calculate returns and covariance
        returns = prices[selected_tickers].pct_change().dropna().tail(self.lookback)

        if len(returns) < self.lookback:
            logger.warning(f"Insufficient data for covariance: {len(returns)}/{self.lookback} days")

        cov_matrix = returns.cov().values

        # Expected returns
        if self.use_factor_scores_as_alpha:
            # Use factor scores as signal of expected returns
            # Normalize to reasonable return range (annualized)
            alpha = factor_scores[selected_tickers].values
            alpha_normalized = (alpha - alpha.mean()) / alpha.std()
            expected_returns = alpha_normalized * 0.10  # Scale to ±10% alpha
        else:
            # Use historical mean returns
            expected_returns = returns.mean().values * 252  # Annualized

        # Individual volatilities for Axioma adjustment
        volatility = returns.std().values * np.sqrt(252)  # Annualized

        # Optimization
        n = len(selected_tickers)
        w = cp.Variable(n)

        # Objective: max(Expected Return - λ * Risk + Axioma adjustment)
        # Standard MVO:  max μ'w - λ * w'Σw
        # With Axioma:   max μ'w - λ * w'Σw - γ * w'σ
        # where μ = expected returns, Σ = covariance, σ = volatility, γ = Axioma penalty

        portfolio_return = w @ expected_returns
        portfolio_variance = cp.quad_form(w, cov_matrix)
        axioma_penalty = self.axioma_penalty * (w @ volatility)

        # Maximize return, minimize risk
        # Note: CVXPY minimizes, so we negate the objective
        objective = cp.Maximize(
            portfolio_return - self.risk_aversion * portfolio_variance - axioma_penalty
        )

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0,          # Long-only
            w <= 0.15,       # Max 15% per position (concentration limit)
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            logger.error(f"Optimization failed: {problem.status}")
            # Fall back to equal weight
            weights = pd.Series(1.0 / n, index=selected_tickers)
        else:
            weights = pd.Series(w.value, index=selected_tickers)
            # Clean up tiny weights
            weights[weights < 1e-6] = 0
            weights = weights / weights.sum()

        # Calculate realized metrics
        realized_return = weights @ expected_returns
        realized_vol = np.sqrt(weights @ cov_matrix @ weights)

        logger.info(f"Mean-variance portfolio: {len(weights[weights > 0.01])} non-zero positions")
        logger.info(f"  Expected return: {realized_return:.2%}")
        logger.info(f"  Expected volatility: {realized_vol:.2%}")
        logger.info(f"  Sharpe estimate: {realized_return / realized_vol:.2f}")
        logger.info(f"  Axioma penalty: {self.axioma_penalty}, Risk aversion: {self.risk_aversion}")

        return weights
