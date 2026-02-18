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
    Robust Mean-Variance Optimizer with shrinkage + Michaud resampling.

    Classical MVO is an "error maximizer" — small noise in expected
    returns produces large, unstable weight swings (Michaud 1989,
    Axioma Research).  This implementation applies three layers of
    defence:

    1. **Ledoit-Wolf covariance shrinkage** — stabilises the sample
       covariance matrix, especially when N assets >> T observations.
    2. **Bayes-Stein return shrinkage** — pulls expected returns
       toward the grand mean, reducing the influence of noisy outlier
       scores.  Controlled by `shrinkage_strength` in [0, 1]:
       0 = no shrinkage (raw scores), 1 = equal returns (no signal).
    3. **Michaud-style resampling** — runs the optimisation
       `n_resample` times on bootstrapped (noise-perturbed) returns,
       then averages weights.  This smooths out sensitivity to any
       single return vector.

    The Axioma risk penalty (gamma * w'sigma) is retained on top,
    penalising high-volatility positions.

    References
    ----------
    - Michaud, "Efficient Asset Management", 1998
    - Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance Matrix",
      Journal of Portfolio Management 30(4), 2004
    - Jorion, "Bayes-Stein Estimation for Portfolio Analysis",
      JFQA 21(3), 1986
    - Axioma Research, "Robust Portfolio Optimization"
    """

    def __init__(self,
                 num_positions: int = 20,
                 min_score: Optional[float] = None,
                 lookback: int = 60,
                 risk_aversion: float = 1.0,
                 axioma_penalty: float = 0.01,
                 use_factor_scores_as_alpha: bool = True,
                 min_weight: float = 0.03,
                 max_weight: float = 0.08,
                 shrinkage_strength: float = 0.5,
                 n_resample: int = 50,
                 resample_noise: float = 0.5):
        """
        Parameters
        ----------
        num_positions : int
            Number of positions to select.
        min_score : float, optional
            Minimum factor score threshold.
        lookback : int
            Days of returns history for covariance estimation.
        risk_aversion : float
            Risk aversion parameter lambda (default: 1.0).
        axioma_penalty : float
            Axioma risk penalty (default: 0.01).
        use_factor_scores_as_alpha : bool
            Use factor scores as expected return signal (default: True).
        min_weight : float
            Minimum weight per position (default: 3%).
        max_weight : float
            Maximum weight per position (default: 8%).
        shrinkage_strength : float
            Bayes-Stein shrinkage toward equal returns (default: 0.5).
            0 = pure factor signal, 1 = equal returns (no signal).
        n_resample : int
            Number of Michaud bootstrap iterations (default: 50).
            0 = single-shot MVO (no resampling).
        resample_noise : float
            Scale of Gaussian noise added to returns per resample,
            as fraction of cross-sectional std (default: 0.5).
        """
        self.num_positions = num_positions
        self.min_score = min_score
        self.lookback = lookback
        self.risk_aversion = risk_aversion
        self.axioma_penalty = axioma_penalty
        self.use_factor_scores_as_alpha = use_factor_scores_as_alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.shrinkage_strength = shrinkage_strength
        self.n_resample = n_resample
        self.resample_noise = resample_noise

    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance with Ledoit-Wolf shrinkage."""
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.values)
            logger.info(
                f"  Ledoit-Wolf shrinkage coefficient: "
                f"{lw.shrinkage_:.3f}"
            )
            return lw.covariance_
        except ImportError:
            logger.warning(
                "sklearn unavailable — using sample covariance"
            )
            return returns.cov().values

    def _shrink_returns(self, raw_alpha: np.ndarray) -> np.ndarray:
        """Bayes-Stein shrinkage toward the cross-sectional mean."""
        grand_mean = raw_alpha.mean()
        return (
            (1 - self.shrinkage_strength) * raw_alpha
            + self.shrinkage_strength * grand_mean
        )

    def _solve_single(self,
                      expected_returns: np.ndarray,
                      cov_matrix: np.ndarray,
                      volatility: np.ndarray,
                      n: int) -> Optional[np.ndarray]:
        """Solve one MVO instance. Returns weight array or None."""
        import cvxpy as cp

        w = cp.Variable(n)
        port_ret = w @ expected_returns
        port_var = cp.quad_form(w, cov_matrix)
        penalty = self.axioma_penalty * (w @ volatility)

        objective = cp.Maximize(
            port_ret
            - self.risk_aversion * port_var
            - penalty
        )
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status in ('optimal', 'optimal_inaccurate'):
            wv = np.array(w.value).flatten()
            wv[wv < 1e-6] = 0
            s = wv.sum()
            if s > 0:
                wv /= s
            return wv
        return None

    def optimize(self,
                 factor_scores: pd.Series,
                 prices: pd.DataFrame) -> pd.Series:
        """
        Robust MVO with shrinkage + Michaud resampling.

        Parameters
        ----------
        factor_scores : pd.Series
            Factor scores (used as expected-return signal).
        prices : pd.DataFrame
            Historical prices for covariance estimation.

        Returns
        -------
        pd.Series
            Portfolio weights.
        """
        try:
            import cvxpy as cp  # noqa: F401
        except ImportError:
            raise ImportError(
                "cvxpy required for MeanVarianceOptimizer. "
                "Install: pip install cvxpy"
            )

        # Select top candidates by factor score
        eligible = factor_scores
        if self.min_score is not None:
            eligible = eligible[eligible >= self.min_score]

        k = min(self.num_positions, len(eligible))
        selected = eligible.nlargest(k).index.tolist()

        # Returns
        rets = (
            prices[selected].pct_change().dropna()
            .tail(self.lookback)
        )
        if len(rets) < self.lookback:
            logger.warning(
                f"Insufficient data: {len(rets)}/{self.lookback}"
            )

        # 1. Ledoit-Wolf covariance shrinkage
        cov_matrix = self._estimate_covariance(rets)

        # 2. Bayes-Stein return shrinkage
        if self.use_factor_scores_as_alpha:
            alpha = factor_scores[selected].values
            alpha_std = alpha.std()
            if alpha_std > 0:
                alpha_norm = (alpha - alpha.mean()) / alpha_std
            else:
                alpha_norm = np.zeros_like(alpha)
            raw_alpha = alpha_norm * 0.02  # ±2% annualised
        else:
            raw_alpha = rets.mean().values * 252

        exp_ret = self._shrink_returns(raw_alpha)
        vol = rets.std().values * np.sqrt(252)

        n = len(selected)

        # 3. Michaud resampling
        if self.n_resample > 0:
            rng = np.random.default_rng(42)
            noise_scale = self.resample_noise * exp_ret.std()
            samples = []
            for _ in range(self.n_resample):
                perturbed = exp_ret + rng.normal(
                    0, noise_scale, size=n
                )
                w_i = self._solve_single(
                    perturbed, cov_matrix, vol, n
                )
                if w_i is not None:
                    samples.append(w_i)

            if samples:
                avg = np.mean(samples, axis=0)
                avg[avg < 1e-6] = 0
                avg /= avg.sum()
                weights = pd.Series(avg, index=selected)
                logger.info(
                    f"  Michaud resampling: "
                    f"{len(samples)}/{self.n_resample} solves OK"
                )
            else:
                logger.error("All resampled solves failed")
                weights = pd.Series(1.0 / n, index=selected)
        else:
            w_opt = self._solve_single(
                exp_ret, cov_matrix, vol, n
            )
            if w_opt is not None:
                weights = pd.Series(w_opt, index=selected)
            else:
                logger.error("Optimisation failed — equal weight")
                weights = pd.Series(1.0 / n, index=selected)

        # Log diagnostics
        wv = weights.values
        pvol = np.sqrt(wv @ cov_matrix @ wv) * np.sqrt(252)
        active = int((weights > 0.01).sum())
        logger.info(
            f"Robust MVO: {active} positions, "
            f"vol={pvol:.2%}"
        )
        logger.info(
            f"  shrinkage={self.shrinkage_strength:.0%}, "
            f"resamples={self.n_resample}, "
            f"axioma={self.axioma_penalty}"
        )

        return weights
