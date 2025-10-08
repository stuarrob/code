# AQR Multi-Factor ETF Strategy - Project Plan

## Executive Summary

**Objective**: Build evidence-based multi-factor ETF portfolio strategy inspired by AQR Capital Management research.

**Core Principles**:
- Multi-factor integration (not mixing) - select ETFs strong on MULTIPLE factors
- Low turnover (~10-20 trades/month maximum)
- Weekly rebalancing with high threshold (only rebalance if significant drift)
- Maximum 20 positions
- Stop-loss protection (10-15%)
- Rigorous testing built into development

**Expected Performance**: 12-18% CAGR with Sharpe > 0.8, max drawdown < 20%

---

## Phase 1: Foundation & Data (Week 1)

### 1.1 Preserve Data Collection ✅
**Keep existing**:
- `src/data_collection/etf_universe.py` - ETF collection from yfinance
- `src/data_collection/asset_class_mapper.py` - Hierarchical asset class mapping
- `src/data_collection/etf_filters.py` - Leverage/volatility filters
- `scripts/collect_etf_universe.py` - Universe collection script

**Action**: No changes needed, already in git

### 1.2 Clean Up Old Implementation 🗑️
**Remove** (safely in git):
- `src/optimization/cvxpy_optimizer.py` - Overly complex, overfitted
- `src/signals/momentum_signal.py` - Single-factor, too noisy
- `src/backtesting/` - Rebuild from scratch with proper testing
- `scripts/signal_exploration_grid_search.py` - No longer needed
- `scripts/backtest_*.py` - Old backtests, replace with new

**Keep for reference**:
- Transaction cost model logic
- Performance metrics calculations
- Stop-loss manager concept

### 1.3 New Directory Structure
```
ETFTrader/
├── data/
│   ├── etf_universe.parquet          # Collected ETF data
│   └── factor_scores.parquet         # Calculated factor scores
├── src/
│   ├── data_collection/              # ✅ Keep as-is
│   ├── factors/                      # 🆕 NEW: Factor calculation
│   │   ├── __init__.py
│   │   ├── base_factor.py           # Abstract base class
│   │   ├── momentum_factor.py       # 6-12 month momentum
│   │   ├── value_factor.py          # Expense ratio, tracking error
│   │   ├── quality_factor.py        # Sharpe, Sortino, stability
│   │   ├── volatility_factor.py     # Realized volatility, beta
│   │   └── factor_integrator.py     # Combine factors (geometric mean)
│   ├── portfolio/                    # 🆕 NEW: Portfolio construction
│   │   ├── __init__.py
│   │   ├── constraints.py           # Position limits, diversification
│   │   ├── optimizer.py             # Simple optimizer (no CVXPY)
│   │   ├── rebalancer.py            # Threshold-based rebalancing
│   │   └── risk_manager.py          # Stop-loss, position sizing
│   ├── backtesting/                  # 🆕 NEW: Rebuild with testing
│   │   ├── __init__.py
│   │   ├── engine.py                # Backtest execution
│   │   ├── metrics.py               # Performance calculation
│   │   └── costs.py                 # Transaction cost model
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py
├── tests/                            # 🆕 NEW: Unit tests
│   ├── test_factors/
│   ├── test_portfolio/
│   └── test_backtesting/
├── scripts/
│   ├── 01_collect_universe.py       # ✅ Keep (renamed)
│   ├── 02_calculate_factors.py      # 🆕 Calculate all factor scores
│   ├── 03_backtest_strategy.py      # 🆕 Run backtests
│   └── 04_run_live_portfolio.py     # 🆕 Generate current signals
└── notebooks/
    ├── 01_factor_analysis.ipynb     # Explore factor distributions
    ├── 02_backtest_results.ipynb    # Analyze backtest
    └── 03_portfolio_monitoring.ipynb # Track live portfolio
```

---

## Phase 2: Factor Library (Week 2)

### 2.1 Base Factor Interface
**File**: `src/factors/base_factor.py`

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseFactor(ABC):
    """Abstract base class for all factors."""

    def __init__(self, name: str, lookback_period: int):
        self.name = name
        self.lookback_period = lookback_period

    @abstractmethod
    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate factor score for each ETF.

        Returns:
            pd.Series: Factor scores, normalized to z-score (mean=0, std=1)
        """
        pass

    def normalize(self, scores: pd.Series) -> pd.Series:
        """Standardize scores to z-score."""
        return (scores - scores.mean()) / scores.std()

    def rank(self, scores: pd.Series) -> pd.Series:
        """Rank scores 0-1 (percentile)."""
        return scores.rank(pct=True)
```

### 2.2 Momentum Factor (AQR Research)
**File**: `src/factors/momentum_factor.py`

**Key Insight from Research**: 12-month momentum works best, skip most recent month to avoid reversal

```python
class MomentumFactor(BaseFactor):
    """
    Time-series momentum factor.

    AQR Research: Use 12-month return, skip most recent month.
    Rationale: Avoids short-term reversal while capturing trend.
    """

    def __init__(self, lookback: int = 252, skip_recent: int = 21):
        super().__init__("momentum", lookback)
        self.skip_recent = skip_recent

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate momentum as (price[t-21] / price[t-252]) - 1

        Skip most recent month to avoid short-term reversal.
        """
        if len(prices) < self.lookback_period:
            raise ValueError(f"Need {self.lookback_period} days of data")

        # Get prices from period [t-252 to t-21]
        end_price = prices.iloc[-self.skip_recent]  # 21 days ago
        start_price = prices.iloc[-self.lookback_period]  # 252 days ago

        momentum = (end_price / start_price) - 1

        # Remove outliers (winsorize at 99%)
        momentum = momentum.clip(
            lower=momentum.quantile(0.01),
            upper=momentum.quantile(0.99)
        )

        return self.normalize(momentum)
```

**Test Coverage**:
```python
# tests/test_factors/test_momentum.py
def test_momentum_calculation():
    """Test momentum is calculated correctly."""
    prices = create_test_prices()  # Mock data
    factor = MomentumFactor(lookback=252, skip_recent=21)
    scores = factor.calculate(prices)

    assert scores.mean() < 0.01  # Approximately zero
    assert 0.9 < scores.std() < 1.1  # Approximately 1
    assert scores.isna().sum() == 0  # No NaN values
```

### 2.3 Value Factor (ETF-Specific)
**File**: `src/factors/value_factor.py`

**Key Insight**: For ETFs, "value" = low expense ratio + low tracking error

```python
class ValueFactor(BaseFactor):
    """
    Value factor for ETFs.

    ETFs don't have P/E ratios, so use:
    - Expense ratio (lower = better value)
    - Tracking error (lower = better quality)
    """

    def calculate(self, prices: pd.DataFrame,
                  expense_ratios: pd.Series,
                  benchmarks: pd.DataFrame) -> pd.Series:
        """
        Value score = -1 * (expense_ratio_zscore + tracking_error_zscore)

        Negative because lower is better.
        """
        # Expense ratio component
        er_score = self.normalize(expense_ratios)

        # Tracking error component (if benchmark available)
        if benchmarks is not None:
            tracking_error = self._calculate_tracking_error(prices, benchmarks)
            te_score = self.normalize(tracking_error)
            value_score = -1 * (er_score + te_score) / 2
        else:
            value_score = -1 * er_score

        return value_score

    def _calculate_tracking_error(self, etf_prices, benchmarks):
        """Calculate tracking error vs benchmark."""
        returns_etf = etf_prices.pct_change()
        returns_bench = benchmarks.pct_change()

        tracking_diff = returns_etf - returns_bench
        tracking_error = tracking_diff.std() * np.sqrt(252)  # Annualized

        return tracking_error
```

### 2.4 Quality Factor
**File**: `src/factors/quality_factor.py`

**Key Insight from AQR**: Quality = profitability + safety + growth

For ETFs: Sharpe ratio, drawdown resilience, return stability

```python
class QualityFactor(BaseFactor):
    """
    Quality factor for ETFs.

    Combines:
    - Sharpe ratio (return per unit risk)
    - Maximum drawdown (resilience)
    - Return stability (low return volatility)
    """

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate quality score."""
        returns = prices.pct_change()

        # Sharpe ratio (252-day)
        sharpe = self._calculate_sharpe(returns)

        # Drawdown resilience (inverse of max drawdown)
        max_dd = self._calculate_max_drawdown(prices)
        dd_resilience = -1 * max_dd  # Lower drawdown = better

        # Return stability (inverse of volatility)
        volatility = returns.std() * np.sqrt(252)
        stability = -1 * volatility

        # Combine components
        quality_score = (
            self.normalize(sharpe) +
            self.normalize(dd_resilience) +
            self.normalize(stability)
        ) / 3

        return quality_score

    def _calculate_sharpe(self, returns: pd.DataFrame,
                         risk_free_rate: float = 0.04) -> pd.Series:
        """Calculate Sharpe ratio."""
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        return (annual_return - risk_free_rate) / annual_vol

    def _calculate_max_drawdown(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()  # Most negative value
```

### 2.5 Low Volatility Factor
**File**: `src/factors/volatility_factor.py`

**Key Insight**: Low-volatility anomaly - lower vol ETFs have better risk-adjusted returns

```python
class VolatilityFactor(BaseFactor):
    """
    Low volatility factor.

    Research: Low-vol stocks outperform high-vol on risk-adjusted basis.
    """

    def calculate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate volatility score (lower = better).

        Returns negative z-score so low vol = high score.
        """
        returns = prices.pct_change()

        # Realized volatility (60-day)
        volatility = returns.rolling(60).std() * np.sqrt(252)
        current_vol = volatility.iloc[-1]

        # Return negative so low vol = high score
        vol_score = -1 * self.normalize(current_vol)

        return vol_score
```

### 2.6 Factor Integrator (Key Innovation)
**File**: `src/factors/factor_integrator.py`

**Critical Concept**: INTEGRATE factors (geometric mean), don't MIX (arithmetic mean)

```python
class FactorIntegrator:
    """
    Integrate multiple factors using geometric mean.

    AQR Research: Select stocks good on ALL factors, not average on each.
    Geometric mean rewards consistency across factors.
    """

    def __init__(self, factors: dict):
        """
        Args:
            factors: {'momentum': 0.3, 'value': 0.2, 'quality': 0.3, 'volatility': 0.2}
        """
        self.factors = factors
        assert abs(sum(factors.values()) - 1.0) < 0.01, "Weights must sum to 1"

    def integrate(self, factor_scores: pd.DataFrame) -> pd.Series:
        """
        Combine factor scores using weighted geometric mean.

        Args:
            factor_scores: DataFrame with columns = factor names

        Returns:
            pd.Series: Integrated score for each ETF
        """
        # Convert z-scores to ranks (0-1) to ensure positive values
        factor_ranks = factor_scores.rank(pct=True)

        # Weighted geometric mean
        integrated = pd.Series(1.0, index=factor_ranks.index)

        for factor_name, weight in self.factors.items():
            integrated *= factor_ranks[factor_name] ** weight

        # Take nth root (where n = sum of weights = 1)
        # This keeps scale comparable to input ranks

        return integrated

    def get_top_etfs(self, factor_scores: pd.DataFrame, n: int = 20) -> pd.Series:
        """Get top N ETFs by integrated score."""
        integrated = self.integrate(factor_scores)
        return integrated.nlargest(n)
```

**Example Usage**:
```python
# Calculate individual factors
momentum = MomentumFactor().calculate(prices)
value = ValueFactor().calculate(prices, expense_ratios)
quality = QualityFactor().calculate(prices)
volatility = VolatilityFactor().calculate(prices)

# Combine into DataFrame
factor_scores = pd.DataFrame({
    'momentum': momentum,
    'value': value,
    'quality': quality,
    'volatility': volatility
})

# Integrate using geometric mean
integrator = FactorIntegrator({
    'momentum': 0.35,    # Slightly favor momentum
    'value': 0.15,       # Value less important for ETFs
    'quality': 0.30,     # Quality important
    'volatility': 0.20   # Low-vol anomaly
})

top_20 = integrator.get_top_etfs(factor_scores, n=20)
```

---

## Phase 3: Portfolio Construction (Week 3)

### 3.1 Simple Optimizer (No CVXPY!)
**File**: `src/portfolio/optimizer.py`

**Key Change**: Use simple ranking + constraints, not complex convex optimization

```python
class SimplePortfolioOptimizer:
    """
    Simple portfolio optimizer using factor scores.

    No complex optimization - just rank, filter, weight.
    Research shows simple often beats complex.
    """

    def __init__(self,
                 max_positions: int = 20,
                 max_single_position: float = 0.10,
                 min_position: float = 0.02):
        self.max_positions = max_positions
        self.max_single_position = max_single_position
        self.min_position = min_position

    def optimize(self,
                 factor_scores: pd.Series,
                 current_positions: dict = None,
                 constraints: dict = None) -> dict:
        """
        Generate target portfolio weights.

        Args:
            factor_scores: Integrated factor scores for each ETF
            current_positions: Current holdings {ticker: weight}
            constraints: Asset class limits, sector limits, etc.

        Returns:
            dict: {ticker: target_weight}
        """
        # Step 1: Select top N by factor score
        top_etfs = factor_scores.nlargest(self.max_positions)

        # Step 2: Initial equal weighting
        base_weight = 1.0 / len(top_etfs)
        weights = {ticker: base_weight for ticker in top_etfs.index}

        # Step 3: Apply score-based tilting (optional)
        # Tilt toward highest scores, but cap at max_single_position
        score_weights = top_etfs / top_etfs.sum()
        for ticker in weights:
            tilted_weight = 0.5 * base_weight + 0.5 * score_weights[ticker]
            weights[ticker] = min(tilted_weight, self.max_single_position)

        # Step 4: Renormalize to sum to 1.0
        total = sum(weights.values())
        weights = {t: w/total for t, w in weights.items()}

        # Step 5: Apply constraints
        if constraints:
            weights = self._apply_constraints(weights, constraints)

        # Step 6: Remove tiny positions
        weights = {t: w for t, w in weights.items() if w >= self.min_position}

        # Step 7: Final renormalization
        total = sum(weights.values())
        weights = {t: w/total for t, w in weights.items()}

        return weights

    def _apply_constraints(self, weights: dict, constraints: dict) -> dict:
        """Apply asset class, sector, and other constraints."""
        # Implementation: Check asset class exposure, adjust if needed
        # Keep simple - just cap exposures, don't optimize
        pass
```

### 3.2 Threshold Rebalancer
**File**: `src/portfolio/rebalancer.py`

**Key Feature**: Only rebalance when positions drift significantly (Vanguard approach)

```python
class ThresholdRebalancer:
    """
    Rebalance only when portfolio drifts beyond threshold.

    Reduces turnover while maintaining target allocation.
    Research: Threshold-based beats calendar-based.
    """

    def __init__(self,
                 drift_threshold: float = 0.05,  # 5% drift triggers rebalance
                 max_trades_per_rebalance: int = 10):
        self.drift_threshold = drift_threshold
        self.max_trades = max_trades_per_rebalance

    def should_rebalance(self,
                        current_weights: dict,
                        target_weights: dict) -> bool:
        """Check if rebalance needed."""
        max_drift = 0

        # Check drift for each position
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            drift = abs(current - target)
            max_drift = max(max_drift, drift)

        return max_drift > self.drift_threshold

    def generate_trades(self,
                       current_weights: dict,
                       target_weights: dict,
                       portfolio_value: float) -> list:
        """
        Generate minimal set of trades to reach target.

        Prioritize largest drifts, limit total number of trades.
        """
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        # Calculate drift for each ticker
        drifts = []
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            drift = target - current

            if abs(drift) > 0.01:  # Only trade if >1% drift
                drifts.append({
                    'ticker': ticker,
                    'current_weight': current,
                    'target_weight': target,
                    'drift': drift,
                    'dollar_amount': drift * portfolio_value
                })

        # Sort by absolute drift (largest first)
        drifts.sort(key=lambda x: abs(x['drift']), reverse=True)

        # Limit to max trades
        trades = drifts[:self.max_trades]

        return trades
```

### 3.3 Risk Manager (Stop-Loss)
**File**: `src/portfolio/risk_manager.py`

```python
class RiskManager:
    """
    Portfolio risk management: stop-loss, position sizing.
    """

    def __init__(self,
                 stop_loss_pct: float = 0.12,  # 12% stop-loss
                 trailing_stop: bool = True):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop = trailing_stop
        self.position_highs = {}  # Track high water mark for trailing stops

    def check_stop_losses(self,
                         positions: dict,
                         current_prices: pd.Series) -> list:
        """
        Check if any positions hit stop-loss.

        Returns:
            list: Tickers that hit stop-loss
        """
        triggered = []

        for ticker, position_info in positions.items():
            current_price = current_prices[ticker]
            purchase_price = position_info['purchase_price']

            # Update high water mark for trailing stop
            if self.trailing_stop:
                if ticker not in self.position_highs:
                    self.position_highs[ticker] = purchase_price
                else:
                    self.position_highs[ticker] = max(
                        self.position_highs[ticker],
                        current_price
                    )

                stop_price = self.position_highs[ticker] * (1 - self.stop_loss_pct)
            else:
                stop_price = purchase_price * (1 - self.stop_loss_pct)

            # Check if stop triggered
            if current_price < stop_price:
                loss_pct = (current_price / purchase_price) - 1
                triggered.append({
                    'ticker': ticker,
                    'purchase_price': purchase_price,
                    'current_price': current_price,
                    'stop_price': stop_price,
                    'loss_pct': loss_pct
                })

        return triggered
```

---

## Phase 4: Backtesting with Tests (Week 4)

### 4.1 Backtest Engine
**File**: `src/backtesting/engine.py`

**Key Feature**: Built with testing in mind, modular components

```python
class BacktestEngine:
    """
    Backtest engine for multi-factor strategy.

    Focuses on:
    - Accurate cost modeling
    - Realistic slippage
    - Stop-loss execution
    - Weekly rebalancing with threshold
    """

    def __init__(self,
                 factor_integrator: FactorIntegrator,
                 optimizer: SimplePortfolioOptimizer,
                 rebalancer: ThresholdRebalancer,
                 risk_manager: RiskManager,
                 initial_capital: float = 1_000_000):

        self.factor_integrator = factor_integrator
        self.optimizer = optimizer
        self.rebalancer = rebalancer
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital

        # State
        self.portfolio_value = initial_capital
        self.positions = {}
        self.cash = initial_capital
        self.trade_history = []
        self.value_history = []

    def run(self,
            prices: pd.DataFrame,
            factor_data: dict,
            start_date: str,
            end_date: str) -> dict:
        """
        Run backtest.

        Returns:
            dict: Performance metrics, trade history, portfolio values
        """
        dates = prices.loc[start_date:end_date].index

        # Weekly rebalance dates (every 7 days)
        rebalance_dates = dates[::7]

        for date in dates:
            # Update portfolio value
            self._update_portfolio_value(date, prices)

            # Check stop-losses daily
            stop_losses = self.risk_manager.check_stop_losses(
                self.positions,
                prices.loc[date]
            )
            if stop_losses:
                self._execute_stop_losses(date, prices, stop_losses)

            # Rebalance weekly (if threshold exceeded)
            if date in rebalance_dates:
                self._rebalance(date, prices, factor_data)

            # Record daily value
            self.value_history.append({
                'date': date,
                'value': self.portfolio_value,
                'cash': self.cash,
                'num_positions': len(self.positions)
            })

        # Calculate performance metrics
        results = self._calculate_metrics()

        return results

    def _rebalance(self, date, prices, factor_data):
        """Execute weekly rebalance."""
        # Calculate factor scores as of this date
        factor_scores = self._calculate_factors_at_date(date, prices, factor_data)

        # Integrate factors
        integrated_scores = self.factor_integrator.integrate(factor_scores)

        # Generate target weights
        target_weights = self.optimizer.optimize(
            integrated_scores,
            current_positions=self.positions
        )

        # Check if rebalance needed
        current_weights = self._get_current_weights()
        if not self.rebalancer.should_rebalance(current_weights, target_weights):
            return  # Skip rebalance

        # Generate trades
        trades = self.rebalancer.generate_trades(
            current_weights,
            target_weights,
            self.portfolio_value
        )

        # Execute trades
        self._execute_trades(date, prices, trades)
```

### 4.2 Unit Tests (Critical!)
**File**: `tests/test_factors/test_momentum.py`

```python
import pytest
import pandas as pd
import numpy as np
from src.factors.momentum_factor import MomentumFactor

class TestMomentumFactor:
    """Test momentum factor calculation."""

    def test_basic_calculation(self):
        """Test momentum calculates correctly."""
        # Create test data: linearly increasing prices
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = pd.DataFrame({
            'ETF_A': np.linspace(100, 150, 300),  # 50% gain
            'ETF_B': np.linspace(100, 100, 300),  # Flat
            'ETF_C': np.linspace(100, 80, 300),   # 20% loss
        }, index=dates)

        factor = MomentumFactor(lookback=252, skip_recent=21)
        scores = factor.calculate(prices)

        # ETF_A should have highest score
        assert scores['ETF_A'] > scores['ETF_B'] > scores['ETF_C']

        # Scores should be normalized (mean ≈ 0, std ≈ 1)
        assert abs(scores.mean()) < 0.1
        assert 0.9 < scores.std() < 1.1

    def test_skip_recent_month(self):
        """Test that recent month is correctly skipped."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create price that jumps in last month
        prices_base = np.linspace(100, 120, 279)  # Gradual rise
        prices_spike = np.linspace(120, 200, 21)  # Huge spike last month
        prices = pd.DataFrame({
            'ETF_A': np.concatenate([prices_base, prices_spike])
        }, index=dates)

        factor = MomentumFactor(lookback=252, skip_recent=21)
        score = factor.calculate(prices)['ETF_A']

        # Score should be based on 100->120 gain (20%), not 100->200 (100%)
        # Exact value depends on normalization, but should be modest
        assert score < 2.0  # If using full gain, would be much higher

    def test_insufficient_data(self):
        """Test error handling with insufficient data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({'ETF_A': np.random.randn(100)}, index=dates)

        factor = MomentumFactor(lookback=252, skip_recent=21)

        with pytest.raises(ValueError):
            factor.calculate(prices)
```

### 4.3 Integration Tests
**File**: `tests/test_portfolio/test_integration.py`

```python
def test_full_strategy_pipeline():
    """Test complete strategy from data -> signals -> trades."""
    # Load test data (small sample)
    prices = load_test_prices()  # 50 ETFs, 2 years

    # Calculate factors
    factors = calculate_all_factors(prices)

    # Integrate
    integrator = FactorIntegrator({
        'momentum': 0.4,
        'quality': 0.3,
        'value': 0.15,
        'volatility': 0.15
    })
    integrated = integrator.integrate(factors)

    # Optimize
    optimizer = SimplePortfolioOptimizer(max_positions=20)
    weights = optimizer.optimize(integrated)

    # Validate results
    assert len(weights) <= 20
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(w >= 0.02 for w in weights.values())
    assert all(w <= 0.10 for w in weights.values())
```

---

## Phase 5: Implementation Scripts (Week 5)

### 5.1 Calculate Factors Script
**File**: `scripts/02_calculate_factors.py`

```python
"""
Calculate factor scores for entire ETF universe.

Saves to: data/factor_scores.parquet
"""

import pandas as pd
from pathlib import Path
from src.factors import (
    MomentumFactor,
    ValueFactor,
    QualityFactor,
    VolatilityFactor
)

def main():
    # Load ETF universe
    prices = pd.read_parquet('data/etf_universe.parquet')

    # Calculate each factor
    print("Calculating momentum...")
    momentum = MomentumFactor(lookback=252, skip_recent=21)
    momentum_scores = momentum.calculate(prices)

    print("Calculating quality...")
    quality = QualityFactor(lookback=252)
    quality_scores = quality.calculate(prices)

    print("Calculating value...")
    value = ValueFactor()
    value_scores = value.calculate(prices, expense_ratios)

    print("Calculating volatility...")
    volatility = VolatilityFactor(lookback=60)
    volatility_scores = volatility.calculate(prices)

    # Combine into single DataFrame
    factor_scores = pd.DataFrame({
        'momentum': momentum_scores,
        'quality': quality_scores,
        'value': value_scores,
        'volatility': volatility_scores
    })

    # Save
    factor_scores.to_parquet('data/factor_scores.parquet')
    print(f"Saved factor scores for {len(factor_scores)} ETFs")

if __name__ == '__main__':
    main()
```

### 5.2 Backtest Script
**File**: `scripts/03_backtest_strategy.py`

```python
"""
Run backtest of multi-factor strategy.

Tests multiple periods:
- 2017-2020 (bull market)
- 2020-2022 (COVID recovery)
- 2022-2024 (inflation/rates)
"""

def main():
    # Load data
    prices = pd.read_parquet('data/etf_universe.parquet')

    # Setup strategy
    integrator = FactorIntegrator({
        'momentum': 0.35,
        'quality': 0.30,
        'value': 0.15,
        'volatility': 0.20
    })

    optimizer = SimplePortfolioOptimizer(max_positions=20)
    rebalancer = ThresholdRebalancer(drift_threshold=0.05)
    risk_manager = RiskManager(stop_loss_pct=0.12)

    engine = BacktestEngine(
        integrator, optimizer, rebalancer, risk_manager,
        initial_capital=1_000_000
    )

    # Run backtests on different periods
    periods = [
        ('2017-01-01', '2020-02-28', 'Bull Market'),
        ('2020-03-01', '2022-01-01', 'COVID Recovery'),
        ('2022-01-01', '2024-10-01', 'Inflation/Rates')
    ]

    for start, end, label in periods:
        print(f"\n{'='*60}")
        print(f"Backtest: {label} ({start} to {end})")
        print(f"{'='*60}")

        results = engine.run(prices, factor_data, start, end)

        print(f"CAGR: {results['cagr']:.1%}")
        print(f"Sharpe: {results['sharpe']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Avg Turnover: {results['avg_monthly_turnover']:.1%}")
        print(f"Total Trades: {results['num_trades']}")

        # Save detailed results
        results.to_pickle(f'results/backtest_{label.lower()}.pkl')
```

---

## Phase 6: Validation & Monitoring (Week 6)

### 6.1 Out-of-Sample Validation
- Train on 2017-2020
- Test on 2020-2022 (unseen data)
- Validate parameters didn't overfit

### 6.2 Performance Targets

| Metric | Target | Baseline (SPY) |
|--------|--------|----------------|
| CAGR | 12-18% | ~12% |
| Sharpe Ratio | > 0.8 | ~0.6 |
| Max Drawdown | < 20% | ~25% |
| Win Rate | > 55% | N/A |
| Monthly Turnover | < 30% | 0% |
| Avg Trades/Month | 10-20 | 0 |

### 6.3 Live Monitoring Dashboard
**Notebook**: `notebooks/03_portfolio_monitoring.ipynb`

Track:
- Current positions vs target weights
- Factor score evolution
- Stop-loss distances
- Transaction costs YTD
- Rolling Sharpe ratio

---

## Implementation Checklist

### Week 1: Foundation ✅
- [x] Clean up old code (safely in git)
- [ ] Create new directory structure
- [ ] Setup testing framework (pytest)
- [ ] Create logging configuration

### Week 2: Factor Library 🏗️
- [ ] Implement BaseFactor abstract class
- [ ] Implement MomentumFactor with tests
- [ ] Implement ValueFactor with tests
- [ ] Implement QualityFactor with tests
- [ ] Implement VolatilityFactor with tests
- [ ] Implement FactorIntegrator with tests
- [ ] Integration test: Full factor pipeline

### Week 3: Portfolio Construction 🏗️
- [ ] Implement SimplePortfolioOptimizer
- [ ] Implement ThresholdRebalancer
- [ ] Implement RiskManager
- [ ] Unit tests for all components
- [ ] Integration test: Optimizer + Rebalancer

### Week 4: Backtesting 🏗️
- [ ] Implement BacktestEngine
- [ ] Implement transaction cost model
- [ ] Implement performance metrics
- [ ] Unit tests for backtest components
- [ ] End-to-end backtest test

### Week 5: Scripts & Execution 🏗️
- [ ] Calculate factors script
- [ ] Backtest script (multi-period)
- [ ] Live portfolio generation script
- [ ] Validate all scripts work end-to-end

### Week 6: Validation & Docs 📊
- [ ] Run backtests on 3 periods
- [ ] Validate performance targets
- [ ] Create monitoring notebook
- [ ] Document findings
- [ ] Final report

---

## Key Design Principles

### 1. Simplicity Over Complexity
- ❌ NO CVXPY, complex solvers
- ✅ YES simple ranking, weighting, constraints
- Research shows simple often beats complex after costs

### 2. Testing Built In
- Every component has unit tests
- Integration tests for pipelines
- Backtest validation on multiple periods

### 3. Low Turnover by Design
- Threshold-based rebalancing (not calendar)
- Skip recent month in momentum (avoid reversal)
- Limit trades per rebalance (max 10-20)
- Factor integration (not mixing) for stability

### 4. Risk Management
- 12% stop-loss (can be trailing)
- Max 20 positions
- Position size limits (2-10%)
- Asset class diversification

### 5. Transparency
- All factors clearly defined
- Performance attribution by factor
- Transaction costs explicitly tracked
- No black boxes

---

## Success Criteria

### Must Have ✅
- [ ] Backtest CAGR > 12% across all 3 periods
- [ ] Sharpe ratio > 0.8 in at least 2/3 periods
- [ ] Monthly turnover < 30%
- [ ] Max drawdown < 25% in worst period
- [ ] All tests passing (>90% coverage)

### Nice to Have 🎯
- [ ] Outperform SPY in 2/3 periods
- [ ] Sharpe > 1.0 in bull market
- [ ] Win rate > 60%
- [ ] Trades/month < 15

### Red Flags 🚩
- Monthly turnover > 50% → Signal too noisy
- Max drawdown > 30% → Risk management failing
- Sharpe < 0.5 → Not adding value after costs
- Win rate < 50% → No edge

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Foundation | Clean structure, testing setup |
| 2 | Factors | 4 factor classes + integrator, all tested |
| 3 | Portfolio | Optimizer, rebalancer, risk manager |
| 4 | Backtesting | Engine + metrics, fully tested |
| 5 | Scripts | End-to-end pipeline working |
| 6 | Validation | Backtests complete, report ready |
| 7 | Web App (Phase 1) | Dashboard - view recommendations, manual trade entry |
| 8 | Web App (Phase 2) | Performance monitoring, portfolio tracking |
| 9 | IB Integration (Future) | Interactive Brokers data + automated trading |

**Total**: 6 weeks to strategy completion, +2 weeks for web interface

---

## Next Steps

1. **Confirm Approval**: Review this plan, provide feedback
2. **Start Week 1**: Clean up codebase, setup testing
3. **Daily Updates**: Share progress, blockers, decisions
4. **Weekly Review**: Validate milestones, adjust as needed

Ready to start building?
