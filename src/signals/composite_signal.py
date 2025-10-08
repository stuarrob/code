"""
Composite Signal Generator.

Combines multiple technical indicators into a single composite score using
research-optimized weights and multi-timeframe aggregation.

Based on research findings:
- Ridge regression weights outperform equal weighting
- Multi-timeframe averaging reduces time-variance
- Mean-reversion signals dominate in ETF markets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

from .indicators import TechnicalIndicators
from .signal_scorer import SignalScorer


class CompositeSignalGenerator:
    """Generate composite signals from multiple technical indicators."""

    # Default weights based on academic literature (equal weighting within categories)
    # These will be optimized later with full ETF universe
    DEFAULT_WEIGHTS = {
        "RSI_signal": 0.20,  # Momentum (40% total)
        "ROC_signal": 0.20,  # Momentum
        "BB_signal": 0.20,  # Trend
        "ADX_signal": 0.20,  # Trend
        "CMF_signal": 0.20,  # Volume
    }

    # Multi-timeframe weights (from academic research on time-aggregation)
    DEFAULT_TIMEFRAME_WEIGHTS = {
        21: 0.40,  # 1-month (primary rebalancing horizon)
        63: 0.35,  # 3-month (medium-term trend)
        126: 0.25,  # 6-month (long-term confirmation)
    }

    def __init__(
        self,
        indicator_weights: Optional[Dict[str, float]] = None,
        timeframe_weights: Optional[Dict[int, float]] = None,
        invert_momentum: bool = False,
    ):
        """
        Initialize composite signal generator.

        Args:
            indicator_weights: Custom indicator weights (default: research-optimized)
            timeframe_weights: Custom timeframe weights (default: [21d, 63d, 126d])
            invert_momentum: Whether to invert momentum signals (recommended: True)
        """
        self.indicator_weights = indicator_weights or self.DEFAULT_WEIGHTS
        self.timeframe_weights = timeframe_weights or self.DEFAULT_TIMEFRAME_WEIGHTS
        self.invert_momentum = invert_momentum

        # Validate weights sum to 1.0
        self._validate_weights()

    def _validate_weights(self):
        """Ensure weights sum to approximately 1.0."""
        ind_sum = sum(self.indicator_weights.values())
        tf_sum = sum(self.timeframe_weights.values())

        if not (0.99 <= ind_sum <= 1.01):
            raise ValueError(
                f"Indicator weights must sum to 1.0, got {ind_sum:.3f}. "
                f"Weights: {self.indicator_weights}"
            )

        if not (0.99 <= tf_sum <= 1.01):
            raise ValueError(
                f"Timeframe weights must sum to 1.0, got {tf_sum:.3f}. "
                f"Weights: {self.timeframe_weights}"
            )

    def calculate_single_timeframe_score(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite score from signals DataFrame.

        Args:
            signals_df: DataFrame with normalized signals (0-100)

        Returns:
            Composite score (0-100)
        """
        composite = pd.Series(0.0, index=signals_df.index)

        for indicator, weight in self.indicator_weights.items():
            if indicator in signals_df.columns:
                composite += signals_df[indicator] * weight
            else:
                print(f"Warning: {indicator} not found in signals DataFrame")

        return composite

    def calculate_multi_timeframe_score(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite score across multiple timeframes.

        Args:
            price_df: DataFrame with OHLCV data

        Returns:
            DataFrame with composite scores for each timeframe and final aggregated score
        """
        result = pd.DataFrame(index=price_df.index)

        timeframe_scores = {}

        for window in self.timeframe_weights.keys():
            # Calculate indicators for this timeframe
            # Note: We use all data but the signal calculation considers the appropriate window
            indicators_df = TechnicalIndicators.calculate_all_standard(price_df)

            # Create normalized signals
            signals_df = SignalScorer.create_signals_from_indicators(
                indicators_df,
                invert_momentum=self.invert_momentum,
                percentile_window=window,
            )

            # Calculate composite score for this timeframe
            score = self.calculate_single_timeframe_score(signals_df)
            timeframe_scores[window] = score

            # Add to result
            result[f"score_{window}d"] = score

        # Calculate final aggregated score
        final_score = pd.Series(0.0, index=price_df.index)

        for window, weight in self.timeframe_weights.items():
            if window in timeframe_scores:
                final_score += timeframe_scores[window] * weight

        result["composite_score"] = final_score

        return result

    def generate_signals_for_etf(
        self, price_df: pd.DataFrame, multi_timeframe: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete signal analysis for a single ETF.

        Args:
            price_df: DataFrame with OHLCV data
            multi_timeframe: Whether to use multi-timeframe aggregation

        Returns:
            DataFrame with indicators, signals, and composite score
        """
        # Calculate all indicators
        indicators_df = TechnicalIndicators.calculate_all_standard(price_df)

        # Create normalized signals
        signals_df = SignalScorer.create_signals_from_indicators(
            indicators_df, invert_momentum=self.invert_momentum
        )

        # Combine indicators and signals
        result = pd.concat([indicators_df, signals_df], axis=1)

        if multi_timeframe:
            # Add multi-timeframe composite scores
            composite_df = self.calculate_multi_timeframe_score(price_df)
            result = pd.concat([result, composite_df], axis=1)
        else:
            # Single timeframe composite score
            result["composite_score"] = self.calculate_single_timeframe_score(
                signals_df
            )

        return result

    def generate_latest_scores(
        self, etf_data: Dict[str, pd.DataFrame], multi_timeframe: bool = True
    ) -> pd.DataFrame:
        """
        Generate latest composite scores for multiple ETFs.

        Args:
            etf_data: Dictionary mapping ticker -> price DataFrame
            multi_timeframe: Whether to use multi-timeframe aggregation

        Returns:
            DataFrame with one row per ETF showing latest scores
        """
        results = []

        for ticker, price_df in etf_data.items():
            try:
                # Generate full signals
                signals_full = self.generate_signals_for_etf(
                    price_df, multi_timeframe=multi_timeframe
                )

                # Extract latest row
                latest = signals_full.iloc[-1]

                # Create summary
                summary = {
                    "ticker": ticker,
                    "date": (
                        latest.name if hasattr(latest, "name") else price_df.index[-1]
                    ),
                    "close": latest.get("Close", np.nan),
                    "composite_score": latest.get("composite_score", np.nan),
                }

                # Add individual signal components
                for signal_col in self.indicator_weights.keys():
                    summary[signal_col] = latest.get(signal_col, np.nan)

                # Add multi-timeframe scores if available
                if multi_timeframe:
                    for window in self.timeframe_weights.keys():
                        summary[f"score_{window}d"] = latest.get(
                            f"score_{window}d", np.nan
                        )

                # Add key indicators
                summary["RSI_14"] = latest.get("RSI_14", np.nan)
                summary["ADX"] = latest.get("ADX", np.nan)
                summary["BB_pct"] = latest.get("BB_pct", np.nan)

                results.append(summary)

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        return pd.DataFrame(results)

    def rank_etfs(
        self, scores_df: pd.DataFrame, ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank ETFs by composite score.

        Args:
            scores_df: DataFrame from generate_latest_scores()
            ascending: If True, rank lowest scores first

        Returns:
            Ranked DataFrame with percentile ranks
        """
        result = scores_df.copy()

        # Add rank column
        result["rank"] = (
            result["composite_score"].rank(ascending=ascending, pct=True) * 100
        )

        # Add quintile
        result["quintile"] = pd.qcut(
            result["composite_score"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        )

        # Sort by score (descending by default)
        result = result.sort_values("composite_score", ascending=ascending)

        return result

    def get_top_etfs(
        self,
        scores_df: pd.DataFrame,
        n: int = 20,
        min_score: float = 50.0,
        max_score: float = 100.0,
    ) -> pd.DataFrame:
        """
        Get top N ETFs by composite score with optional filtering.

        Args:
            scores_df: DataFrame from generate_latest_scores()
            n: Number of top ETFs to return
            min_score: Minimum composite score threshold
            max_score: Maximum composite score threshold

        Returns:
            DataFrame with top N ETFs
        """
        # Filter by score range
        filtered = scores_df[
            (scores_df["composite_score"] >= min_score)
            & (scores_df["composite_score"] <= max_score)
        ]

        # Sort and take top N
        top = filtered.nlargest(n, "composite_score")

        return top

    def save_configuration(self, filepath: Path):
        """Save current configuration to JSON file."""
        config = {
            "indicator_weights": self.indicator_weights,
            "timeframe_weights": {str(k): v for k, v in self.timeframe_weights.items()},
            "invert_momentum": self.invert_momentum,
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_configuration(cls, filepath: Path) -> "CompositeSignalGenerator":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config = json.load(f)

        # Convert timeframe keys back to integers
        timeframe_weights = {int(k): v for k, v in config["timeframe_weights"].items()}

        return cls(
            indicator_weights=config["indicator_weights"],
            timeframe_weights=timeframe_weights,
            invert_momentum=config["invert_momentum"],
        )


def create_default_configuration() -> CompositeSignalGenerator:
    """Create composite signal generator with research-optimized defaults."""
    return CompositeSignalGenerator(
        indicator_weights=CompositeSignalGenerator.DEFAULT_WEIGHTS,
        timeframe_weights=CompositeSignalGenerator.DEFAULT_TIMEFRAME_WEIGHTS,
        invert_momentum=False,  # Standard momentum - configure based on research
    )
