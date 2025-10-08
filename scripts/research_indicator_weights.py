#!/usr/bin/env python3
"""
Research script to determine optimal technical indicator weights and periods.

This script analyzes ETF price data to find time-invariant technical indicators
and their optimal weights for composite signal generation.

Based on academic research:
- Standard parameters vs optimized parameters
- Equal weighting vs optimal weighting
- Time-invariant performance analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# Dynamic project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


class TechnicalIndicatorResearcher:
    """Research optimal technical indicator parameters and weights."""

    def __init__(self, forward_return_days=21):
        """
        Initialize researcher.

        Args:
            forward_return_days: Days to look ahead for return prediction (default 21 = 1 month)
        """
        self.forward_return_days = forward_return_days
        self.results = {}

    def load_etf_data(self, ticker, min_days=500):
        """Load price data for a single ETF."""
        file_path = DATA_DIR / f"{ticker}.csv"
        if not file_path.exists():
            return None

        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Standardize column names
        df.columns = [col.capitalize() for col in df.columns]

        if len(df) < min_days:
            return None

        return df

    def calculate_indicators_standard(self, df):
        """Calculate technical indicators with STANDARD parameters."""
        df = df.copy()

        # Momentum Indicators
        df["RSI_14"] = ta.rsi(df["Close"], length=14)
        df["ROC_12"] = ta.roc(df["Close"], length=12)
        df["WILLR_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)

        # MACD
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd_cols = macd.columns.tolist()
            macd_col = (
                [
                    c
                    for c in macd_cols
                    if "MACD_" in c and "s_" not in c.lower() and "h_" not in c.lower()
                ][0]
                if any(
                    "MACD_" in c and "s_" not in c.lower() and "h_" not in c.lower()
                    for c in macd_cols
                )
                else None
            )
            signal_col = (
                [c for c in macd_cols if "MACDs" in c or "signal" in c.lower()][0]
                if any("MACDs" in c or "signal" in c.lower() for c in macd_cols)
                else None
            )
            hist_col = (
                [c for c in macd_cols if "MACDh" in c or "histogram" in c.lower()][0]
                if any("MACDh" in c or "histogram" in c.lower() for c in macd_cols)
                else None
            )

            if macd_col:
                df["MACD"] = macd[macd_col]
            if signal_col:
                df["MACD_signal"] = macd[signal_col]
            if hist_col:
                df["MACD_hist"] = macd[hist_col]

        # Trend Indicators
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["SMA_50"] = ta.sma(df["Close"], length=50)
        df["SMA_200"] = ta.sma(df["Close"], length=200)
        df["EMA_12"] = ta.ema(df["Close"], length=12)
        df["EMA_26"] = ta.ema(df["Close"], length=26)

        # Bollinger Bands
        bbands = ta.bbands(df["Close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            # Find columns dynamically (pandas-ta naming can vary)
            bb_cols = bbands.columns.tolist()
            upper_col = (
                [c for c in bb_cols if "BBU" in c][0]
                if any("BBU" in c for c in bb_cols)
                else None
            )
            middle_col = (
                [c for c in bb_cols if "BBM" in c][0]
                if any("BBM" in c for c in bb_cols)
                else None
            )
            lower_col = (
                [c for c in bb_cols if "BBL" in c][0]
                if any("BBL" in c for c in bb_cols)
                else None
            )

            if upper_col and middle_col and lower_col:
                df["BB_upper"] = bbands[upper_col]
                df["BB_middle"] = bbands[middle_col]
                df["BB_lower"] = bbands[lower_col]
                df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
                df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (
                    df["BB_upper"] - df["BB_lower"]
                )

        # Volume Indicators
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
        df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)

        # ADX
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        if adx is not None and not adx.empty:
            adx_cols = adx.columns.tolist()
            adx_col = (
                [
                    c
                    for c in adx_cols
                    if "ADX" in c and "DMP" not in c and "DMN" not in c
                ][0]
                if any(
                    "ADX" in c and "DMP" not in c and "DMN" not in c for c in adx_cols
                )
                else None
            )
            if adx_col:
                df["ADX"] = adx[adx_col]

        return df

    def calculate_indicators_alternative(self, df):
        """Calculate technical indicators with ALTERNATIVE shorter periods."""
        df = df.copy()

        # Momentum Indicators - shorter periods
        df["RSI_7"] = ta.rsi(df["Close"], length=7)
        df["RSI_21"] = ta.rsi(df["Close"], length=21)
        df["ROC_6"] = ta.roc(df["Close"], length=6)
        df["ROC_21"] = ta.roc(df["Close"], length=21)

        # MACD - alternative parameters
        macd_fast = ta.macd(df["Close"], fast=8, slow=17, signal=9)
        if macd_fast is not None and not macd_fast.empty:
            macd_cols = macd_fast.columns.tolist()
            macd_col = (
                [c for c in macd_cols if "MACD_" in c and "h" not in c.lower()][0]
                if any("MACD_" in c and "h" not in c.lower() for c in macd_cols)
                else None
            )
            hist_col = (
                [c for c in macd_cols if "MACDh" in c][0]
                if any("MACDh" in c for c in macd_cols)
                else None
            )

            if macd_col:
                df["MACD_fast"] = macd_fast[macd_col]
            if hist_col:
                df["MACD_fast_hist"] = macd_fast[hist_col]

        # Trend - multiple timeframes
        df["SMA_10"] = ta.sma(df["Close"], length=10)
        df["EMA_21"] = ta.ema(df["Close"], length=21)

        # Bollinger - tighter bands
        bbands_tight = ta.bbands(df["Close"], length=10, std=1.5)
        if bbands_tight is not None and not bbands_tight.empty:
            bb_cols = bbands_tight.columns.tolist()
            upper_col = (
                [c for c in bb_cols if "BBU" in c][0]
                if any("BBU" in c for c in bb_cols)
                else None
            )
            middle_col = (
                [c for c in bb_cols if "BBM" in c][0]
                if any("BBM" in c for c in bb_cols)
                else None
            )
            lower_col = (
                [c for c in bb_cols if "BBL" in c][0]
                if any("BBL" in c for c in bb_cols)
                else None
            )

            if upper_col and middle_col and lower_col:
                df["BB_tight_width"] = (
                    bbands_tight[upper_col] - bbands_tight[lower_col]
                ) / bbands_tight[middle_col]

        return df

    def create_normalized_signals(self, df):
        """
        Create normalized signals (0-100 scale) from indicators.

        Normalization approaches:
        - RSI: Already 0-100
        - MACD: Percentile rank
        - Moving averages: Price position relative to MA
        - Bollinger: %B already 0-1, scale to 0-100
        """
        signals = pd.DataFrame(index=df.index)

        # RSI - already 0-100
        if "RSI_14" in df.columns:
            signals["RSI_14_signal"] = df["RSI_14"]

        # MACD - convert to percentile rank (0-100)
        if "MACD_hist" in df.columns:
            signals["MACD_signal"] = (
                df["MACD_hist"].rolling(window=252).rank(pct=True) * 100
            )

        # Price vs SMA - momentum signal
        if "SMA_50" in df.columns:
            price_vs_sma = ((df["Close"] - df["SMA_50"]) / df["SMA_50"]) * 100
            # Convert to 0-100 scale using tanh normalization
            signals["SMA50_signal"] = (np.tanh(price_vs_sma / 10) + 1) * 50

        # Bollinger %B - already 0-1, scale to 0-100
        if "BB_pct" in df.columns:
            signals["BB_signal"] = df["BB_pct"] * 100

        # ROC - rate of change momentum
        if "ROC_12" in df.columns:
            roc_percentile = df["ROC_12"].rolling(window=252).rank(pct=True) * 100
            signals["ROC_signal"] = roc_percentile

        # CMF - volume-based
        if "CMF" in df.columns:
            # CMF ranges -1 to 1, scale to 0-100
            signals["CMF_signal"] = (df["CMF"] + 1) * 50

        # ADX - trend strength (already 0-100)
        if "ADX" in df.columns:
            signals["ADX_signal"] = df["ADX"]

        return signals

    def calculate_forward_returns(self, df, days):
        """Calculate forward returns."""
        df = df.copy()
        df["forward_return"] = (
            df["Close"].shift(-days) / df["Close"] - 1
        ) * 100  # percentage
        return df

    def evaluate_indicator_predictive_power(self, df, signals):
        """
        Evaluate each indicator's correlation with forward returns.

        Returns:
            Dictionary with correlation metrics for each indicator
        """
        df = self.calculate_forward_returns(df, self.forward_return_days)

        results = {}
        for col in signals.columns:
            valid_mask = ~(signals[col].isna() | df["forward_return"].isna())

            if valid_mask.sum() < 100:  # Need minimum data points
                continue

            signal_vals = signals.loc[valid_mask, col]
            forward_ret = df.loc[valid_mask, "forward_return"]

            # Calculate correlations
            pearson_corr, pearson_p = pearsonr(signal_vals, forward_ret)
            spearman_corr, spearman_p = spearmanr(signal_vals, forward_ret)

            # Quintile analysis - divide signals into 5 groups
            quintiles = pd.qcut(signal_vals, q=5, labels=False, duplicates="drop")
            quintile_returns = []
            for q in range(5):
                q_mask = quintiles == q
                if q_mask.sum() > 0:
                    quintile_returns.append(forward_ret[q_mask].mean())

            # Information coefficient (IC) - average correlation
            results[col] = {
                "pearson_corr": pearson_corr,
                "pearson_pvalue": pearson_p,
                "spearman_corr": spearman_corr,
                "spearman_pvalue": spearman_p,
                "quintile_returns": quintile_returns,
                "quintile_spread": (
                    quintile_returns[-1] - quintile_returns[0]
                    if len(quintile_returns) == 5
                    else 0
                ),
                "n_observations": valid_mask.sum(),
            }

        return results

    def test_time_invariance(self, df, signals, n_periods=4):
        """
        Test if indicator performance is consistent across time periods.

        Splits data into n_periods and checks correlation stability.
        """
        df = self.calculate_forward_returns(df, self.forward_return_days)
        period_length = len(df) // n_periods

        invariance_results = {}

        for col in signals.columns:
            period_corrs = []

            for i in range(n_periods):
                start_idx = i * period_length
                end_idx = (i + 1) * period_length if i < n_periods - 1 else len(df)

                period_signals = signals.iloc[start_idx:end_idx][col]
                period_returns = df.iloc[start_idx:end_idx]["forward_return"]

                valid_mask = ~(period_signals.isna() | period_returns.isna())
                if valid_mask.sum() < 50:
                    continue

                corr, _ = spearmanr(
                    period_signals[valid_mask], period_returns[valid_mask]
                )
                period_corrs.append(corr)

            if len(period_corrs) >= 3:
                invariance_results[col] = {
                    "period_correlations": period_corrs,
                    "mean_correlation": np.mean(period_corrs),
                    "std_correlation": np.std(period_corrs),
                    "coefficient_of_variation": (
                        np.std(period_corrs) / abs(np.mean(period_corrs))
                        if np.mean(period_corrs) != 0
                        else np.inf
                    ),
                    "min_correlation": min(period_corrs),
                    "max_correlation": max(period_corrs),
                }

        return invariance_results

    def optimize_composite_weights(self, df, signals):
        """
        Find optimal weights for composite signal using regression.

        Compares:
        1. Equal weighting
        2. Correlation-based weighting
        3. Ridge regression optimization
        """
        df = self.calculate_forward_returns(df, self.forward_return_days)

        # Prepare data
        valid_mask = ~df["forward_return"].isna()
        for col in signals.columns:
            valid_mask &= ~signals[col].isna()

        if valid_mask.sum() < 100:
            return None

        X = signals.loc[valid_mask].values
        y = df.loc[valid_mask, "forward_return"].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Method 1: Equal weights
        equal_weights = np.ones(len(signals.columns)) / len(signals.columns)
        composite_equal = X_scaled @ equal_weights
        corr_equal, _ = spearmanr(composite_equal, y)

        # Method 2: Ridge regression (L2 regularization for stability)
        ridge_alphas = [0.1, 1.0, 10.0, 100.0]
        best_alpha = None
        best_score = -np.inf
        best_weights = None

        for alpha in ridge_alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_scaled, y)
            score = ridge.score(X_scaled, y)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_weights = ridge.coef_

        # Normalize weights to sum to 1
        ridge_weights = np.abs(best_weights) / np.sum(np.abs(best_weights))
        composite_ridge = X_scaled @ ridge_weights
        corr_ridge, _ = spearmanr(composite_ridge, y)

        # Method 3: Correlation-based (weight by correlation strength)
        correlations = []
        for col in signals.columns:
            corr, _ = spearmanr(signals.loc[valid_mask, col], y)
            correlations.append(abs(corr))

        corr_weights = np.array(correlations) / np.sum(correlations)
        composite_corr = X_scaled @ corr_weights
        corr_corr_weighted, _ = spearmanr(composite_corr, y)

        return {
            "equal_weights": dict(zip(signals.columns, equal_weights)),
            "equal_weight_ic": corr_equal,
            "ridge_weights": dict(zip(signals.columns, ridge_weights)),
            "ridge_weight_ic": corr_ridge,
            "ridge_alpha": best_alpha,
            "corr_weights": dict(zip(signals.columns, corr_weights)),
            "corr_weight_ic": corr_corr_weighted,
        }

    def analyze_single_etf(self, ticker):
        """Run complete analysis on a single ETF."""
        print(f"Analyzing {ticker}...")

        df = self.load_etf_data(ticker)
        if df is None:
            print(f"  Skipping {ticker} - insufficient data")
            return None

        # Calculate indicators
        df = self.calculate_indicators_standard(df)
        df = self.calculate_indicators_alternative(df)

        # Create normalized signals
        signals = self.create_normalized_signals(df)

        # Evaluate predictive power
        predictive_power = self.evaluate_indicator_predictive_power(df, signals)

        # Test time invariance
        time_invariance = self.test_time_invariance(df, signals)

        # Optimize weights
        optimal_weights = self.optimize_composite_weights(df, signals)

        return {
            "ticker": ticker,
            "predictive_power": predictive_power,
            "time_invariance": time_invariance,
            "optimal_weights": optimal_weights,
        }

    def analyze_universe(self, max_etfs=50):
        """Analyze multiple ETFs and aggregate results."""
        etf_files = list(DATA_DIR.glob("*.csv"))
        etf_files = sorted(etf_files)[:max_etfs]  # Limit for faster analysis

        all_results = []

        for etf_file in etf_files:
            ticker = etf_file.stem
            result = self.analyze_single_etf(ticker)
            if result:
                all_results.append(result)

        return all_results

    def aggregate_results(self, all_results):
        """Aggregate results across all ETFs to find robust indicators."""
        # Aggregate predictive power
        indicator_performance = {}
        for result in all_results:
            for indicator, metrics in result["predictive_power"].items():
                if indicator not in indicator_performance:
                    indicator_performance[indicator] = {
                        "spearman_corrs": [],
                        "quintile_spreads": [],
                    }
                indicator_performance[indicator]["spearman_corrs"].append(
                    metrics["spearman_corr"]
                )
                indicator_performance[indicator]["quintile_spreads"].append(
                    metrics["quintile_spread"]
                )

        # Calculate average performance
        indicator_summary = {}
        for indicator, data in indicator_performance.items():
            indicator_summary[indicator] = {
                "mean_spearman": np.mean(data["spearman_corrs"]),
                "median_spearman": np.median(data["spearman_corrs"]),
                "std_spearman": np.std(data["spearman_corrs"]),
                "mean_quintile_spread": np.mean(data["quintile_spreads"]),
                "n_etfs": len(data["spearman_corrs"]),
            }

        # Aggregate time invariance
        invariance_summary = {}
        for result in all_results:
            for indicator, metrics in result["time_invariance"].items():
                if indicator not in invariance_summary:
                    invariance_summary[indicator] = {"cv_values": []}
                invariance_summary[indicator]["cv_values"].append(
                    metrics["coefficient_of_variation"]
                )

        for indicator, data in invariance_summary.items():
            invariance_summary[indicator]["mean_cv"] = np.mean(data["cv_values"])
            invariance_summary[indicator]["median_cv"] = np.median(data["cv_values"])

        # Aggregate optimal weights
        weight_methods = ["equal_weights", "ridge_weights", "corr_weights"]
        aggregated_weights = {method: {} for method in weight_methods}
        weight_ics = {method: [] for method in weight_methods}

        for result in all_results:
            if result["optimal_weights"]:
                for method in weight_methods:
                    weights = result["optimal_weights"][method]
                    for indicator, weight in weights.items():
                        if indicator not in aggregated_weights[method]:
                            aggregated_weights[method][indicator] = []
                        aggregated_weights[method][indicator].append(weight)

                    # Track IC performance
                    ic_key = f"{method.replace('weights', 'weight_ic')}"
                    weight_ics[method].append(result["optimal_weights"][ic_key])

        # Calculate mean weights
        mean_weights = {}
        for method in weight_methods:
            mean_weights[method] = {}
            for indicator, weights in aggregated_weights[method].items():
                mean_weights[method][indicator] = {
                    "mean": np.mean(weights),
                    "median": np.median(weights),
                    "std": np.std(weights),
                }
            mean_weights[method]["mean_ic"] = np.mean(weight_ics[method])

        return {
            "indicator_performance": indicator_summary,
            "time_invariance": invariance_summary,
            "optimal_weights": mean_weights,
        }

    def generate_recommendations(self, aggregated_results):
        """Generate final recommendations based on analysis."""
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {},
            "recommended_indicators": [],
            "recommended_weights": {},
            "reasoning": [],
        }

        # Find best performing indicators
        perf = aggregated_results["indicator_performance"]
        inv = aggregated_results["time_invariance"]

        ranked_indicators = sorted(
            perf.items(), key=lambda x: x[1]["mean_spearman"], reverse=True
        )

        # Select top indicators with good time invariance
        for indicator, metrics in ranked_indicators:
            if metrics["mean_spearman"] > 0.05:  # Positive predictive power
                if indicator in inv:
                    cv = inv[indicator]["mean_cv"]
                    if cv < 2.0:  # Reasonably stable across time
                        recommendations["recommended_indicators"].append(
                            {
                                "name": indicator,
                                "mean_ic": metrics["mean_spearman"],
                                "stability_cv": cv,
                                "quintile_spread": metrics["mean_quintile_spread"],
                            }
                        )

        # Compare weighting methods
        opt_weights = aggregated_results["optimal_weights"]
        best_method = max(opt_weights.items(), key=lambda x: x[1].get("mean_ic", 0))[0]

        recommendations["recommended_method"] = best_method
        recommendations["recommended_weights"] = opt_weights[best_method]

        # Generate reasoning
        recommendations["reasoning"].append(
            f"Analysis conducted on {len(perf)} indicators across multiple ETFs"
        )
        recommendations["reasoning"].append(
            f"Best weighting method: {best_method} with mean IC: {opt_weights[best_method]['mean_ic']:.4f}"
        )
        recommendations["reasoning"].append(
            f"Selected {len(recommendations['recommended_indicators'])} indicators with positive IC and low time-variance"
        )

        return recommendations


def main():
    """Run the analysis and generate report."""
    print("=" * 80)
    print("TECHNICAL INDICATOR WEIGHT RESEARCH")
    print("=" * 80)
    print()

    # Initialize researcher
    researcher = TechnicalIndicatorResearcher(forward_return_days=21)

    # Analyze universe
    print("Analyzing ETF universe...")
    print("(Limited to 50 ETFs for faster analysis)")
    print()

    all_results = researcher.analyze_universe(max_etfs=50)

    print(f"\nSuccessfully analyzed {len(all_results)} ETFs")
    print()

    # Aggregate results
    print("Aggregating results across ETFs...")
    aggregated = researcher.aggregate_results(all_results)

    # Generate recommendations
    print("Generating recommendations...")
    recommendations = researcher.generate_recommendations(aggregated)

    # Save results
    results_file = RESULTS_DIR / "indicator_weight_analysis.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "aggregated_results": aggregated,
                "recommendations": recommendations,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to: {results_file}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    print()
    print(f"Recommended Weighting Method: {recommendations['recommended_method']}")
    print(
        f"Mean Information Coefficient: {recommendations['recommended_weights']['mean_ic']:.4f}"
    )
    print()
    print("Top Indicators by Predictive Power:")
    print()

    for i, ind in enumerate(recommendations["recommended_indicators"][:10], 1):
        print(
            f"{i:2d}. {ind['name']:20s} - IC: {ind['mean_ic']:7.4f}, "
            f"Stability CV: {ind['stability_cv']:6.3f}"
        )

    print()
    print("=" * 80)

    return recommendations


if __name__ == "__main__":
    main()
