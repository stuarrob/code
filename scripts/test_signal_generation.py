#!/usr/bin/env python3
"""
Test script to validate signal generation modules.
"""

import sys
from pathlib import Path
import pandas as pd

# Dynamic project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.indicators import TechnicalIndicators
from src.signals.signal_scorer import SignalScorer
from src.signals.composite_signal import CompositeSignalGenerator

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "prices"


def load_etf_data(ticker):
    """Load price data for testing."""
    file_path = DATA_DIR / f"{ticker}.csv"
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.columns = [col.capitalize() for col in df.columns]

    return df


def test_indicators():
    """Test technical indicator calculation."""
    print("Testing Technical Indicators...")

    # Load sample ETF
    df = load_etf_data("SPY")
    if df is None:
        print("  ❌ Failed to load SPY data")
        return False

    # Calculate indicators
    df_with_ind = TechnicalIndicators.calculate_all_standard(df)

    # Check for expected indicators
    expected_indicators = [
        "RSI_14",
        "MACD",
        "MACD_hist",
        "BB_upper",
        "BB_lower",
        "BB_pct",
        "SMA_20",
        "SMA_50",
        "ADX",
        "CMF",
    ]

    missing = [ind for ind in expected_indicators if ind not in df_with_ind.columns]

    if missing:
        print(f"  ❌ Missing indicators: {missing}")
        return False

    # Check for valid values (not all NaN)
    for ind in expected_indicators:
        valid_pct = df_with_ind[ind].notna().sum() / len(df_with_ind)
        if valid_pct < 0.5:
            print(f"  ❌ {ind} has only {valid_pct*100:.1f}% valid values")
            return False

    print(f"  ✅ All {len(expected_indicators)} indicators calculated successfully")
    print(f"  ✅ {len(df_with_ind)} days of data processed")
    return True


def test_signal_scorer():
    """Test signal normalization."""
    print("\nTesting Signal Scorer...")

    # Load and calculate indicators
    df = load_etf_data("SPY")
    df_with_ind = TechnicalIndicators.calculate_all_standard(df)

    # Create signals
    signals = SignalScorer.create_signals_from_indicators(df_with_ind)

    expected_signals = [
        "RSI_signal",
        "BB_signal",
        "ROC_signal",
        "ADX_signal",
        "CMF_signal",
    ]

    missing = [sig for sig in expected_signals if sig not in signals.columns]

    if missing:
        print(f"  ❌ Missing signals: {missing}")
        return False

    # Check signals are in 0-100 range
    for sig in expected_signals:
        valid_vals = signals[sig].dropna()
        if len(valid_vals) == 0:
            print(f"  ❌ {sig} has no valid values")
            return False

        if valid_vals.min() < 0 or valid_vals.max() > 100:
            print(
                f"  ❌ {sig} out of range: [{valid_vals.min():.2f}, {valid_vals.max():.2f}]"
            )
            return False

    print(f"  ✅ All {len(expected_signals)} signals normalized correctly")
    print(f"  ✅ Signals in 0-100 range")
    return True


def test_composite_signal():
    """Test composite signal generation."""
    print("\nTesting Composite Signal Generator...")

    # Load data
    df = load_etf_data("SPY")

    # Create generator with default config
    generator = CompositeSignalGenerator()

    # Generate signals
    result = generator.generate_signals_for_etf(df, multi_timeframe=True)

    # Check for composite score
    if "composite_score" not in result.columns:
        print("  ❌ composite_score not found")
        return False

    # Check for multi-timeframe scores
    for window in [21, 63, 126]:
        col = f"score_{window}d"
        if col not in result.columns:
            print(f"  ❌ {col} not found")
            return False

    # Check composite score range
    valid_scores = result["composite_score"].dropna()
    if len(valid_scores) == 0:
        print("  ❌ No valid composite scores")
        return False

    if valid_scores.min() < 0 or valid_scores.max() > 100:
        print(
            f"  ❌ Composite score out of range: [{valid_scores.min():.2f}, {valid_scores.max():.2f}]"
        )
        return False

    latest_score = result["composite_score"].iloc[-1]
    print(f"  ✅ Composite signal generated successfully")
    print(f"  ✅ Latest SPY composite score: {latest_score:.2f}")

    return True


def test_multi_etf_scoring():
    """Test scoring multiple ETFs."""
    print("\nTesting Multi-ETF Scoring...")

    # Load multiple ETFs
    tickers = ["SPY", "QQQ", "IWM", "AGG", "GLD"]
    etf_data = {}

    for ticker in tickers:
        df = load_etf_data(ticker)
        if df is not None:
            etf_data[ticker] = df

    if len(etf_data) == 0:
        print("  ❌ No ETF data loaded")
        return False

    # Generate scores
    generator = CompositeSignalGenerator()
    scores_df = generator.generate_latest_scores(etf_data, multi_timeframe=True)

    # Check results
    if len(scores_df) != len(etf_data):
        print(
            f"  ❌ Expected {len(etf_data)} results, got {len(scores_df)}"
        )
        return False

    # Check columns
    required_cols = [
        "ticker",
        "composite_score",
        "RSI_signal",
        "BB_signal",
        "ADX_signal",
    ]
    missing = [col for col in required_cols if col not in scores_df.columns]

    if missing:
        print(f"  ❌ Missing columns: {missing}")
        return False

    # Rank ETFs
    ranked = generator.rank_etfs(scores_df)

    print(f"  ✅ Scored {len(etf_data)} ETFs successfully")
    print(f"\n  Top 3 ETFs by composite score:")
    for i, row in ranked.head(3).iterrows():
        print(
            f"    {i+1}. {row['ticker']:6s} - Score: {row['composite_score']:6.2f}, Rank: {row['rank']:6.2f}%ile"
        )

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("SIGNAL GENERATION MODULE TESTS")
    print("=" * 80)
    print()

    tests = [
        ("Indicators", test_indicators),
        ("Signal Scorer", test_signal_scorer),
        ("Composite Signal", test_composite_signal),
        ("Multi-ETF Scoring", test_multi_etf_scoring),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            results.append((name, False))

    # Summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status:10s} - {name}")

    print()
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
