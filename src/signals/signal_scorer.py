"""
Signal Scorer - Normalize and transform technical indicators into 0-100 signals.

Provides configurable signal transformation including optional inversion for
mean-reversion strategies. Default behavior follows standard momentum interpretation.
"""

import pandas as pd
import numpy as np
from typing import Optional


class SignalScorer:
    """Transform raw technical indicators into normalized 0-100 signals."""

    @staticmethod
    def normalize_rsi(rsi: pd.Series, invert: bool = False) -> pd.Series:
        """
        Normalize RSI to 0-100 signal.

        Args:
            rsi: RSI values (already 0-100)
            invert: If True, invert signal (low RSI = bullish mean-reversion)

        Returns:
            Normalized signal (0-100)
        """
        if invert:
            # Low RSI (oversold) = high signal (bullish)
            return 100 - rsi
        return rsi

    @staticmethod
    def normalize_macd(
        macd_hist: pd.Series, window: int = 252, invert: bool = False
    ) -> pd.Series:
        """
        Normalize MACD histogram to 0-100 signal using percentile rank.

        Args:
            macd_hist: MACD histogram values
            window: Rolling window for percentile calculation
            invert: If True, invert signal

        Returns:
            Normalized signal (0-100)
        """
        # Percentile rank within rolling window
        signal = (
            macd_hist.rolling(window=window, min_periods=window // 2).rank(pct=True)
            * 100
        )

        if invert:
            signal = 100 - signal

        return signal

    @staticmethod
    def normalize_bollinger_pct(bb_pct: pd.Series, invert: bool = False) -> pd.Series:
        """
        Normalize Bollinger %B to 0-100 signal.

        Args:
            bb_pct: Bollinger %B (0-1 range, but can exceed)
            invert: If True, invert signal (low %B = bullish)

        Returns:
            Normalized signal (0-100)
        """
        # Clip to 0-1 range and scale to 0-100
        signal = bb_pct.clip(0, 1) * 100

        if invert:
            # Low %B (near lower band) = high signal (bullish)
            signal = 100 - signal

        return signal

    @staticmethod
    def normalize_roc(
        roc: pd.Series, window: int = 252, invert: bool = False
    ) -> pd.Series:
        """
        Normalize Rate of Change to 0-100 signal using percentile rank.

        Args:
            roc: ROC percentage values
            window: Rolling window for percentile calculation
            invert: If True, invert signal

        Returns:
            Normalized signal (0-100)
        """
        signal = (
            roc.rolling(window=window, min_periods=window // 2).rank(pct=True) * 100
        )

        if invert:
            signal = 100 - signal

        return signal

    @staticmethod
    def normalize_adx(adx: pd.Series, invert: bool = False) -> pd.Series:
        """
        Normalize ADX to 0-100 signal.

        Args:
            adx: ADX values (already 0-100)
            invert: If True, invert signal (typically False for ADX)

        Returns:
            Normalized signal (0-100)
        """
        # ADX is already 0-100 and shows trend strength
        # High ADX = strong trend (typically bullish for momentum)
        if invert:
            return 100 - adx
        return adx

    @staticmethod
    def normalize_cmf(cmf: pd.Series, invert: bool = False) -> pd.Series:
        """
        Normalize Chaikin Money Flow to 0-100 signal.

        Args:
            cmf: CMF values (-1 to +1 range)
            invert: If True, invert signal

        Returns:
            Normalized signal (0-100)
        """
        # Scale from [-1, 1] to [0, 100]
        signal = (cmf.clip(-1, 1) + 1) * 50

        if invert:
            signal = 100 - signal

        return signal

    @staticmethod
    def normalize_sma_cross(
        close: pd.Series, sma: pd.Series, invert: bool = False
    ) -> pd.Series:
        """
        Normalize price vs SMA to 0-100 signal.

        Args:
            close: Close price series
            sma: SMA series
            invert: If True, invert signal

        Returns:
            Normalized signal (0-100)
        """
        # Calculate percentage deviation from SMA
        pct_deviation = ((close - sma) / sma) * 100

        # Use tanh normalization to convert to 0-100 range
        # tanh compresses infinite range to [-1, 1]
        signal = (np.tanh(pct_deviation / 10) + 1) * 50

        if invert:
            signal = 100 - signal

        return signal

    @classmethod
    def create_signals_from_indicators(
        cls,
        df: pd.DataFrame,
        invert_momentum: bool = True,
        percentile_window: int = 252,
    ) -> pd.DataFrame:
        """
        Create normalized signals from technical indicators DataFrame.

        Args:
            df: DataFrame with technical indicators (from TechnicalIndicators.calculate_all_standard)
            invert_momentum: Whether to invert momentum indicators (recommended: True)
            percentile_window: Window for percentile-based normalization

        Returns:
            DataFrame with normalized signals (0-100 scale)
        """
        signals = pd.DataFrame(index=df.index)

        # Momentum signals (INVERTED for mean-reversion)
        if "RSI_14" in df.columns:
            signals["RSI_signal"] = cls.normalize_rsi(
                df["RSI_14"], invert=invert_momentum
            )

        if "BB_pct" in df.columns:
            signals["BB_signal"] = cls.normalize_bollinger_pct(
                df["BB_pct"], invert=invert_momentum
            )

        if "ROC_12" in df.columns:
            signals["ROC_signal"] = cls.normalize_roc(
                df["ROC_12"], window=percentile_window, invert=invert_momentum
            )

        if "MACD_hist" in df.columns:
            signals["MACD_signal"] = cls.normalize_macd(
                df["MACD_hist"], window=percentile_window, invert=invert_momentum
            )

        # Trend strength signal (NOT inverted - high ADX is good)
        if "ADX" in df.columns:
            signals["ADX_signal"] = cls.normalize_adx(df["ADX"], invert=False)

        # Volume signal (INVERTED)
        if "CMF" in df.columns:
            signals["CMF_signal"] = cls.normalize_cmf(df["CMF"], invert=invert_momentum)

        # Trend direction signals
        if "SMA_50" in df.columns and "Close" in df.columns:
            signals["SMA50_signal"] = cls.normalize_sma_cross(
                df["Close"], df["SMA_50"], invert=invert_momentum
            )

        if "SMA_200" in df.columns and "Close" in df.columns:
            signals["SMA200_signal"] = cls.normalize_sma_cross(
                df["Close"], df["SMA_200"], invert=invert_momentum
            )

        # Stochastic (momentum oscillator - INVERTED)
        if "STOCH_k" in df.columns:
            signals["STOCH_signal"] = cls.normalize_rsi(
                df["STOCH_k"], invert=invert_momentum
            )  # Stoch is also 0-100

        return signals

    @staticmethod
    def calculate_quintile_scores(
        signal: pd.Series, ascending: bool = False
    ) -> pd.Series:
        """
        Convert continuous signal to quintile scores (1-5).

        Args:
            signal: Continuous signal values
            ascending: If True, lower values get lower quintiles

        Returns:
            Quintile scores (1-5)
        """
        return pd.qcut(signal, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(
            float
        )

    @staticmethod
    def calculate_z_score(signal: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling z-score of signal.

        Args:
            signal: Signal values
            window: Rolling window for mean/std calculation

        Returns:
            Z-score normalized signal
        """
        rolling_mean = signal.rolling(window=window, min_periods=window // 2).mean()
        rolling_std = signal.rolling(window=window, min_periods=window // 2).std()

        z_score = (signal - rolling_mean) / rolling_std
        return z_score

    @classmethod
    def add_signal_ranks(cls, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add percentile rank versions of all signals.

        Args:
            signals_df: DataFrame with normalized signals

        Returns:
            DataFrame with additional '_rank' columns
        """
        result = signals_df.copy()

        for col in signals_df.columns:
            if "_signal" in col:
                rank_col = col.replace("_signal", "_rank")
                result[rank_col] = signals_df[col].rank(pct=True) * 100

        return result
