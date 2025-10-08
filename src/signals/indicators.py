"""
Technical Indicators Library.

Implements standard technical indicators using pandas-ta with robust error handling.
Based on research findings showing mean-reversion behavior in ETF markets.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, Dict
import warnings

warnings.filterwarnings("ignore")


class TechnicalIndicators:
    """Calculate technical indicators for ETF price data."""

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            close: Close price series
            period: Lookback period (default: 14)

        Returns:
            RSI values (0-100)
        """
        rsi = ta.rsi(close, length=period)
        return rsi if rsi is not None else pd.Series(np.nan, index=close.index)

    @staticmethod
    def calculate_macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            close: Close price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Dictionary with 'macd', 'signal', 'histogram' keys
        """
        macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)

        if macd_df is None or macd_df.empty:
            nan_series = pd.Series(np.nan, index=close.index)
            return {"macd": nan_series, "signal": nan_series, "histogram": nan_series}

        # Dynamically find columns
        cols = macd_df.columns.tolist()
        macd_col = next(
            (
                c
                for c in cols
                if "MACD_" in c and "s_" not in c.lower() and "h_" not in c.lower()
            ),
            None,
        )
        signal_col = next(
            (c for c in cols if "MACDs" in c or "signal" in c.lower()), None
        )
        hist_col = next(
            (c for c in cols if "MACDh" in c or "histogram" in c.lower()), None
        )

        return {
            "macd": (
                macd_df[macd_col] if macd_col else pd.Series(np.nan, index=close.index)
            ),
            "signal": (
                macd_df[signal_col]
                if signal_col
                else pd.Series(np.nan, index=close.index)
            ),
            "histogram": (
                macd_df[hist_col] if hist_col else pd.Series(np.nan, index=close.index)
            ),
        }

    @staticmethod
    def calculate_bollinger_bands(
        close: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            close: Close price series
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)

        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'width', 'percent_b' keys
        """
        bb_df = ta.bbands(close, length=period, std=std_dev)

        if bb_df is None or bb_df.empty:
            nan_series = pd.Series(np.nan, index=close.index)
            return {
                "upper": nan_series,
                "middle": nan_series,
                "lower": nan_series,
                "width": nan_series,
                "percent_b": nan_series,
            }

        # Dynamically find columns
        cols = bb_df.columns.tolist()
        upper_col = next((c for c in cols if "BBU" in c), None)
        middle_col = next((c for c in cols if "BBM" in c), None)
        lower_col = next((c for c in cols if "BBL" in c), None)

        if not all([upper_col, middle_col, lower_col]):
            nan_series = pd.Series(np.nan, index=close.index)
            return {
                "upper": nan_series,
                "middle": nan_series,
                "lower": nan_series,
                "width": nan_series,
                "percent_b": nan_series,
            }

        upper = bb_df[upper_col]
        middle = bb_df[middle_col]
        lower = bb_df[lower_col]

        # Calculate derived metrics
        width = (upper - lower) / middle
        percent_b = (close - lower) / (upper - lower)

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "width": width,
            "percent_b": percent_b,
        }

    @staticmethod
    def calculate_sma(close: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            close: Close price series
            period: Lookback period

        Returns:
            SMA values
        """
        sma = ta.sma(close, length=period)
        return sma if sma is not None else pd.Series(np.nan, index=close.index)

    @staticmethod
    def calculate_ema(close: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            close: Close price series
            period: Lookback period

        Returns:
            EMA values
        """
        ema = ta.ema(close, length=period)
        return ema if ema is not None else pd.Series(np.nan, index=close.index)

    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 12) -> pd.Series:
        """
        Calculate Rate of Change.

        Args:
            close: Close price series
            period: Lookback period (default: 12)

        Returns:
            ROC values (percentage)
        """
        roc = ta.roc(close, length=period)
        return roc if roc is not None else pd.Series(np.nan, index=close.index)

    @staticmethod
    def calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (trend strength).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period (default: 14)

        Returns:
            ADX values (0-100)
        """
        adx_df = ta.adx(high, low, close, length=period)

        if adx_df is None or adx_df.empty:
            return pd.Series(np.nan, index=close.index)

        # Find ADX column (not DMP or DMN)
        cols = adx_df.columns.tolist()
        adx_col = next(
            (c for c in cols if "ADX" in c and "DMP" not in c and "DMN" not in c),
            None,
        )

        return adx_df[adx_col] if adx_col else pd.Series(np.nan, index=close.index)

    @staticmethod
    def calculate_cmf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """
        Calculate Chaikin Money Flow (volume-weighted).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Lookback period (default: 20)

        Returns:
            CMF values (-1 to +1)
        """
        cmf = ta.cmf(high, low, close, volume, length=period)
        return cmf if cmf is not None else pd.Series(np.nan, index=close.index)

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            OBV values (cumulative)
        """
        obv = ta.obv(close, volume)
        return obv if obv is not None else pd.Series(np.nan, index=close.index)

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period (default: 14)
            d_period: %D smoothing period (default: 3)

        Returns:
            Dictionary with 'k' and 'd' keys (0-100)
        """
        stoch_df = ta.stoch(high, low, close, k=k_period, d=d_period)

        if stoch_df is None or stoch_df.empty:
            nan_series = pd.Series(np.nan, index=close.index)
            return {"k": nan_series, "d": nan_series}

        cols = stoch_df.columns.tolist()
        k_col = next((c for c in cols if "STOCHk" in c), None)
        d_col = next((c for c in cols if "STOCHd" in c), None)

        return {
            "k": stoch_df[k_col] if k_col else pd.Series(np.nan, index=close.index),
            "d": stoch_df[d_col] if d_col else pd.Series(np.nan, index=close.index),
        }

    @classmethod
    def calculate_all_standard(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all standard technical indicators on a price DataFrame.

        Args:
            df: DataFrame with columns: Date, Open, High, Low, Close, Volume

        Returns:
            DataFrame with original data plus all technical indicators
        """
        result = df.copy()

        # Ensure proper column names
        if "close" in result.columns:
            result.columns = [col.capitalize() for col in result.columns]

        # Momentum indicators
        result["RSI_14"] = cls.calculate_rsi(result["Close"], 14)
        result["ROC_12"] = cls.calculate_roc(result["Close"], 12)

        # MACD
        macd = cls.calculate_macd(result["Close"])
        result["MACD"] = macd["macd"]
        result["MACD_signal"] = macd["signal"]
        result["MACD_hist"] = macd["histogram"]

        # Trend indicators
        result["SMA_20"] = cls.calculate_sma(result["Close"], 20)
        result["SMA_50"] = cls.calculate_sma(result["Close"], 50)
        result["SMA_200"] = cls.calculate_sma(result["Close"], 200)
        result["EMA_12"] = cls.calculate_ema(result["Close"], 12)
        result["EMA_26"] = cls.calculate_ema(result["Close"], 26)

        # Bollinger Bands
        bb = cls.calculate_bollinger_bands(result["Close"])
        result["BB_upper"] = bb["upper"]
        result["BB_middle"] = bb["middle"]
        result["BB_lower"] = bb["lower"]
        result["BB_width"] = bb["width"]
        result["BB_pct"] = bb["percent_b"]

        # ADX (trend strength)
        result["ADX"] = cls.calculate_adx(
            result["High"], result["Low"], result["Close"]
        )

        # Volume indicators
        result["OBV"] = cls.calculate_obv(result["Close"], result["Volume"])
        result["CMF"] = cls.calculate_cmf(
            result["High"], result["Low"], result["Close"], result["Volume"]
        )

        # Stochastic
        stoch = cls.calculate_stochastic(result["High"], result["Low"], result["Close"])
        result["STOCH_k"] = stoch["k"]
        result["STOCH_d"] = stoch["d"]

        return result

    @classmethod
    def calculate_multi_timeframe(
        cls, df: pd.DataFrame, windows: list = [21, 63, 126]
    ) -> Dict[int, pd.DataFrame]:
        """
        Calculate indicators across multiple timeframes.

        Args:
            df: Price DataFrame
            windows: List of lookback windows in days (default: [21, 63, 126])

        Returns:
            Dictionary mapping window size to DataFrame with indicators
        """
        results = {}

        for window in windows:
            # Calculate indicators using the window-appropriate data
            window_df = cls.calculate_all_standard(df)

            # Tag with window identifier
            window_df["window"] = window

            results[window] = window_df

        return results
