# technical_indicators.py

import numpy as np
import pandas as pd
from typing import Tuple


class TechnicalIndicators:
    @staticmethod
    def sma(prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, span: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        prices: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9
    ) -> pd.Series:
        """Calculate Moving Average Convergence Divergence"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return ranges.max(axis=1).rolling(window=period).mean()

    @staticmethod
    def stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        return 100 * (df["close"] - low_min) / (high_max - low_min)

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        return (high_max - df["close"]) / (high_max - low_min) * -100

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, period: int = 20, num_std: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band

    @staticmethod
    def bollinger_band_width(
        df: pd.DataFrame, period: int = 20, num_std: int = 2
    ) -> pd.Series:
        """Calculate Bollinger Band Width"""
        upper_band, rolling_mean, lower_band = TechnicalIndicators.bollinger_bands(
            df, period, num_std
        )
        return (upper_band - lower_band) / rolling_mean

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high_diff = pd.Series(df["high"]).diff()
        low_diff = -pd.Series(df["low"]).diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        true_range = pd.DataFrame(
            {
                "h-l": df["high"] - df["low"],
                "h-pc": np.abs(df["high"] - df["close"].shift()),
                "l-pc": np.abs(df["low"] - df["close"].shift()),
            }
        ).max(axis=1)

        plus_di = 100 * (
            pd.Series(plus_dm).rolling(window=period).mean()
            / true_range.rolling(window=period).mean()
        )
        minus_di = 100 * (
            pd.Series(minus_dm).rolling(window=period).mean()
            / true_range.rolling(window=period).mean()
        )

        adx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return pd.Series(adx).rolling(window=period).mean()

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        if "volume" not in df.columns:
            df["volume"] = 1

        obv = np.where(
            df["close"] > df["close"].shift(),
            df["volume"],
            np.where(df["close"] < df["close"].shift(), -df["volume"], 0),
        )
        return pd.Series(np.cumsum(obv))

    @staticmethod
    def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        if "volume" not in df.columns:
            df["volume"] = 1

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]

        pos_flow = pd.Series(
            np.where(typical_price > typical_price.shift(), raw_money_flow, 0)
        )
        neg_flow = pd.Series(
            np.where(typical_price < typical_price.shift(), raw_money_flow, 0)
        )

        pos_mf_sum = pos_flow.rolling(window=period).sum()
        neg_mf_sum = neg_flow.rolling(window=period).sum()

        money_ratio = pos_mf_sum / neg_mf_sum
        return 100 - (100 / (1 + money_ratio))

    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a given DataFrame"""
        result = df.copy()

        # Moving Averages
        result["SMA_10"] = cls.sma(result["close"], 10)
        result["SMA_50"] = cls.sma(result["close"], 50)
        result["EMA_20"] = cls.ema(result["close"], 20)

        # Momentum Indicators
        result["RSI"] = cls.rsi(result["close"])
        result["MACD"] = cls.macd(result["close"])
        result["Stochastic"] = cls.stochastic(result)
        result["Williams_R"] = cls.williams_r(result)

        # Volatility Indicators
        result["ATR"] = cls.atr(result)
        result["Bollinger_Band_Width"] = cls.bollinger_band_width(result)

        # Trend Indicators
        result["ADX"] = cls.adx(result)
        result["CCI"] = cls.cci(result)

        # Volume Indicators
        result["OBV"] = cls.obv(result)
        result["MFI"] = cls.money_flow_index(result)

        # Price Changes
        result["price_change_1"] = result["close"].pct_change()
        result["price_change_5"] = result["close"].pct_change(5)
        result["price_change_volatility"] = (
            result["price_change_1"].rolling(window=10).std()
        )

        # Relative Performance
        result["relative_strength"] = (
            result["close"] / result["close"].rolling(window=50).mean()
        )

        return result
