#  ml/features/feature_engineering.py

import numpy as np
import pandas as pd
from .technical_indicators import TechnicalIndicators


class FeatureEngineer:
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering with expanded indicator set"""
        result = df.copy()

        # Moving Averages
        result["SMA_10"] = TechnicalIndicators.sma(result["close"], 10)
        result["SMA_50"] = TechnicalIndicators.sma(result["close"], 50)
        result["EMA_20"] = TechnicalIndicators.ema(result["close"], 20)

        # Momentum Indicators
        result["RSI"] = TechnicalIndicators.rsi(result["close"])
        result["MACD"] = TechnicalIndicators.macd(result["close"])
        result["Stochastic"] = TechnicalIndicators.stochastic(result)
        result["Williams_R"] = TechnicalIndicators.williams_r(result)

        # Volatility Indicators
        result["ATR"] = TechnicalIndicators.atr(result)
        result["Bollinger_Band_Width"] = TechnicalIndicators.bollinger_band_width(
            result
        )

        # Trend Indicators
        result["ADX"] = TechnicalIndicators.adx(result)
        result["CCI"] = TechnicalIndicators.cci(result)

        # Volume-based Indicators
        result["OBV"] = TechnicalIndicators.obv(result)
        result["MFI"] = TechnicalIndicators.money_flow_index(result)

        # Advanced Price Change Features
        result["price_change_1"] = result["close"].pct_change()
        result["price_change_5"] = result["close"].pct_change(5)
        result["price_change_volatility"] = (
            result["price_change_1"].rolling(window=10).std()
        )

        # Relative Performance Indicators
        result["relative_strength"] = (
            result["close"] / result["close"].rolling(window=50).mean()
        )

        # Target Features
        result["future_close"] = result["close"].shift(-1)
        result["target_return"] = (result["future_close"] - result["close"]) / result[
            "close"
        ]

        # Multi-threshold classification target
        result["target_direction"] = np.select(
            [
                result["target_return"]
                > result["target_return"].quantile(0.7),  # Top 30% positive
                result["target_return"]
                < result["target_return"].quantile(0.3),  # Bottom 30% negative
            ],
            [1, -1],
            default=0,  # Neutral movement
        )

        return result
