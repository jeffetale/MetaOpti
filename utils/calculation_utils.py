# utils/calculation_utils.py

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List
from ml.features.feature_engineering import FeatureEngineer


def prepare_training_data(
    df: pd.DataFrame, min_samples: int = 100
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """
    Prepare data for ML model training

    Args:
        df: Raw price data DataFrame
        min_samples: Minimum number of samples required

    Returns:
        Tuple of (features DataFrame, direction labels, return labels)
    """
    try:
        # Apply feature engineering
        df = FeatureEngineer.engineer_features(df)
        df.dropna(inplace=True)

        logging.info(f"Total data points: {len(df)}")
        logging.info(
            f"Class Distribution before filtering: {df['target_direction'].value_counts(normalize=True)}"
        )

        # Filter out neutral cases
        df_filtered = df[df["target_direction"] != 0].copy()

        logging.info(f"Data points after filtering: {len(df_filtered)}")
        logging.info(
            f"Class Distribution after filtering: {df_filtered['target_direction'].value_counts(normalize=True)}"
        )

        if len(df_filtered) < min_samples:
            logging.warning(
                f"Insufficient data. Need at least {min_samples} data points."
            )
            return None, None, None

        # Create binary target
        df_filtered.loc[:, "target_direction_binary"] = np.where(
            df_filtered["target_direction"] > 0, 1, 0
        )

        # Define feature set
        features = [
            "SMA_10",
            "SMA_50",
            "EMA_20",  # Moving Averages
            "RSI",
            "MACD",
            "Stochastic",
            "Williams_R",  # Momentum
            "ATR",
            "Bollinger_Band_Width",  # Volatility
            "ADX",
            "CCI",  # Trend
            "OBV",
            "MFI",  # Volume
            "price_change_1",
            "price_change_5",
            "price_change_volatility",
            "relative_strength",
        ]

        # Validate features
        missing_features = [f for f in features if f not in df_filtered.columns]
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
            return None, None, None

        X = df_filtered[features]
        y_direction = df_filtered["target_direction_binary"]
        y_return = df_filtered["target_return"]

        # Validate shapes
        if not (len(X) == len(y_direction) == len(y_return)):
            logging.error("Inconsistent sample sizes")
            logging.error(f"X shape: {X.shape}")
            logging.error(f"y_direction shape: {y_direction.shape}")
            logging.error(f"y_return shape: {y_return.shape}")
            return None, None, None

        return X, y_direction, y_return

    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        return None, None, None


def prepare_prediction_data(
    df: pd.DataFrame, features: List[str]
) -> Optional[pd.DataFrame]:
    """
    Prepare data for prediction

    Args:
        df: Raw price data DataFrame
        features: List of feature names used in training

    Returns:
        DataFrame with calculated features
    """
    try:
        if len(df) < 50:  # Minimum required for calculations
            logging.warning(f"Insufficient data points: {len(df)}")
            return None

        # Apply feature engineering
        df = FeatureEngineer.engineer_features(df)

        # Drop NaN values
        df.dropna(inplace=True)

        if len(df) == 0:
            logging.warning("No valid data points after feature engineering")
            return None

        # Select only the features used in training
        return df[features]

    except Exception as e:
        logging.error(f"Error preparing prediction data: {e}")
        return None
