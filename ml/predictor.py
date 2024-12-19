import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import logging
from typing import Tuple, Optional
from config import mt5
from utils.market_utils import fetch_historical_data
from utils.calculation_utils import prepare_prediction_data


class MLPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.direction_model = None
        self.return_model = None
        self.scaler = None
        self.features = None

        self.load_models()

        self.predict_direction = tf.function(
            self._predict_direction, reduce_retracing=True
        )
        self.predict_return = tf.function(self._predict_return, reduce_retracing=True)

    def load_models(self):
        """Load pre-trained models for a specific symbol"""
        try:
            # Load metadata first to get feature names
            metadata = joblib.load(f"ml/ml_models/{self.symbol}_metadata.pkl")
            self.features = metadata.get("features", [])

            self.direction_model = tf.keras.models.load_model(
                f"ml/ml_models/{self.symbol}_direction_model.keras"
            )
            self.return_model = tf.keras.models.load_model(
                f"ml/ml_models/{self.symbol}_return_model.keras"
            )
            self.scaler = joblib.load(f"ml/ml_models/{self.symbol}_scaler.pkl")

            logging.info(f"Models loaded for {self.symbol}")
        except FileNotFoundError:
            logging.error(f"Models for {self.symbol} not found. Train models first.")
            return None

    def _predict_direction(self, scaled_features):
        """Dedicated method for direction prediction"""
        return self.direction_model(scaled_features)

    def _predict_return(self, scaled_features):
        """Dedicated method for return prediction"""
        return self.return_model(scaled_features)

    def predict(
        self, timeframe=mt5.TIMEFRAME_M1, look_back=100, threshold=0.6
    ) -> Tuple[Optional[str], float, float]:
        """Predict trading signal and potential return"""
        if not all(
            [self.direction_model, self.return_model, self.scaler, self.features]
        ):
            logging.error("Models not loaded. Cannot predict.")
            return None, 0, 0

        # Fetch latest price data
        rates_frame = fetch_historical_data(self.symbol, timeframe, look_back)
        if rates_frame is None:
            return None, 0, 0

        try:
            # Extract features ensuring exact column names and order
            features_df = prepare_prediction_data(rates_frame, self.features)

            # If no features remain, return neutral
            if features_df is None or len(features_df) == 0:
                logging.info(f"{self.symbol} insufficient data for prediction")
                return None, 0, 0

            # Scale features
            scaled_features = self.scaler.transform(features_df)

            # Predict direction and return
            direction_prob = self.predict_direction(
                tf.convert_to_tensor(scaled_features, dtype=tf.float32)
            )[0][0].numpy()

            predicted_return = self.predict_return(
                tf.convert_to_tensor(scaled_features, dtype=tf.float32)
            )[0][0].numpy()

            # Interpret predictions
            if direction_prob > threshold:
                signal = "buy"
            elif direction_prob < (1 - threshold):
                signal = "sell"
            else:
                signal = "hold"

            return signal, direction_prob, predicted_return

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None, 0, 0
