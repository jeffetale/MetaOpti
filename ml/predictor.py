import tensorflow as tf
import joblib
import logging
from typing import Tuple, Optional
from config import (
    mt5,
    MODEL_SAVE_DIR,
    BACKTEST_MODEL_SAVE_DIR,
    TRADING_CONFIG,
    MT5Config,
    BackTest,
)
from utils.market_utils import fetch_historical_data
from utils.calculation_utils import prepare_prediction_data
import os
from datetime import datetime

from logging_config import setup_comprehensive_logging

setup_comprehensive_logging()


class MLPredictor:
    def __init__(
        self,
        symbol: str,
        backtest_mode: bool = False,
        backtest_date: Optional[datetime] = None,
    ):
        self.symbol = symbol
        self.backtest_mode = backtest_mode
        self.backtest_date = backtest_date
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
        """Load pre-trained models based on mode (live or backtest)"""
        try:
            if self.backtest_mode and self.backtest_date:
                # For backtest mode, load models with timestamp
                timestamp = self.backtest_date.strftime("%Y%m%d_%H")
                model_dir = BACKTEST_MODEL_SAVE_DIR
                model_prefix = f"{self.symbol}_{timestamp}"
            else:
                # For live trading, load latest models
                model_dir = MODEL_SAVE_DIR
                model_prefix = self.symbol

            # Load metadata first to get feature names
            metadata = joblib.load(
                os.path.join(model_dir, f"{model_prefix}_metadata.pkl")
            )
            self.features = metadata.get("features", [])

            # Load direction and return models
            self.direction_model = tf.keras.models.load_model(
                os.path.join(model_dir, f"{model_prefix}_direction_model.keras")
            )
            self.return_model = tf.keras.models.load_model(
                os.path.join(model_dir, f"{model_prefix}_return_model.keras")
            )
            self.scaler = joblib.load(
                os.path.join(model_dir, f"{model_prefix}_scaler.pkl")
            )

            logging.info(
                f"Models loaded for {self.symbol} {'(backtest)' if self.backtest_mode else '(live)'}"
            )
        except FileNotFoundError:
            logging.error(
                f"Models for {self.symbol} not found in "
                f"{'backtest' if self.backtest_mode else 'live'} mode. "
                "Train models first."
            )
            return None

    def _predict_direction(self, scaled_features):
        """Dedicated method for direction prediction"""
        return self.direction_model(scaled_features)

    def _predict_return(self, scaled_features):
        """Dedicated method for return prediction"""
        return self.return_model(scaled_features)

    def predict(
        self,
        timeframe=None,
        look_back=None,
        threshold=None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[Optional[str], float, float]:
        """Predict trading signal and potential return"""
        # Set default parameters based on mode
        if timeframe is None:
            timeframe = (
                BackTest.TIMEFRAME if self.backtest_mode else MT5Config.TIMEFRAME
            )
        if look_back is None:
            look_back = (
                BackTest.PREDICTION_LOOKBACK
                if self.backtest_mode
                else TRADING_CONFIG.MODEL_PREDICTION_LOOKBACK_PERIODS
            )
        if threshold is None:
            threshold = TRADING_CONFIG.HIGH_CONFIDENCE_THRESHOLD

        if not all(
            [self.direction_model, self.return_model, self.scaler, self.features]
        ):
            logging.error("Models not loaded. Cannot predict.")
            return None, 0, 0

        # Fetch historical data based on mode
        if self.backtest_mode and current_time:
            rates_frame = fetch_historical_data(
                self.symbol, timeframe, look_back, end_date=current_time
            )
        else:
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
