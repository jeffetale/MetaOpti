# ml_trainer.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
from utils.market_utils import fetch_historical_data
from utils.calculation_utils import prepare_training_data
from config import mt5


class MLTrainer:
    def __init__(self, symbols, timeframe=mt5.TIMEFRAME_M1, look_back=1000):
        self.symbols = symbols
        self.timeframe = timeframe
        self.look_back = look_back
        self.models = {}

    def prepare_data(self, symbol):
        """Prepare data for neural network training"""
        df = fetch_historical_data(symbol, self.timeframe, self.look_back)
        if df is None:
            return None, None, None

        return prepare_training_data(df)

    def create_direction_model(self, input_shape):
        """Create neural network for direction classification"""
        model = Sequential(
            [
                Dense(
                    64,
                    activation="relu",
                    input_shape=(input_shape,),
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                BatchNormalization(),
                Dropout(0.3),
                Dense(
                    32,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                BatchNormalization(),
                Dropout(0.3),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),  # Binary classification
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def create_return_model(self, input_shape):
        """Create neural network for return regression"""
        model = Sequential(
            [
                Dense(
                    64,
                    activation="relu",
                    input_shape=(input_shape,),
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                BatchNormalization(),
                Dropout(0.3),
                Dense(
                    32,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                BatchNormalization(),
                Dropout(0.3),
                Dense(16, activation="relu"),
                Dense(1),  # Linear output for regression
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae"],
        )
        return model

    def train_models(self):
        for symbol in self.symbols:
            logging.info(f"Training models for {symbol}")
            try:
                X, y_direction, y_return = self.prepare_data(symbol)

                # Additional safety checks
                if X is None or len(X) == 0:
                    logging.warning(
                        f"Skipping {symbol} due to insufficient or invalid data"
                    )
                    continue

                # If only one class exists, skip this symbol
                if len(y_direction.unique()) < 2:
                    logging.warning(f"Skipping {symbol} - only one class present")
                    continue

                # Apply SMOTE for balancing
                try:
                    smote = SMOTE(random_state=42)
                    X_resampled, y_direction_resampled = smote.fit_resample(
                        X, y_direction
                    )

                    y_return_resampled = pd.Series(
                        list(y_return) * ((len(X_resampled) // len(y_return)) + 1)
                    )[: len(X_resampled)]

                    (
                        X_train,
                        X_test,
                        y_dir_train,
                        y_dir_test,
                        y_ret_train,
                        y_ret_test,
                    ) = train_test_split(
                        X_resampled,
                        y_direction_resampled,
                        y_return_resampled,
                        test_size=0.2,
                        random_state=42,
                    )

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Create and train direction model
                    direction_model = self.create_direction_model(
                        X_train_scaled.shape[1]
                    )

                    # Callbacks for training
                    callbacks = [
                        EarlyStopping(
                            monitor="val_loss",
                            patience=15,
                            restore_best_weights=True,
                            min_delta=0.001,
                            verbose=1,
                        ),
                        ReduceLROnPlateau(
                            monitor="val_loss", factor=0.5, patience=7, min_lr=0.000001
                        ),
                    ]

                    # Train direction model
                    direction_history = direction_model.fit(
                        X_train_scaled,
                        y_dir_train,
                        validation_split=0.2,
                        epochs=150,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=1,
                    )

                    # Create and train return model
                    return_model = self.create_return_model(X_train_scaled.shape[1])

                    return_history = return_model.fit(
                        X_train_scaled,
                        y_ret_train,
                        validation_split=0.2,
                        epochs=150,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=1,
                    )

                    # Evaluate models
                    dir_loss, dir_accuracy = direction_model.evaluate(
                        X_test_scaled, y_dir_test
                    )
                    ret_loss, ret_mae = return_model.evaluate(X_test_scaled, y_ret_test)

                    logging.info(f"Direction Model - Test Accuracy: {dir_accuracy}")
                    logging.info(f"Return Model - Test MAE: {ret_mae}")

                    # Save models and scaler
                    os.makedirs("ml_models", exist_ok=True)

                    # Use save() method without additional options
                    direction_model.save(f"ml_models/{symbol}_direction_model.keras")
                    return_model.save(f"ml_models/{symbol}_return_model.keras")
                    joblib.dump(scaler, f"ml_models/{symbol}_scaler.pkl")

                    # Save model metadata
                    model_metadata = {
                        "features": X.columns.tolist(),
                        "direction_model_architecture": str(direction_model.summary()),
                        "return_model_architecture": str(return_model.summary()),
                    }
                    joblib.dump(model_metadata, f"ml_models/{symbol}_metadata.pkl")

                except Exception as e:
                    logging.error(f"SMOTE or training error for {symbol}: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    continue

            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

        logging.info("Model training completed.")

if __name__ == "__main__":
    # Initialize MT5 connection
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        exit()

    # Symbols from config to train models
    symbols = [
        "XAUUSD",
        "XTIUSD",
        "XAGUSD",
        "US30",
        "USTEC",
        "BTCUSD",
        "ETHUSD",
        "BCHUSD",
        "DOGUSD",
        "LTCUSD",
        "EURJPY",
        "USDJPY",
        "GBPUSD",
        "EURUSD",
    ]

    ml_trainer = MLTrainer(symbols)
    ml_trainer.train_models()

    mt5.shutdown()
