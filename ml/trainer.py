# ml/trainer.py

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
from config import mt5, MODEL_SAVE_DIR
from symbols import SYMBOLS as symbols
from datetime import datetime
import time

from logging_config import setup_comprehensive_logging
setup_comprehensive_logging()

class MLTrainer:
    def __init__(self, symbols, timeframe=mt5.TIMEFRAME_M1, look_back=1000):
        self.logger = logging.getLogger(__name__)
        self.symbols = symbols
        self.timeframe = timeframe
        self.look_back = look_back
        self.models = {}
        self.training_stats = {
            "total_symbols": len(symbols),
            "trained_symbols": 0,
            "failed_symbols": 0,
            "skipped_symbols": 0,
            "start_time": None,
            "end_time": None,
            "training_times": {},
        }

        self.logger.info(
            f"""🚀 Initializing MLTrainer:
            Symbols: {len(symbols)}
            Timeframe: {timeframe}
            Look Back: {look_back}"""
        )

    def prepare_data(self, symbol):
        self.logger.info(f"📥 Preparing data for {symbol}")
        df = fetch_historical_data(symbol, self.timeframe, self.look_back)

        if df is None:
            self.logger.warning(f"⚠️ No historical data retrieved for {symbol}")
            return None, None, None

        self.logger.info(f"📊 Retrieved {len(df)} data points for {symbol}")
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
        self.training_stats["start_time"] = datetime.now()
        self.logger.info(
            f"""
            {'='*50}
            🎮 STARTING MODEL TRAINING SESSION
            🕒 Time: {self.training_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
            📊 Total Symbols: {len(self.symbols)}
            {'='*50}
            """
        )

        for symbol in self.symbols:
            symbol_start_time = time.time()
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"⚡ Training models for {symbol}")
            self.logger.info(f"{'='*50}")

            try:
                X, y_direction, y_return = self.prepare_data(symbol)

                # Additional safety checks
                if X is None or len(X) == 0:
                    self.logger.warning(
                        f"""❌ Skipping {symbol}:
                        Reason: Insufficient or invalid data
                        Data shape: X={None if X is None else X.shape}"""
                    )
                    self.training_stats['skipped_symbols'] += 1
                    continue

                # If only one class exists, skip this symbol
                if len(y_direction.unique()) < 2:
                    self.logger.warning(
                        f"""❌ Skipping {symbol}:
                        Reason: Only one class present
                        Classes: {y_direction.unique()}"""
                    )
                    self.training_stats["skipped_symbols"] += 1
                    continue

                self.logger.info(
                    f"""📊 {symbol} Data Summary:
                    Samples: {len(X)}
                    Features: {len(X.columns)}
                    Class Distribution: {dict(y_direction.value_counts(normalize=True))}"""
                )

                # Apply SMOTE for balancing
                try:
                    self.logger.info(f"⚖️ Applying SMOTE balancing for {symbol}")

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
                    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist

                    # Use save() method without additional options
                    direction_model.save(os.path.join(MODEL_SAVE_DIR, f"{symbol}_direction_model.keras"))
                    return_model.save(os.path.join(MODEL_SAVE_DIR, f"{symbol}_return_model.keras"))
                    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, f"{symbol}_scaler.pkl"))

                    # Save model metadata
                    model_metadata = {
                        "features": X.columns.tolist(),
                        "direction_model_architecture": str(direction_model.summary()),
                        "return_model_architecture": str(return_model.summary()),
                    }
                    joblib.dump(model_metadata, os.path.join(MODEL_SAVE_DIR, f"{symbol}_metadata.pkl"))

                    self.logger.info(
                        f"""✨ {symbol} Direction Model Results:
                        Accuracy: {dir_accuracy:.4f}
                        Loss: {dir_loss:.4f}"""
                    )

                    self.logger.info(
                        f"""📈 {symbol} Return Model Results:
                        MAE: {ret_mae:.4f}
                        Loss: {ret_loss:.4f}"""
                    )

                    training_time = time.time() - symbol_start_time
                    self.training_stats["training_times"][symbol] = training_time
                    self.training_stats["trained_symbols"] += 1

                    self.logger.info(
                        f"""✅ {symbol} Training Complete:
                        Time: {training_time:.2f} seconds"""
                    )

                except Exception as e:
                    self.logger.error(
                        f"""💥 Training Error - {symbol}:
                        Error: {str(e)}""",
                        exc_info=True,
                    )
                    self.training_stats["failed_symbols"] += 1
                    continue

            except Exception as e:
                self.logger.error(
                    f"""🚨 Process Error - {symbol}:
                    Error: {str(e)}""",
                    exc_info=True,
                )
                self.training_stats["failed_symbols"] += 1
                continue

        self.training_stats["end_time"] = datetime.now()
        total_time = (
            self.training_stats["end_time"] - self.training_stats["start_time"]
        ).total_seconds()

        self.logger.info(f"\n{'='*50}")
        self.logger.info("""📊 TRAINING SESSION SUMMARY""")
        self.logger.info(
            f"""
            📈 Statistics:
            ✅ Total Symbols: {self.training_stats['total_symbols']}
            ⭐ Successfully Trained: {self.training_stats['trained_symbols']}
            ❌ Failed: {self.training_stats['failed_symbols']}
            ⏭️ Skipped: {self.training_stats['skipped_symbols']}
            
            ⏱️ Timing:
            Total Time: {total_time:.2f} seconds
            Avg Time/Symbol: {sum(self.training_stats['training_times'].values())/len(self.training_stats['training_times']):.2f} seconds
            
            📊 Individual Symbol Times:
            {'-'*30}"""
        )
        for symbol, time_taken in self.training_stats["training_times"].items():
            self.logger.info(f"    {symbol}: {time_taken:.2f} seconds")

        self.logger.info(f"{'='*50}")
        self.logger.info("🏁 Model training completed!")


if __name__ == "__main__":
    # Initialize MT5 connection
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        exit()

    ml_trainer = MLTrainer(symbols)
    ml_trainer.train_models()

    mt5.shutdown()
