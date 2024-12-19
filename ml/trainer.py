# ml/trainer.py

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
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
import psutil

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

    def create_model_with_hp(self, model_type, input_shape, hp):
        """
        Create model with hyperparameter optimization support
        """
        if model_type == "direction":
            model = Sequential(
                [
                    Dense(
                        hp.get("hidden_units", 64),
                        activation=hp.get("activation", "relu"),
                        input_shape=(input_shape,),
                        kernel_regularizer=tf.keras.regularizers.l2(
                            hp.get("l2_reg", 0.001)
                        ),
                    ),
                    BatchNormalization(),
                    Dropout(hp.get("dropout_rate", 0.3)),
                    Dense(
                        hp.get("hidden_units", 64) // 2,
                        activation=hp.get("activation", "relu"),
                        kernel_regularizer=tf.keras.regularizers.l2(
                            hp.get("l2_reg", 0.001)
                        ),
                    ),
                    BatchNormalization(),
                    Dropout(hp.get("dropout_rate", 0.3)),
                    Dense(16, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )

            optimizer = Adam(
                learning_rate=hp.get("learning_rate", 0.001),
                beta_1=hp.get("beta_1", 0.9),
                beta_2=hp.get("beta_2", 0.999),
            )

            model.compile(
                optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
            )
        else:  # return model
            model = Sequential(
                [
                    Dense(
                        hp.get("hidden_units", 64),
                        activation=hp.get("activation", "relu"),
                        input_shape=(input_shape,),
                        kernel_regularizer=tf.keras.regularizers.l2(
                            hp.get("l2_reg", 0.001)
                        ),
                    ),
                    BatchNormalization(),
                    Dropout(hp.get("dropout_rate", 0.3)),
                    Dense(
                        hp.get("hidden_units", 64) // 2,
                        activation=hp.get("activation", "relu"),
                        kernel_regularizer=tf.keras.regularizers.l2(
                            hp.get("l2_reg", 0.001)
                        ),
                    ),
                    BatchNormalization(),
                    Dropout(hp.get("dropout_rate", 0.3)),
                    Dense(16, activation="relu"),
                    Dense(1),
                ]
            )

            optimizer = Adam(
                learning_rate=hp.get("learning_rate", 0.001),
                beta_1=hp.get("beta_1", 0.9),
                beta_2=hp.get("beta_2", 0.999),
            )

            model.compile(
                optimizer=optimizer, loss="mean_squared_error", metrics=["mae"]
            )

        return model

    def perform_hyperparameter_optimization(self, X_train_scaled, y_train, model_type):
        """
        Perform hyperparameter optimization with expanded parameter grid and advanced techniques
        """
        def check_memory():
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                return False
            return True

        # Default parameters to use when memory is constrained
        default_params = {
            "hidden_units": 64,
            "dropout_rate": 0.3,
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "l2_reg": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999
        }

        if not check_memory():
            self.logger.warning("High memory usage detected. Using default parameters instead of optimization")

            if model_type == "direction":
                model = self.create_model_with_hp("direction", X_train_scaled.shape[1], default_params)
            else:
                model = self.create_model_with_hp("return", X_train_scaled.shape[1], default_params)

            # Add callbacks for training
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
            ]

            # Train with default parameters
            history = model.fit(
                X_train_scaled, y_train,
                epochs=default_params['epochs'],
                batch_size=default_params['batch_size'],
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )

            return model, default_params

        param_grid = {
            "hidden_units": [32, 64, 128],
            "dropout_rate": [0.2, 0.3, 0.4],
            "batch_size": [16, 32, 64],
            "epochs": [50, 100, 150],
            "learning_rate": [0.0001, 0.001, 0.01],
            "l2_reg": [0.0001, 0.001, 0.01],
            "beta_1": [0.9, 0.95],
            "beta_2": [0.999, 0.9999],
        }

        # Create base model wrapper with activation set in the model creation
        if model_type == "direction":
            model_wrapper = KerasClassifier(
                model=lambda: self.create_model_with_hp(
                    "direction",
                    X_train_scaled.shape[1],
                    {"activation": "relu", **default_params},
                ),
                verbose=0,
            )
            scoring = "accuracy"
        else:
            model_wrapper = KerasRegressor(
                model=lambda: self.create_model_with_hp(
                    "return",
                    X_train_scaled.shape[1],
                    {"activation": "relu", **default_params},
                ),
                verbose=0,
            )
            scoring = "neg_mean_absolute_error"

        # Initialize GridSearchCV with advanced configuration
        grid_search = GridSearchCV(
            estimator=model_wrapper,
            param_grid=param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            refit=True,
        )

        # Perform search
        self.logger.info(f"Starting GridSearchCV for {model_type} model...")
        grid_result = grid_search.fit(X_train_scaled, y_train)

        # Log results
        self.logger.info(
            f"\nBest {model_type} model parameters: {grid_result.best_params_}"
        )
        self.logger.info(
            f"Best {model_type} model score: {grid_result.best_score_:.4f}"
        )

        # Create detailed performance report
        cv_results = pd.DataFrame(grid_result.cv_results_)
        best_runs = cv_results.nlargest(5, "mean_test_score")

        self.logger.info("\nTop 5 performing parameter combinations:")
        for idx, run in best_runs.iterrows():
            self.logger.info(
                f"""
            Parameters: {dict((k, run[f'param_{k}']) for k in param_grid.keys())}
            Mean Test Score: {run['mean_test_score']:.4f}
            Mean Train Score: {run['mean_train_score']:.4f}
            """
            )

        return grid_result.best_estimator_.model, grid_result.best_params_

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

                    # Split the resampled data
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

                    # Perform hyperparameter optimization for direction model
                    best_direction_model, best_direction_params = self.perform_hyperparameter_optimization(
                        X_train_scaled, y_dir_train, "direction"
                    )

                    # Perform hyperparameter optimization for return model
                    best_return_model, best_return_params = self.perform_hyperparameter_optimization(
                        X_train_scaled, y_ret_train, "return"
                    )

                    # Save hyperparameter optimization results
                    optimization_results = {
                        "direction_model": {
                            "best_params": best_direction_params,
                            "training_history": best_direction_model.history.history if hasattr(best_direction_model, 'history') else None
                        },
                        "return_model": {
                            "best_params": best_return_params,
                            "training_history": best_return_model.history.history if hasattr(best_return_model, 'history') else None
                        }
                    }

                    joblib.dump(
                        optimization_results,
                        os.path.join(MODEL_SAVE_DIR, f"{symbol}_optimization_results.pkl")
                    )

                    # Evaluate models
                    dir_loss, dir_accuracy = best_direction_model.evaluate(
                        X_test_scaled, y_dir_test, verbose=0
                    )
                    ret_loss, ret_mae = best_return_model.evaluate(
                        X_test_scaled, y_ret_test, verbose=0
                    )

                    self.logger.info(f"Direction Model - Test Accuracy: {dir_accuracy:.4f}")
                    self.logger.info(f"Return Model - Test MAE: {ret_mae:.4f}")

                    # Save models and scaler
                    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

                    # Save the best models
                    best_direction_model.save(os.path.join(MODEL_SAVE_DIR, f"{symbol}_direction_model.keras"))
                    best_return_model.save(os.path.join(MODEL_SAVE_DIR, f"{symbol}_return_model.keras"))
                    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, f"{symbol}_scaler.pkl"))

                    # Save model metadata
                    model_metadata = {
                        "features": X.columns.tolist(),
                        "direction_model_params": best_direction_params,
                        "return_model_params": best_return_params,
                        "direction_model_performance": {
                            "accuracy": dir_accuracy,
                            "loss": dir_loss
                        },
                        "return_model_performance": {
                            "mae": ret_mae,
                            "loss": ret_loss
                        }
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
        self.logger.info("📊 TRAINING SESSION SUMMARY")

        summary_stats = {
            "📈 Statistics": {
                "✅ Total Symbols": self.training_stats["total_symbols"],
                "⭐ Successfully Trained": self.training_stats["trained_symbols"],
                "❌ Failed": self.training_stats["failed_symbols"],
                "⏭️ Skipped": self.training_stats["skipped_symbols"],
            },
            "⏱️ Timing": {"Total Time": f"{total_time:.2f} seconds"},
        }

        if self.training_stats["training_times"]:
            avg_time = sum(self.training_stats["training_times"].values()) / len(
                self.training_stats["training_times"]
            )
            summary_stats["⏱️ Timing"]["Avg Time/Symbol"] = f"{avg_time:.2f} seconds"

            self.logger.info(
                "\n".join(
                    f"{category}\n" + "\n".join(f"    {k}: {v}" for k, v in stats.items())
                    for category, stats in summary_stats.items()
                )
            )

            self.logger.info(f"\n📊 Individual Symbol Times:\n{'-'*30}")
            for symbol, time_taken in self.training_stats["training_times"].items():
                self.logger.info(f"    {symbol}: {time_taken:.2f} seconds")
        else:
            self.logger.warning("No symbols were successfully trained in this session")
            self.logger.info(
                "\n".join(
                    f"{category}\n" + "\n".join(f"    {k}: {v}" for k, v in stats.items())
                    for category, stats in summary_stats.items()
                )
            )

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
