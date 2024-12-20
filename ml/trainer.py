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
    def __init__(self, symbols, timeframe=mt5.TIMEFRAME_M5, look_back=1000):
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
        # Create optimizer with hyperparameters
        optimizer = Adam(
            learning_rate=hp['learning_rate'],
            beta_1=hp['beta_1'],
            beta_2=hp['beta_2']
        )

        if model_type == "direction":
            model = Sequential([
                Dense(hp['hidden_units'], 
                    activation=hp['activation'],
                    input_shape=(input_shape,),
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(hp['hidden_units'] // 2,
                    activation=hp['activation'],
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid")
            ])

            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        else:
            model = Sequential([
                Dense(hp['hidden_units'],
                    activation=hp['activation'],
                    input_shape=(input_shape,),
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(hp['hidden_units'] // 2,
                    activation=hp['activation'],
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer=optimizer,
                loss="mean_squared_error",
                metrics=["mae"]
            )

        return model

    def perform_hyperparameter_optimization(self, X_train_scaled, y_train, model_type):
        """
        Perform hyperparameter optimization with proper model wrapping
        """
        def create_model(hidden_units, dropout_rate, activation, l2_reg, learning_rate, beta_1, beta_2):
            hp = {
                'hidden_units': hidden_units,
                'dropout_rate': dropout_rate,
                'activation': activation,
                'l2_reg': l2_reg,
                'learning_rate': learning_rate,
                'beta_1': beta_1,
                'beta_2': beta_2
            }
            return self.create_model_with_hp(model_type, X_train_scaled.shape[1], hp)

        # Define the parameter grid
        param_grid = {
            'hidden_units': [32, 64],
            'dropout_rate': [0.2, 0.3],
            'activation': ['relu'],
            'l2_reg': [0.001],
            'learning_rate': [0.001, 0.01],
            'beta_1': [0.9],
            'beta_2': [0.999],
            'batch_size': [32, 64],
            'epochs': [30, 50]
        }

        # Create the correct wrapper
        if model_type == "direction":
            model_wrapper = KerasClassifier(
                model=create_model,
                verbose=0
            )
            scoring = 'accuracy'
        else:
            model_wrapper = KerasRegressor(
                model=create_model,
                verbose=0
            )
            scoring = 'neg_mean_absolute_error'

        try:
            # Initialize and run RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model_wrapper,
                param_distributions=param_grid,
                n_iter=5,
                cv=3,
                scoring=scoring,
                n_jobs=1,
                verbose=1,
                return_train_score=True,
                random_state=42
            )

            search_start_time = time.time()
            random_search_result = random_search.fit(X_train_scaled, y_train)
            search_time = time.time() - search_start_time

            self.logger.info(f"\nHyperparameter optimization completed in {search_time:.2f} seconds")
            self.logger.info(f"Best {model_type} model parameters: {random_search_result.best_params_}")
            self.logger.info(f"Best {model_type} model score: {random_search_result.best_score_:.4f}")

            # Return the best model and parameters
            best_params = random_search_result.best_params_
            best_model = create_model(
                hidden_units=best_params['hidden_units'],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation'],
                l2_reg=best_params['l2_reg'],
                learning_rate=best_params['learning_rate'],
                beta_1=best_params['beta_1'],
                beta_2=best_params['beta_2']
            )

            # Train the best model with the best parameters
            history = best_model.fit(
                X_train_scaled,
                y_train,
                batch_size=best_params['batch_size'],
                epochs=best_params['epochs'],
                verbose=0
            )

            return best_model, best_params

        except Exception as e:
            self.logger.error(f"Error during hyperparameter optimization: {str(e)}")
            # Create model with default parameters
            default_params = {
                'hidden_units': 64,
                'dropout_rate': 0.3,
                'activation': 'relu',
                'l2_reg': 0.001,
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999
            }

            default_model = create_model(
                hidden_units=default_params['hidden_units'],
                dropout_rate=default_params['dropout_rate'],
                activation=default_params['activation'],
                l2_reg=default_params['l2_reg'],
                learning_rate=default_params['learning_rate'],
                beta_1=default_params['beta_1'],
                beta_2=default_params['beta_2']
            )

            # Train the default model
            history = default_model.fit(
                X_train_scaled,
                y_train,
                batch_size=32,
                epochs=50,
                verbose=0
            )

            return default_model, default_params

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
                            "loss": dir_loss,
                        },
                        "return_model_performance": {"mae": ret_mae, "loss": ret_loss},
                        "training_timestamp": datetime.now().isoformat(),  # Add training completion timestamp
                        "training_duration_seconds": time.time() - symbol_start_time,
                    }

                
                    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

                    try:
                        joblib.dump(
                            model_metadata,
                            os.path.join(MODEL_SAVE_DIR, f"{symbol}_metadata.pkl"),
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error saving metadata for {symbol}: {str(e)}"
                        )
                        raise  

                    self.logger.info(
                        f"✅ Saved metadata for {symbol} with training timestamp"
                    )

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
                    self.training_stats["training_times"][symbol] = {
                        "time": training_time,
                        "accuracy": dir_accuracy,
                        "mae": ret_mae
                    }
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
            avg_time = sum(
                stats["time"]
                for stats in self.training_stats["training_times"].values()
            ) / len(self.training_stats["training_times"])
            summary_stats["⏱️ Timing"]["Avg Time/Symbol"] = f"{avg_time:.2f} seconds"

            self.logger.info(
                "\n".join(
                    f"{category}\n"
                    + "\n".join(f"    {k}: {v}" for k, v in stats.items())
                    for category, stats in summary_stats.items()
                )
            )

            self.logger.info(f"\n📊 Individual Symbol Times:\n{'-'*30}")
            for symbol, metrics in self.training_stats["training_times"].items():
                self.logger.info(
                    f"{symbol}: {metrics['time']:.2f} seconds | Accuracy: {metrics['accuracy']:.4f} | MAE: {metrics['mae']:.4f}"
                )
        else:
            self.logger.warning("No symbols were successfully trained in this session")
            self.logger.info(
                "\n".join(
                    f"{category}\n"
                    + "\n".join(f"    {k}: {v}" for k, v in stats.items())
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
