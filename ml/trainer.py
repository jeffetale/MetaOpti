# ml/trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import logging
import os
from utils.market_utils import fetch_historical_data
from utils.calculation_utils import prepare_training_data
from config import mt5, MODEL_SAVE_DIR, MT5Config, TRADING_CONFIG
from symbols import SYMBOLS as symbols
from datetime import datetime
import time
from model_optimization import perform_hyperparameter_optimization

from logging_config import setup_comprehensive_logging
setup_comprehensive_logging()

class MLTrainer:
    def __init__(self, symbols, timeframe=MT5Config.TIMEFRAME, look_back=TRADING_CONFIG.MODEL_TRAINING_LOOKBACK_PERIOD):
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
            "total_data_points": 0,
            "total_data_size_mb": 0,
            "data_points_by_symbol": {},
            "data_size_by_symbol_mb": {}
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

        data_size_mb = df.memory_usage(deep=True).sum() / (
            1024 * 1024
        )  # Convert bytes to MB

        # Track data points and size for this symbol
        data_points = len(df)
        self.training_stats["total_data_points"] += data_points
        self.training_stats["total_data_size_mb"] += data_size_mb
        self.training_stats["data_points_by_symbol"][symbol] = data_points
        self.training_stats["data_size_by_symbol_mb"][symbol] = data_size_mb

        self.logger.info(
            f"""📊 Retrieved data for {symbol}:
            Rows: {data_points:,}
            Size: {data_size_mb:.2f} MB"""
        )

        return prepare_training_data(df)


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

        self.training_stats["total_data_points"] = 0
        self.training_stats["total_data_size_mb"] = 0
        self.training_stats["data_points_by_symbol"] = {}
        self.training_stats["data_size_by_symbol_mb"] = {}

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
                    best_direction_model, best_direction_params = perform_hyperparameter_optimization(
                        X_train_scaled, y_dir_train, "direction"
                    )

                    # Perform hyperparameter optimization for return model
                    best_return_model, best_return_params = perform_hyperparameter_optimization(
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

                    model_path = Path(MODEL_SAVE_DIR)

                    joblib.dump(
                        optimization_results,
                        model_path / f"{symbol}_optimization_results.pkl"
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

                    # Save the best models
                    best_direction_model.save(str(model_path / f"{symbol}_direction_model.keras"))
                    best_return_model.save(str(model_path / f"{symbol}_return_model.keras"))
                    joblib.dump(scaler, model_path / f"{symbol}_scaler.pkl")

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
                            model_path / f"{symbol}_metadata.pkl"
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
                "📥 Total Data Points": f"{self.training_stats['total_data_points']:,}",
                "💾 Total Data Size": f"{self.training_stats['total_data_size_mb']:.2f} MB",
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

            self.logger.info(f"\n📊 Individual Symbol Performance:\n{'-'*30}")
            for symbol, metrics in self.training_stats["training_times"].items():
                data_points = self.training_stats["data_points_by_symbol"].get(
                    symbol, 0
                )
                data_size = self.training_stats["data_size_by_symbol_mb"].get(symbol, 0)
                self.logger.info(
                    f"""{symbol}:
                    Time: {metrics['time']:.2f} seconds
                    Data: {data_points:,} rows ({data_size:.2f} MB)
                    Accuracy: {metrics['accuracy']:.4f}
                    MAE: {metrics['mae']:.4f}"""
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
