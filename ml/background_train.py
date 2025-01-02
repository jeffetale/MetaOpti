# ml/background_train.py

from threading import Thread
import schedule
import time
import logging
from datetime import datetime, timedelta
from ml.trainer import MLTrainer
from logging_config import setup_comprehensive_logging
import joblib
from pathlib import Path
from config import MODEL_SAVE_DIR, initialize_mt5

setup_comprehensive_logging()


class BackgroundTrainer:
    def __init__(self, symbols, training_interval_minutes=15, max_model_age_minutes=30):
        self.symbols = symbols
        self.training_interval = training_interval_minutes
        self.max_model_age = max_model_age_minutes
        self.trainer = MLTrainer(symbols)
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.training_thread = None
        self.cumulative_stats = {
            "total_training_sessions": 0,
            "total_data_points": 0,
            "total_data_size_mb": 0,
            "data_by_session": [],
        }

        self.logger.info(
            f"""🎯 Initialized BackgroundTrainer:
            Symbols: {len(symbols)}
            Training Interval: {training_interval_minutes} minutes
            Max Model Age: {max_model_age_minutes} minutes"""
        )

    def initialize(self):
        """Initialize and validate models before starting the bot"""
        self.logger.info("🔍 Checking model status before bot start...")

        if not self._validate_models():
            self.logger.info("🔄 Initial model training required")
            return self._perform_initial_training()

        self.logger.info("✅ Model validation complete - ready to start")
        return True

    def _validate_models(self):
        """Check if models exist and are up to date"""
        try:
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

            for symbol in self.symbols:
                # Check if model files exist
                model_files = [
                    f"{symbol}_direction_model.keras",
                    f"{symbol}_return_model.keras",
                    f"{symbol}_scaler.pkl",
                    f"{symbol}_metadata.pkl",
                ]

                for file in model_files:
                    file_path = MODEL_SAVE_DIR / file
                    if not file_path.exists():
                        self.logger.warning(f"❌ Missing model file: {file}")
                        return False

                # Check model age
                metadata_path = MODEL_SAVE_DIR / f"{symbol}_metadata.pkl"
                try:
                    metadata = joblib.load(metadata_path)
                    training_time = metadata.get("training_time", None)

                    if not training_time:
                        self.logger.warning(f"⚠️ No training timestamp for {symbol}")
                        return False

                    age = datetime.now() - training_time
                    if age > timedelta(minutes=self.max_model_age):
                        self.logger.warning(
                            f"""⚠️ Model for {symbol} is too old:
                            Age: {age.total_seconds() / 60:.1f} minutes
                            Max allowed: {self.max_model_age} minutes"""
                        )
                        return False

                except Exception as e:
                    self.logger.error(
                        f"💥 Error reading metadata for {symbol}: {str(e)}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"💥 Error during model validation: {str(e)}")
            return False

    def _perform_initial_training(self):
        """Perform initial model training"""
        try:
            self.logger.info(
                f"""
                {'='*50}
                🎮 Starting initial model training
                🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                {'='*50}
                """
            )

            # Execute training synchronously for initial training
            self.trainer.train_models()

            # Update metadata with training time
            for symbol in self.symbols:
                metadata_path = MODEL_SAVE_DIR / f"{symbol}_metadata.pkl"
                try:
                    metadata = joblib.load(metadata_path)
                    metadata["training_time"] = datetime.now()
                    joblib.dump(metadata, metadata_path)
                except Exception as e:
                    self.logger.error(f"Error updating metadata for {symbol}: {str(e)}")
                    return False

            self.logger.info("✅ Initial training completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"💥 Error during initial training: {str(e)}")
            return False

    def start(self):
        """Start the background training scheduler"""
        if self.is_running:
            self.logger.warning("⚠️ Background trainer is already running")
            return

        self.is_running = True
        self.training_thread = Thread(target=self._run_scheduler, daemon=True)
        self.training_thread.start()

        self.logger.info("✨ Background trainer started successfully")

    def stop(self):
        """Stop the background training scheduler"""
        self.is_running = False
        if self.training_thread:
            self.training_thread.join(timeout=1)
        self.logger.info("🛑 Background trainer stopped")

    def _run_scheduler(self):
        """Run the scheduler loop"""
        schedule.every(self.training_interval).minutes.do(self._training_job)

        self.logger.info(
            f"📅 Scheduled training every {self.training_interval} minutes"
        )

        while self.is_running:
            schedule.run_pending()
            time.sleep(1)

    def _training_job(self):
        """Execute the training job"""
        try:
            self.logger.info(
                f"""
                {'='*50}
                🎮 Starting scheduled model training
                🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                {'='*50}
                """
            )

            if not initialize_mt5():
                self.logger.warning("⚠️ Failed to initialize MT5, skipping training")
                return

            training_thread = Thread(target=self._execute_training)
            training_thread.start()
            training_thread.join(timeout=self.training_interval * 60)

            if training_thread.is_alive():
                self.logger.warning(
                    "⚠️ Training job exceeded time limit and will continue in background"
                )

        except Exception as e:
            self.logger.error(f"💥 Error in training job: {str(e)}", exc_info=True)

    def _execute_training(self):
        """Execute the actual training process and track data size"""
        try:
            training_stats = self.trainer.train_models()

            # Update cumulative statistics
            self.cumulative_stats["total_training_sessions"] += 1
            self.cumulative_stats["total_data_points"] += training_stats[
                "total_data_points"
            ]
            self.cumulative_stats["total_data_size_mb"] += training_stats[
                "total_data_size_mb"
            ]
            self.cumulative_stats["data_by_session"].append(
                {
                    "timestamp": datetime.now(),
                    "data_points": training_stats["total_data_points"],
                    "data_size_mb": training_stats["total_data_size_mb"],
                }
            )

            # Log cumulative statistics
            self.logger.info(
                f"""
            {'='*50}
            📊 CUMULATIVE TRAINING STATISTICS
            {'='*50}
            🔄 Total Training Sessions: {self.cumulative_stats['total_training_sessions']}
            📥 Total Data Points Processed: {self.cumulative_stats['total_data_points']:,}
            💾 Total Data Size Processed: {self.cumulative_stats['total_data_size_mb']:.2f} MB
            📈 Average Data per Session: 
                Points: {self.cumulative_stats['total_data_points'] / self.cumulative_stats['total_training_sessions']:,.0f}
                Size: {self.cumulative_stats['total_data_size_mb'] / self.cumulative_stats['total_training_sessions']:.2f} MB
            {'='*50}
            """
            )

            # Update metadata with new training time and data size
            for symbol in self.symbols:
                metadata_path = MODEL_SAVE_DIR / f"{symbol}_metadata.pkl"
                try:
                    metadata = joblib.load(metadata_path)
                    metadata["training_time"] = datetime.now()
                    metadata["data_points"] = training_stats[
                        "data_points_by_symbol"
                    ].get(symbol, 0)
                    metadata["data_size_mb"] = training_stats[
                        "data_size_by_symbol_mb"
                    ].get(symbol, 0)
                    joblib.dump(metadata, metadata_path)
                except Exception as e:
                    self.logger.error(f"Error updating metadata for {symbol}: {str(e)}")

            self.logger.info("✅ Scheduled training completed successfully")
        except Exception as e:
            self.logger.error(f"💥 Training execution error: {str(e)}", exc_info=True)

    def force_train(self):
        """Force immediate training"""
        self.logger.info("🎯 Forcing immediate model training")
        self._training_job()
        self.logger.info("✅ Immediate training completed successfully")
