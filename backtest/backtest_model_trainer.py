# backtest/backtest_model_trainer.py

from datetime import datetime, timedelta
import pandas as pd
from typing import Dict
import logging
import os
from ml.trainer import MLTrainer
from config import BackTest, BACKTEST_MODEL_SAVE_DIR
from backtest_data_fetcher import BacktestDataFetcher


class BacktestMLTrainer:
    def __init__(self, symbols: list, backtest_date: datetime):
        self.symbols = symbols
        self.backtest_date = backtest_date
        self.data_fetcher = BacktestDataFetcher()
        self.logger = logging.getLogger(__name__)
        self.historical_trainer = MLTrainer(self.symbols)  
        self.models = {} 

    def train_historical_models(self) -> Dict:
        for symbol in self.symbols:
            try:
                df = self.data_fetcher.fetch_training_data(symbol, self.backtest_date)
                if df is None or len(df) < self.data_fetcher.TRAINING_LOOKBACK:
                    self.logger.error(f"Insufficient data for {symbol}")
                    continue

                model = self.historical_trainer.train_single_model(symbol, df)
                if model is None:
                    continue

                self.models[symbol] = model  # Use self.models

                timestamp = self.backtest_date.strftime("%Y%m%d_%H")
                model_path = os.path.join(BACKTEST_MODEL_SAVE_DIR, f"{symbol}_{timestamp}.keras")
                scaler_path = os.path.join(BACKTEST_MODEL_SAVE_DIR, f"{symbol}_{timestamp}_scaler.pkl")

                self.historical_trainer.save_model(symbol, model_path, scaler_path)
                self.logger.info(f"Saved backtest model for {symbol} at {model_path}")

            except Exception as e:
                self.logger.error(f"Error training model for {symbol}: {e}")
                continue

        if not self.models:
            self.logger.error("No models were successfully trained")
            return {}

        return self.models
