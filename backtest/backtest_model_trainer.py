from datetime import timedelta
import pandas as pd
import logging
from ml.model_optimization import perform_hyperparameter_optimization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from utils.calculation_utils import prepare_training_data
from config import mt5, BACKTEST_MODEL_SAVE_DIR, BackTest
from backtest_data_fetcher import BacktestDataFetcher

class BacktestModelTrainer:
    def __init__(self, symbols: list, backtest_config: BackTest):
        self.logger = logging.getLogger(__name__)
        self.symbols = symbols
        self.config = backtest_config
        self.models = {}
        self.data_fetcher = BacktestDataFetcher()
        
    def validate_symbol_data(self, symbol: str) -> bool:
        """Validate if sufficient data exists for the symbol"""
        return self.data_fetcher.check_data_availability(
            symbol,
            self.config.START_DATE - timedelta(days=60),  # Extra buffer for training
            self.config.END_DATE
        )
        
    def train_historical_models(self) -> dict:
        """Train models for each symbol within the backtest period"""
        trained_models = {}
        valid_symbols = []
        
        for symbol in self.symbols:
            if self.validate_symbol_data(symbol):
                valid_symbols.append(symbol)
            else:
                self.logger.warning(f"Skipping {symbol} due to insufficient data")
                
        if not valid_symbols:
            self.logger.error("No symbols have sufficient data for training")
            return {}
        
        current_date = self.config.START_DATE
        while current_date < self.config.END_DATE:
            window_end = min(current_date + self.config.TRAIN_WINDOW, self.config.END_DATE)
            window_start = window_end - timedelta(days=30)  # Training window
            
            self.logger.info(f"Training models for period: {window_start} to {window_end}")
            
            for symbol in valid_symbols:
                try:
                    # Fetch hourly data for training window
                    rates = mt5.copy_rates_range(
                        symbol, 
                        self.config.TIMEFRAME,
                        window_start,
                        window_end
                    )
                    
                    if rates is None or len(rates) < self.config.TRAINING_LOOKBACK:
                        self.logger.warning(f"Insufficient data for {symbol}")
                        continue
                        
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    X, y_direction, y_return = prepare_training_data(df)
                    
                    if X is None or len(X) == 0:
                        continue
                        
                    # Apply SMOTE and split data
                    smote = SMOTE(random_state=42)
                    X_resampled, y_direction_resampled = smote.fit_resample(X, y_direction)
                    y_return_resampled = pd.Series(list(y_return) * ((len(X_resampled) // len(y_return)) + 1))[:len(X_resampled)]
                    
                    X_train, X_test, y_dir_train, y_dir_test, y_ret_train, y_ret_test = train_test_split(
                        X_resampled, y_direction_resampled, y_return_resampled,
                        test_size=0.2, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train models
                    direction_model, _ = perform_hyperparameter_optimization(X_train_scaled, y_dir_train, "direction")
                    return_model, _ = perform_hyperparameter_optimization(X_train_scaled, y_ret_train, "return")
                    
                    # Save models
                    timestamp = current_date.strftime("%Y%m%d_%H")
                    model_base_path = Path(BACKTEST_MODEL_SAVE_DIR) / symbol / timestamp
                    model_base_path.mkdir(parents=True, exist_ok=True)
                    
                    direction_model.save(str(model_base_path / "direction_model.keras"))
                    return_model.save(str(model_base_path / "return_model.keras"))
                    joblib.dump(scaler, model_base_path / "scaler.pkl")
                    
                    trained_models.setdefault(symbol, {})[timestamp] = {
                        'direction_model': str(model_base_path / "direction_model.keras"),
                        'return_model': str(model_base_path / "return_model.keras"),
                        'scaler': str(model_base_path / "scaler.pkl"),
                        'valid_until': window_end.isoformat()
                    }
                    
                    self.logger.info(f"Successfully trained models for {symbol} at {timestamp}")
                    
                except Exception as e:
                    self.logger.error(f"Error training model for {symbol}: {str(e)}")
                    continue
            
            current_date += self.config.RETRAIN_FREQUENCY
            
        return trained_models