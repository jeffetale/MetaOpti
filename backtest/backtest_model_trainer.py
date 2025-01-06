# backtest/backtest_model_trainer.py

from datetime import timedelta, datetime
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from backtest_data_preparation import prepare_training_data
from config import mt5, BACKTEST_MODEL_SAVE_DIR, BackTest
from backtest_data_fetcher import BacktestDataFetcher

class BacktestModelTrainer:
    def __init__(self, symbols: list, backtest_config: BackTest):
        self.logger = logging.getLogger(__name__)
        self.symbols = symbols
        self.config = backtest_config
        self.models = {}
        self.data_fetcher = BacktestDataFetcher()
        
    def _train_direction_model(self, X_train, y_train):
        """Train the direction prediction model"""
        try:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error in training direction model: {str(e)}")
            return None

    def _train_return_model(self, X_train, y_train):
        """Train the return prediction model"""
        try:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error in training return model: {str(e)}")
            return None

    def validate_symbol_data(self, symbol: str) -> bool:
        """Validate if sufficient data exists for the symbol"""
        extended_start = self.config.START_DATE - timedelta(days=120)
        return self.data_fetcher.check_data_availability(
            symbol,
            extended_start,
            self.config.END_DATE
        )
        
    def train_historical_models(self) -> dict:
        """Train models for each symbol within the backtest period"""
        trained_models = {}
        
        for symbol in self.symbols:
            try:
                # Fetch data with larger timeframe
                rates = mt5.copy_rates_range(
                    symbol, 
                    self.config.TIMEFRAME,
                    self.config.START_DATE - timedelta(days=90), 
                    self.config.END_DATE
                )
                
                if rates is None or len(rates) < 500:
                    self.logger.error(f"Insufficient historical data for {symbol}")
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                self.logger.info(f"Preparing training data for {symbol} with {len(df)} data points")
                
                # Prepare training data
                X, y_direction, y_return = prepare_training_data(df)
                
                if X is None or y_direction is None or y_return is None:
                    self.logger.error(f"Failed to prepare training data for {symbol}")
                    continue
                    
                # Save features list for later use
                feature_list = list(X.columns)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_dir_train, y_dir_test = train_test_split(
                    X_scaled, y_direction, test_size=0.2, random_state=42
                )
                
                # Train models
                direction_model = self._train_direction_model(X_train, y_dir_train)
                return_model = self._train_return_model(X_train, y_return[:len(X_train)])
                
                if direction_model is None or return_model is None:
                    self.logger.error(f"Failed to train models for {symbol}")
                    continue
                
                # Save models and metadata
                timestamp = self.config.START_DATE.strftime("%Y%m%d_%H")
                model_base_path = Path(BACKTEST_MODEL_SAVE_DIR) / symbol
                model_base_path.mkdir(parents=True, exist_ok=True)
                
                metadata = {
                    'features': feature_list,
                    'training_date': datetime.now().isoformat(),
                    'data_points': len(X),
                    'performance': {
                        'training_size': len(X_train),
                        'test_size': len(X_test)
                    }
                }
                
                model_prefix = f"{symbol}_{timestamp}"
                
                direction_model.save(str(model_base_path / f"{model_prefix}_direction_model.keras"))
                return_model.save(str(model_base_path / f"{model_prefix}_return_model.keras"))
                joblib.dump(scaler, model_base_path / f"{model_prefix}_scaler.pkl")
                joblib.dump(metadata, model_base_path / f"{model_prefix}_metadata.pkl")
                
                trained_models[symbol] = {
                    'direction_model': str(model_base_path / f"{model_prefix}_direction_model.keras"),
                    'return_model': str(model_base_path / f"{model_prefix}_return_model.keras"),
                    'scaler': str(model_base_path / f"{model_prefix}_scaler.pkl"),
                    'metadata': str(model_base_path / f"{model_prefix}_metadata.pkl")
                }
                
                self.logger.info(f"Successfully trained and saved models for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error training model for {symbol}: {str(e)}")
                continue
        
        if not trained_models:
            self.logger.error("Failed to train any models")
            return {}
            
        return trained_models