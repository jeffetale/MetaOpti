# ml_trainer.py

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class MLTrader:
    def __init__(self, symbols, timeframe=mt5.TIMEFRAME_M1, look_back=50):
        self.symbols = symbols
        self.timeframe = timeframe
        self.look_back = look_back
        self.models = {}
        
    def fetch_historical_data(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, 2000) # Fetch 2000 bars
        if rates is None:
            raise ValueError(f"Failed to fetch data for {symbol}")
        return pd.DataFrame(rates)

    
    def feature_engineering(self, df):
        """Advanced feature engineering with expanded indicator set"""
        # Existing technical indicators
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Advanced Momentum Indicators
        df['RSI'] = self._calculate_rsi(df['close'])
        df['MACD'] = self._calculate_macd(df['close'])
        df['Stochastic'] = self._calculate_stochastic(df)
        df['Williams_R'] = self._calculate_williams_r(df)
        
        # Volatility Indicators
        df['ATR'] = self._calculate_atr(df)
        df['Bollinger_Band_Width'] = self._calculate_bollinger_band_width(df)
        
        # Trend Indicators
        df['ADX'] = self._calculate_adx(df)
        df['CCI'] = self._calculate_cci(df)
        
        # Volume-based Indicators
        df['OBV'] = self._calculate_obv(df)
        df['MFI'] = self._calculate_money_flow_index(df)
        
        # Advanced Price Change Features
        df['price_change_1'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_volatility'] = df['price_change_1'].rolling(window=10).std()
        
        # Relative Performance Indicators
        df['relative_strength'] = df['close'] / df['close'].rolling(window=50).mean()
        
        # Predict based on future price movements with multiple thresholds
        df['future_close'] = df['close'].shift(-1)
        df['target_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Use multiple thresholds to create a more balanced classification without leaving lots of neutral data
        df['target_direction'] = np.select(
            [
                df['target_return'] > df['target_return'].quantile(0.7),  # Top 30% positive
                df['target_return'] < df['target_return'].quantile(0.3)   # Bottom 30% negative
            ],
            [1, -1],
            default=0  # Neutral movement
        )
        
        return df

    def _calculate_stochastic(self, df, period=14):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stochastic = 100 * (df['close'] - low_min) / (high_max - low_min)
        return stochastic

    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R Indicator"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        williams_r = (high_max - df['close']) / (high_max - low_min) * -100
        return williams_r

    def _calculate_bollinger_band_width(self, df, period=20, num_std=2):
        """Calculate Bollinger Band Width"""
        rolling_mean = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return (upper_band - lower_band) / rolling_mean

    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        high_diff = pd.Series(df['high']).diff()
        low_diff = -pd.Series(df['low']).diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        true_range = pd.DataFrame({
            'h-l': df['high'] - df['low'],
            'h-pc': np.abs(df['high'] - df['close'].shift()),
            'l-pc': np.abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / true_range.rolling(window=period).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / true_range.rolling(window=period).mean())
        
        adx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return pd.Series(adx).rolling(window=period).mean()

    def _calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        # If 'volume' column doesn't exist, use a default of 1
        if 'volume' not in df.columns:
            df['volume'] = 1
        
        obv = np.where(df['close'] > df['close'].shift(), df['volume'], 
                    np.where(df['close'] < df['close'].shift(), -df['volume'], 0))
        return np.cumsum(obv)

    def _calculate_money_flow_index(self, df, period=14):
        """Calculate Money Flow Index"""
        # If 'volume' column doesn't exist, use a default of 1
        if 'volume' not in df.columns:
            df['volume'] = 1
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        pos_flow = pd.Series(np.where(typical_price > typical_price.shift(), raw_money_flow, 0))
        neg_flow = pd.Series(np.where(typical_price < typical_price.shift(), raw_money_flow, 0))
        
        pos_mf_sum = pos_flow.rolling(window=period).sum()
        neg_mf_sum = neg_flow.rolling(window=period).sum()
        
        money_ratio = pos_mf_sum / neg_mf_sum
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """Calculate Moving Average Convergence Divergence"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return ranges.max(axis=1).rolling(window=period).mean()
    
    def prepare_data(self, symbol):
        df = self.fetch_historical_data(symbol)
        df = self.feature_engineering(df)
        df.dropna(inplace=True)
        
        # Detailed logging of data preparation
        logging.info(f"Total data points for {symbol}: {len(df)}")
        logging.info(f"Class Distribution before filtering: {df['target_direction'].value_counts(normalize=True)}")
        
        # Modified filtering to ensure some data remains
        df_filtered = df[df['target_direction'] != 0]
        
        logging.info(f"Data points after filtering: {len(df_filtered)}")
        logging.info(f"Class Distribution after filtering: {df_filtered['target_direction'].value_counts(normalize=True)}")

        # Safety check for insufficient data
        if len(df_filtered) < 100:  # Minimum threshold for training
            logging.warning(f"Insufficient data for {symbol}. Need at least 100 data points.")
            return None, None, None
        
        # Remap target_direction to binary (0 and 1)
        df_filtered['target_direction_binary'] = np.where(df_filtered['target_direction'] > 0, 1, 0)
        
        # Prepare features and target
        features = [
            'SMA_10', 'SMA_50', 'EMA_20',  # Moving Averages
            'RSI', 'MACD', 'Stochastic', 'Williams_R',  # Momentum
            'ATR', 'Bollinger_Band_Width',  # Volatility
            'ADX', 'CCI',  # Trend
            'OBV', 'MFI',  # Volume
            'price_change_1', 'price_change_5', 'price_change_volatility',
            'relative_strength'
        ]
        
        X = df_filtered[features]
        y_direction = df_filtered['target_direction_binary']
        y_return = df_filtered['target_return']
                
        return X, y_direction, y_return

    def train_models(self):
        for symbol in self.symbols:
            logging.info(f"Training models for {symbol}")
            try:
                X, y_direction, y_return = self.prepare_data(symbol)
                
                # Additional safety checks
                if X is None or len(X) == 0:
                    logging.warning(f"Skipping {symbol} due to insufficient data")
                    continue
                
                # If only one class exists, skip this symbol
                if len(y_direction.unique()) < 2:
                    logging.warning(f"Skipping {symbol} - only one class present")
                    continue
                
                # Apply SMOTE for balancing
                try:
                    smote = SMOTE(random_state=42)
                    X_resampled, y_direction_resampled = smote.fit_resample(X, y_direction)
                    
                    X_train, X_test, y_dir_train, y_dir_test, y_ret_train, y_ret_test = train_test_split(
                        X_resampled, y_direction_resampled, y_return, test_size=0.2, random_state=42
                    )
                    
                    # Log detailed resampled class distribution
                    logging.info(f"Class Distribution after SMOTE:\n{pd.Series(y_direction_resampled).value_counts(normalize=True)}")
                
                except Exception as smote_error:
                    logging.warning(f"SMOTE failed for {symbol}. Falling back to standard split. Error: {smote_error}")
                    X_train, X_test, y_dir_train, y_dir_test, y_ret_train, y_ret_test = train_test_split(
                        X, y_direction, y_return, test_size=0.2, random_state=42
                    )
                    
                    # Log original class distribution
                    logging.info(f"Class Distribution:\n{y_direction.value_counts(normalize=True)}")
                
                # Scale features while preserving column names
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train), 
                    columns=X_train.columns, 
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test), 
                    columns=X_test.columns, 
                    index=X_test.index
                )
                
                # Hyperparameter tuning
                dir_classifier = RandomForestClassifier(
                    n_estimators=200,  # Increased ensemble size
                    max_depth=15,  # Prevent overfitting
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced', 
                    random_state=42
                )
                
                return_predictor = GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=20,
                    loss='huber',  # More robust to outliers
                    random_state=42
                )

                # Train models
                dir_classifier.fit(X_train_scaled, y_dir_train)
                return_predictor.fit(X_train, y_ret_train)
                
                # Evaluate models
                dir_pred = dir_classifier.predict(X_test_scaled)
                ret_pred = return_predictor.predict(X_test_scaled)
                
                logging.info(f"Direction Classification Report for {symbol}:")
                logging.info(classification_report(y_dir_test, dir_pred))
                logging.info(f"Return Prediction MSE: {mean_squared_error(y_ret_test, ret_pred)}")
                
                # Save models and scaler
                os.makedirs('ml_models', exist_ok=True)
                joblib.dump(dir_classifier, f'ml_models/{symbol}_direction_model.pkl')
                joblib.dump(return_predictor, f'ml_models/{symbol}_return_model.pkl')
                joblib.dump(scaler, f'ml_models/{symbol}_scaler.pkl')
                
                # Save metadata
                try:
                    features = X.columns.tolist()
                    model_metadata = {
                        "features": features,
                        "scaler": scaler,
                        "model_params": {
                            "direction_model": dir_classifier.get_params(),
                            "return_model": return_predictor.get_params()
                        }
                    }
                    joblib.dump(model_metadata, f'ml_models/{symbol}_metadata.pkl')
                except Exception as e:
                    logging.error(f"Error saving metadata for {symbol}: {e}")
            
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                continue
        
        logging.info("Model training completed.")

    def analyze_feature_importance(self, symbol):
        X, y_direction, y_return = self.prepare_data(symbol)
        
        # Direction classifier feature importance
        dir_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        dir_classifier.fit(X, y_direction)
        
        # Return predictor feature importance
        return_predictor = GradientBoostingRegressor(n_estimators=150, random_state=42)
        return_predictor.fit(X, y_return)
        
        # Create feature importance DataFrame
        dir_importance = pd.DataFrame({
            'feature': X.columns,
            'direction_importance': dir_classifier.feature_importances_
        }).sort_values('direction_importance', ascending=False)
        
        ret_importance = pd.DataFrame({
            'feature': X.columns,
            'return_importance': return_predictor.feature_importances_
        }).sort_values('return_importance', ascending=False)
        
        # Visualization could be added here
        logging.info("Direction Classifier Feature Importance:")
        logging.info(dir_importance)
        logging.info("\nReturn Predictor Feature Importance:")
        logging.info(ret_importance)
        
        return dir_importance, ret_importance


if __name__ == "__main__":
    # Initialize MT5 connection
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        exit()
    
    # Symbols from config to train models
    symbols = ["EURAUD", "AUDUSD", "GBPJPY", "EURJPY", "USDJPY", "USDCHF", "GBPUSD", "EURUSD", "NZDUSD", "USDSEK", "USDCNH", "USDCAD"]
    
    ml_trainer = MLTrader(symbols)
    ml_trainer.train_models()
    
    mt5.shutdown()