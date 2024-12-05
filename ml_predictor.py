# ml_predictor.py

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import joblib
import logging
from typing import Tuple, Optional

class MLPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.direction_model = None
        self.return_model = None
        self.scaler = None
        self.features = None
        
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models for a specific symbol"""
        try:
            self.direction_model = joblib.load(f'ml_models/{self.symbol}_direction_model.pkl')
            self.return_model = joblib.load(f'ml_models/{self.symbol}_return_model.pkl')
            self.scaler = joblib.load(f'ml_models/{self.symbol}_scaler.pkl')
            self.features = joblib.load(f'ml_models/{self.symbol}_metadata.pkl')
            logging.info(f"Models loaded for {self.symbol}")
        except FileNotFoundError:
            logging.error(f"Models for {self.symbol} not found. Train models first.")
            return None
    
    def extract_features(self, rates_frame: pd.DataFrame) -> np.ndarray:
        """Extract features from price data"""
        # Technical indicators
        rates_frame['SMA_10'] = rates_frame['close'].rolling(window=10).mean()
        rates_frame['SMA_50'] = rates_frame['close'].rolling(window=50).mean()
        
        # Momentum indicators
        rates_frame['RSI'] = self._calculate_rsi(rates_frame['close'])
        rates_frame['MACD'] = self._calculate_macd(rates_frame['close'])
        
        # Volatility
        rates_frame['ATR'] = self._calculate_atr(rates_frame)
        
        # Price changes
        rates_frame['price_change_1'] = rates_frame['close'].pct_change()
        rates_frame['price_change_5'] = rates_frame['close'].pct_change(5)
        
        features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'ATR', 
                    'price_change_1', 'price_change_5']
        
        return rates_frame[features].iloc[-1].values
    
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
    
    def predict(self, timeframe=mt5.TIMEFRAME_M1, look_back=50, threshold=0.6) -> Tuple[Optional[str], float, float]:
        if not all([self.direction_model, self.return_model, self.scaler]):
            logging.error("Models not loaded. Cannot predict.")
            return None, 0, 0
        
        # Fetch latest price data
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, look_back)
        if rates is None:
            logging.error(f"Failed to fetch rates for {self.symbol}")
            return None, 0, 0
        
        rates_frame = pd.DataFrame(rates)
        
        try:
            # Extract features
            features = self.extract_features(rates_frame)
            
            # Ensure features match exactly the training columns
            features_df = pd.DataFrame([features], columns=self.features["features"])
            scaled_features = self.scaler.transform(features_df)
            
            # Rest of the prediction code remains the same
            direction_prob = self.direction_model.predict_proba(scaled_features)[0]
            predicted_return = self.return_model.predict(scaled_features)[0]
            
            # Interpret predictions
            if direction_prob[1] > threshold:
                signal = "buy"
            elif direction_prob[0] > threshold:
                signal = "sell"
            else:
                signal = "hold"
            
            return signal, direction_prob[1], predicted_return

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None, 0, 0

