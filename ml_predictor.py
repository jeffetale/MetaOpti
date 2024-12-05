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
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models for a specific symbol"""
        try:
            self.direction_model = joblib.load(f'ml_models/{self.symbol}_direction_model.pkl')
            self.return_model = joblib.load(f'ml_models/{self.symbol}_return_model.pkl')
            self.scaler = joblib.load(f'ml_models/{self.symbol}_scaler.pkl')
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
        # (same as in MLTrader)
    
    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """Calculate Moving Average Convergence Divergence"""
        # (same as in MLTrader)
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        # (same as in MLTrader)
    
    def predict(self, timeframe=mt5.TIMEFRAME_M1, look_back=50) -> Tuple[Optional[str], float, float]:
        """Predict trading signal and potential return"""
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
            # Extract and scale features
            features = self.extract_features(rates_frame)
            scaled_features = self.scaler.transform([features])
            
            # Predict direction and return
            direction_prob = self.direction_model.predict_proba(scaled_features)[0]
            predicted_return = self.return_model.predict(scaled_features)[0]
            
            # Interpret predictions
            if direction_prob[1] > 0.6:  # High confidence in upward movement
                signal = "buy"
            elif direction_prob[0] > 0.6:  # High confidence in downward movement
                signal = "sell"
            else:
                signal = None
            
            return signal, direction_prob[1], predicted_return
        
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None, 0, 0

# Example of how to use in your main trading script
def enhance_signal_with_ml(symbol):
    ml_predictor = MLPredictor(symbol)
    ml_signal, ml_confidence, ml_return = ml_predictor.predict()
    
    # You can incorporate ML predictions into your existing signal generation
    # For example, adjust signal strength or bias based on ML confidence
    
    return ml_signal, ml_confidence, ml_return