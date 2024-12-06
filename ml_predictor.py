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
            # Load metadata first to get feature names
            metadata = joblib.load(f'ml_models/{self.symbol}_metadata.pkl')
            self.features = metadata.get('features', [])
            
            self.direction_model = joblib.load(f'ml_models/{self.symbol}_direction_model.pkl')
            self.return_model = joblib.load(f'ml_models/{self.symbol}_return_model.pkl')
            self.scaler = joblib.load(f'ml_models/{self.symbol}_scaler.pkl')
            
            logging.info(f"Models loaded for {self.symbol}")
            logging.info(f"Features: {self.features}")
        except FileNotFoundError:
            logging.error(f"Models for {self.symbol} not found. Train models first.")
            return None
    
    def extract_features(self, rates_frame: pd.DataFrame) -> pd.DataFrame:
        """Extract features from price data"""
        # Technical indicators
        rates_frame['SMA_10'] = rates_frame['close'].rolling(window=10).mean()
        rates_frame['SMA_50'] = rates_frame['close'].rolling(window=50).mean()
        rates_frame['EMA_20'] = rates_frame['close'].ewm(span=20, adjust=False).mean()
        
        # Momentum indicators
        rates_frame['RSI'] = self._calculate_rsi(rates_frame['close'])
        rates_frame['MACD'] = self._calculate_macd(rates_frame['close'])
        rates_frame['Stochastic'] = self._calculate_stochastic(rates_frame)
        rates_frame['Williams_R'] = self._calculate_williams_r(rates_frame)
        
        # Volatility
        rates_frame['ATR'] = self._calculate_atr(rates_frame)
        rates_frame['Bollinger_Band_Width'] = self._calculate_bollinger_band_width(rates_frame)
        
        # Trend indicators
        rates_frame['ADX'] = self._calculate_adx(rates_frame)
        rates_frame['CCI'] = self._calculate_cci(rates_frame)
        
        # Volume-based indicators
        rates_frame['OBV'] = self._calculate_obv(rates_frame)
        rates_frame['MFI'] = self._calculate_money_flow_index(rates_frame)
        
        # Price changes
        rates_frame['price_change_1'] = rates_frame['close'].pct_change()
        rates_frame['price_change_5'] = rates_frame['close'].pct_change(5)
        rates_frame['price_change_volatility'] = rates_frame['price_change_1'].rolling(window=10).std()
        
        # Relative performance
        rates_frame['relative_strength'] = rates_frame['close'] / rates_frame['close'].rolling(window=50).mean()
        
        # Select and order features exactly as during training
        features_df = rates_frame[self.features].iloc[-1].to_frame().T
        
        return features_df
    
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
    
    def _calculate_stochastic(self, df, period=14):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stochastic = 100 * (df['close'] - low_min) / (high_max - low_min)
        return stochastic

    def _calculate_williams_r(self, df, period=14):
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        williams_r = (high_max - df['close']) / (high_max - low_min) * -100
        return williams_r

    def _calculate_bollinger_band_width(self, df, period=20, num_std=2):
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
    
    def predict(self, timeframe=mt5.TIMEFRAME_M1, look_back=50, threshold=0.6) -> Tuple[Optional[str], float, float]:
        """Predict trading signal and potential return"""
        if not all([self.direction_model, self.return_model, self.scaler, self.features]):
            logging.error("Models not loaded. Cannot predict.")
            return None, 0, 0
        
        # Fetch latest price data
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, look_back)
        if rates is None:
            logging.error(f"Failed to fetch rates for {self.symbol}")
            return None, 0, 0
        
        rates_frame = pd.DataFrame(rates)
        
        try:
            # Extract features ensuring exact column names and order
            features_df = self.extract_features(rates_frame)
            
            # Ensure column names match exactly
            features_df.columns = self.features

            #scale features
            scaled_features = self.scaler.transform(features_df)
            
            # Predict direction and return
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

