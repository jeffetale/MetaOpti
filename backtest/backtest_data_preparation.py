# backtest/backtest_data_preparation.py

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def prepare_training_data(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """Prepare data for model training with enhanced logging and validation"""
    logger = logging.getLogger(__name__)
    
    try:
        if df.empty:
            logger.error("Empty dataframe provided")
            return None, None, None
            
        logger.info(f"Starting data preparation with {len(df)} initial rows")
        
        # Calculate technical indicators
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        bb = BollingerBands(close=df['close'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # Calculate returns and future returns
        df['returns'] = df['close'].pct_change()
        df['future_returns'] = df['returns'].shift(-1)  # Next period's returns
        
        # Calculate price changes
        df['price_change'] = df['close'] - df['open']
        df['high_low_range'] = df['high'] - df['low']
        
        # Create target variables
        df['direction'] = np.where(df['future_returns'] > 0, 1, 0)
        
        # Drop rows with NaN values
        df_cleaned = df.dropna()
        
        if len(df_cleaned) == 0:
            logger.error("No data remains after cleaning")
            return None, None, None
            
        logger.info(f"Data preparation completed. {len(df_cleaned)} rows remain after cleaning")
        
        # Select features for training
        feature_columns = [
            'sma_20', 'ema_20', 'rsi', 
            'bb_upper', 'bb_lower', 'bb_mid',
            'atr', 'returns', 'price_change', 'high_low_range'
        ]
        
        X = df_cleaned[feature_columns]
        y_direction = df_cleaned['direction']
        y_return = df_cleaned['future_returns']
        
        # Validate final output
        if len(X) < 100:  # Minimum required data points
            logger.warning(f"Insufficient data. Need at least 100 data points.")
            return None, None, None
            
        logger.info(f"Final dataset: {len(X)} samples with {len(feature_columns)} features")
        logger.info(f"Direction distribution: {y_direction.value_counts(normalize=True)}")
        
        return X, y_direction, y_return
        
    except Exception as e:
        logger.error(f"Error in prepare_training_data: {str(e)}")
        return None, None, None

def prepare_prediction_data(df: pd.DataFrame, required_features: list) -> Optional[pd.DataFrame]:
    """Prepare data for prediction with validation"""
    logger = logging.getLogger(__name__)
    
    try:
        if df.empty:
            logger.error("Empty dataframe provided for prediction")
            return None
            
        # Calculate all technical indicators as before
        prepared_data = df.copy()
        prepared_data['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        prepared_data['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        prepared_data['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        bb = BollingerBands(close=df['close'], window=20)
        prepared_data['bb_upper'] = bb.bollinger_hband()
        prepared_data['bb_lower'] = bb.bollinger_lband()
        prepared_data['bb_mid'] = bb.bollinger_mavg()
        
        prepared_data['atr'] = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()
        
        prepared_data['returns'] = df['close'].pct_change()
        prepared_data['price_change'] = df['close'] - df['open']
        prepared_data['high_low_range'] = df['high'] - df['low']
        
        # Drop NaN values
        prepared_data = prepared_data.dropna()
        
        if len(prepared_data) == 0:
            logger.error("No data remains after preparation for prediction")
            return None
            
        # Select only the required features in the correct order
        features_df = prepared_data[required_features].copy()
        
        logger.info(f"Prepared {len(features_df)} samples for prediction")
        return features_df
        
    except Exception as e:
        logger.error(f"Error in prepare_prediction_data: {str(e)}")
        return None