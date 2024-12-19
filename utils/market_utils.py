import pandas as pd
import logging
from typing import Optional
from config import mt5


def fetch_historical_data(
    symbol: str, timeframe: int, look_back: int
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data from MT5

    Args:
        symbol: Trading symbol
        timeframe: MT5 timeframe constant
        look_back: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, look_back)
        if rates is None:
            logging.error(f"Failed to fetch data for {symbol}")
            return None
        return pd.DataFrame(rates)
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None
