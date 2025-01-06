# utils/market_utils.py

import pandas as pd
import logging
from typing import Optional
from config import mt5
import time


def ensure_mt5_initialized(max_attempts: int = 3, retry_delay: float = 2.0) -> bool:
    for attempt in range(max_attempts):
        if mt5.initialize():
            return True

        if attempt < max_attempts - 1:  # Don't log on last attempt
            logging.warning(
                f"MT5 initialization attempt {attempt + 1} failed, retrying..."
            )
            time.sleep(retry_delay)

    logging.error("Failed to initialize MT5 after multiple attempts")
    return False


def fetch_historical_data(
    symbol: str, timeframe: int, look_back: int
) -> Optional[pd.DataFrame]:
    try:
        if not mt5.initialize():
            if not ensure_mt5_initialized():
                logging.error("Cannot fetch data - MT5 initialization failed")
                return None

        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select symbol {symbol}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, look_back)
        if rates is None:
            logging.error(f"Failed to fetch data for {symbol}")
            return None

        df = pd.DataFrame(rates)
        logging.info(f"Successfully fetched {len(df)} data points for {symbol}")
        return df

    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None