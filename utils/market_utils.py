# utils/market_utils.py

import pandas as pd
import logging
from typing import Optional
from datetime import datetime, timezone
import pytz
from config import mt5
import time


def ensure_mt5_initialized(max_attempts: int = 3, retry_delay: float = 2.0) -> bool:
    for attempt in range(max_attempts):
        if mt5.initialize():
            return True
        if attempt < max_attempts - 1:
            logging.warning(
                f"MT5 initialization attempt {attempt + 1} failed, retrying..."
            )
            time.sleep(retry_delay)
    logging.error("Failed to initialize MT5 after multiple attempts")
    return False

def datetime_to_mt5_timestamp(dt: datetime) -> datetime:
    """Convert datetime to MT5-compatible format"""
    return prepare_datetime_for_mt5(dt)


def fetch_historical_data(
    symbol: str,
    timeframe: int,
    look_back: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data with flexible date range support
    """
    logger = logging.getLogger(__name__)

    try:
        if not mt5.initialize():
            if not ensure_mt5_initialized():
                logger.error("Cannot fetch data - MT5 initialization failed")
                return None

        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None

        if start_date and end_date:
            # Convert dates to proper UTC datetime objects for MT5
            start_dt = prepare_datetime_for_mt5(start_date)
            end_dt = prepare_datetime_for_mt5(end_date)

            logger.info(f"Fetching {symbol} data from {start_dt} to {end_dt}")

            # Ensure both dates are passed as datetime objects
            rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
        else:
            logger.info(f"Fetching last {look_back} candles for {symbol}")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, look_back)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch data for {symbol}")
            return None

        # Convert the rates to a DataFrame
        df = pd.DataFrame(rates)

        # Convert time column to datetime with UTC timezone
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)

        logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


def prepare_datetime_for_mt5(dt: datetime) -> datetime:     
    """Prepare datetime object for MT5 API calls"""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))

    # Convert to UTC if not already
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)

    # MT5 requires naive datetime objects
    return dt.replace(tzinfo=None)
