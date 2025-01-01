# backtest/backtest_data_fetcher.py

from config import mt5
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BacktestDataFetcher:
    def __init__(self):
        # Initialize MT5 connection if not already initialized
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

        # Hardcoded config values
        self.TIMEFRAME = mt5.TIMEFRAME_H1
        self.TRAINING_LOOKBACK = 1000
        self.TRAIN_WINDOW = timedelta(days=30)
        self.START_DATE = datetime(2023, 12, 1, tzinfo=timezone.utc)
        self.END_DATE = datetime(2023, 12, 2, tzinfo=timezone.utc)

    def fetch_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for the given symbol and timeframe using MT5's copy_rates_range

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            start_date: Start date for data fetching (must be timezone-aware)
            end_date: End date for data fetching (must be timezone-aware)

        Returns:
            DataFrame with historical data or None if there's an error
        """
        try:
            # Ensure dates are timezone-aware
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            # Fetch data from MT5
            rates = mt5.copy_rates_range(symbol, self.TIMEFRAME, start_date, end_date)

            if rates is None:
                logger.error(f"Failed to fetch data for {symbol}: {mt5.last_error()}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert timestamp to datetime
            df["time"] = pd.to_datetime(df["time"], unit="s")

            # Set index
            df.set_index("time", inplace=True)

            # Basic data validation
            if len(df) < self.TRAINING_LOOKBACK:
                logger.warning(
                    f"Retrieved {len(df)} rows for {symbol}, "
                    f"less than required {self.TRAINING_LOOKBACK}"
                )

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_training_data(
        self, symbol: str, backtest_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch training data for the given symbol up to the backtest date"""  

        # Calculate training period
        training_end = backtest_date
        training_start = training_end - self.TRAIN_WINDOW

        return self.fetch_historical_data(symbol, training_start, training_end)

    def __del__(self):
        """Cleanup MT5 connection on object destruction"""
        mt5.shutdown()
