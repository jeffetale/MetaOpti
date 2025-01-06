# backtest/backtest_data_fetcher.py

from datetime import datetime, timezone, timedelta
import pandas as pd
import logging
from typing import Optional, Tuple
from config import mt5

class BacktestDataFetcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.TIMEFRAME = mt5.TIMEFRAME_H1
        self.MIN_REQUIRED_PERIODS = 100  # Minimum periods needed for meaningful training

    def validate_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> Tuple[datetime, datetime]:
        """Validate and adjust date range if needed"""
        if not start_date.tzinfo:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if not end_date.tzinfo:
            end_date = end_date.replace(tzinfo=timezone.utc)
            
        # Add buffer to start date to ensure enough training data
        adjusted_start = start_date - timedelta(days=60)  # Extra buffer for training
        
        return adjusted_start, end_date

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch and validate historical data"""
        try:
            start_date, end_date = self.validate_date_range(start_date, end_date)
            
            # Fetch data with correct timeframe
            rates = mt5.copy_rates_range(
                symbol,
                self.TIMEFRAME,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                self.logger.error(
                    f"No data received for {symbol} between {start_date} and {end_date}"
                )
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Validate data sufficiency
            if len(df) < self.MIN_REQUIRED_PERIODS:
                self.logger.error(
                    f"Insufficient data for {symbol}. Got {len(df)} periods, "
                    f"need at least {self.MIN_REQUIRED_PERIODS}"
                )
                return None
                
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def check_data_availability(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """Check if sufficient data is available before proceeding"""
        try:
            df = self.fetch_historical_data(symbol, start_date, end_date)
            return df is not None and len(df) >= self.MIN_REQUIRED_PERIODS
        except Exception:
            return False