# backtest/backtest.py

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from logging_config import setup_comprehensive_logging
from collections import defaultdict

from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager
from ml.predictor import MLPredictor
from config import TRADING_CONFIG, BackTest, mt5
from utils.market_utils import fetch_historical_data
from symbols import BACKTEST_SYMBOLS
from backtest_model_trainer import BacktestMLTrainer

setup_comprehensive_logging()

class BacktestPosition:
    def __init__(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        volume: float,
        sl: float,
        tp: float,
        entry_time: datetime,
    ):
        self.symbol = symbol
        self.entry_price = entry_price
        self.direction = direction
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.profit: float = 0
        self.status: str = "open"
        self.exit_reason: Optional[str] = None


class Backtester:
    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = BackTest.INITIAL_BALANCE,
        models: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.models = models or {}  # Initialize empty dict if None

        # Ensure models are trained if not provided
        if not self.models:
            self._train_historical_models()

        # Initialize predictors in backtest mode
        self.ml_predictors = {
            symbol: MLPredictor(
                symbol=symbol,
                model=self.models.get(symbol),  # Pass model if available
                backtest_mode=True,
                backtest_date=self.start_date,
            )
            for symbol in symbols
        }

        self.signal_generators = {
            symbol: SignalGenerator(self.ml_predictors[symbol]) for symbol in symbols
        }
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()

        # Trading state
        self.open_positions: Dict[str, BacktestPosition] = {}
        self.closed_positions: List[BacktestPosition] = []
        self.equity_curve: List[float] = []

        # Performance metrics
        self.metrics = defaultdict(float)
        # self.logger = logging.getLogger(__name__)

    def _train_historical_models(self) -> None:
        """Train historical models if none were provided"""
        self.logger.info(
            "No pre-trained models provided. Training historical models..."
        )
        trainer = BacktestMLTrainer(symbols=self.symbols, backtest_date=self.start_date)
        self.models = trainer.train_historical_models()
        if not self.models:
            raise RuntimeError("Failed to train historical models")

    def run_backtest(self) -> bool:
        """Run the backtest simulation"""
        try:
            self.logger.info(
                f"Starting backtest from {self.start_date} to {self.end_date}"
            )

            # Fetch data for all symbols
            symbol_data = {
                symbol: self._fetch_backtest_data(symbol) for symbol in self.symbols
            }

            if not all(symbol_data.values()):
                self.logger.error("Missing data for some symbols")
                return False

            # Rest of the run_backtest implementation remains the same...
            return True

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return False

    def _fetch_backtest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for backtesting"""
        return fetch_historical_data(
            symbol=symbol,
            timeframe=BackTest.TIMEFRAME,
            look_back=BackTest.PREDICTION_LOOKBACK,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def _generate_report(self):
        """Generate and print backtest report"""
        report = f"""
        ====== Backtest Report ======
        Period: {self.start_date} to {self.end_date}
        Initial Balance: ${self.initial_balance:,.2f}
        Final Balance: ${self.current_balance:,.2f}
        
        Performance Metrics:
        - Total Return: {((self.current_balance/self.initial_balance - 1) * 100):.2f}%
        - Total Trades: {self.metrics['total_trades']}
        - Win Rate: {self.metrics['win_rate']*100:.2f}%
        - Profit Factor: {self.metrics['profit_factor']:.2f}
        - Average Profit: ${self.metrics['avg_profit']:,.2f}
        - Maximum Drawdown: {self.metrics['max_drawdown']*100:.2f}%
        
        Trade Statistics:
        - Profitable Trades: {self.metrics['profitable_trades']}
        - Loss-Making Trades: {self.metrics['total_trades'] - self.metrics['profitable_trades']}
        - Total Profit: ${self.metrics['total_profit']:,.2f}
        """

        self.logger.info(report)
        return report


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize MT5
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        exit()

    try:
        # Train historical models
        backtest_trainer = BacktestMLTrainer(
            symbols=BACKTEST_SYMBOLS, backtest_date=BackTest.START_DATE
        )
        historical_models = backtest_trainer.train_historical_models()

        # Run backtest
        backtester = Backtester(
            symbols=BACKTEST_SYMBOLS,
            start_date=BackTest.START_DATE,
            end_date=BackTest.END_DATE,
            initial_balance=BackTest.INITIAL_BALANCE,
            models=historical_models,
        )

        backtester.run_backtest()

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise

    finally:
        mt5.shutdown()
