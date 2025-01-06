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
from trading.order_manager import OrderManager
from ml.predictor import MLPredictor
from config import BackTest, mt5, BACKTEST_MODEL_SAVE_DIR
from backtest_data_fetcher import BacktestDataFetcher
from symbols import BACKTEST_SYMBOLS
from backtest_model_trainer import BacktestModelTrainer
from datetime import timezone

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
        initial_balance: float = 10000,
        models: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.models = models or {}
        
        self.data_fetcher = BacktestDataFetcher()

        # Ensure models are trained if not provided
        if not self.models:
            self._train_historical_models()

        try:
            # Initialize predictors in backtest mode
            self.ml_predictors = {}
            for symbol in symbols:
                try:
                    predictor = MLPredictor(
                        symbol=symbol,
                        backtest_mode=True,
                        backtest_date=self.start_date
                    )
                    self.ml_predictors[symbol] = predictor
                except Exception as e:
                    self.logger.error(f"Failed to initialize predictor for {symbol}: {str(e)}")
                    raise

            # Initialize trading components
            self.order_manager = OrderManager()
            self.risk_manager = RiskManager()
            self.position_manager = PositionManager(
                order_manager=self.order_manager,
                risk_manager=self.risk_manager
            )
            self.signal_generators = {
                symbol: SignalGenerator(self.ml_predictors[symbol], backtest_mode=True) 
                for symbol in symbols
            }

            # Trading state
            self.open_positions = {}
            self.closed_positions = []
            self.equity_curve = []
            self.metrics = defaultdict(float)

        except Exception as e:
            self.logger.error(f"Error initializing Backtester: {str(e)}")
            raise

    def _train_historical_models(self) -> None:
        """Train historical models if none were provided"""
        self.logger.info("No pre-trained models provided. Training historical models...")
        
        backtest_config = BackTest(
            START_DATE=self.start_date,
            END_DATE=self.end_date
        )
        
        trainer = BacktestModelTrainer(
            symbols=self.symbols,
            backtest_config=backtest_config
        )
        
        self.models = trainer.train_historical_models()
        if not self.models:
            raise RuntimeError("Failed to train historical models")
        
    def _fetch_backtest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for backtesting"""
        return self.data_fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=self.start_date,
            end_date=self.end_date,
        )

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

            # Check if any symbol data is None or empty
            for symbol, data in symbol_data.items():
                if data is None or data.empty:
                    self.logger.error(f"Missing or empty data for symbol {symbol}")
                    return False

            # Initialize metrics
            self.metrics = defaultdict(float)
            self.metrics['total_trades'] = 0
            self.metrics['profitable_trades'] = 0
            self.metrics['total_profit'] = 0.0
            self.equity_curve = [self.initial_balance]

            # Get the common time index across all symbols
            common_times = set.intersection(*[set(df.index) for df in symbol_data.values()])
            time_index = sorted(common_times)

            # Main simulation loop
            for current_time in time_index:
                # Update current data point for each symbol
                current_data = {
                    symbol: df.loc[df.index == current_time] 
                    for symbol, df in symbol_data.items()
                }

                # Process open positions
                self._process_open_positions(current_data, current_time)

                # Generate new signals
                for symbol in self.symbols:
                    if symbol not in self.open_positions:
                        signal = self.signal_generators[symbol].get_signal(
                            symbol=symbol,
                            current_time=current_time
                        )
                        
                        if signal and signal != "hold":
                            # Calculate position size
                            price = current_data[symbol]['close'].iloc[0]
                            volume = self.risk_manager.calculate_position_size(
                                symbol=symbol,
                                direction=signal,
                                current_price=price,
                                balance=self.current_balance
                            )

                            # Calculate SL/TP levels
                            sl, tp = self.risk_manager.calculate_sl_tp(
                                symbol=symbol,
                                direction=signal,
                                entry_price=price,
                                atr=current_data[symbol]['atr'].iloc[0]
                            )

                            # Open new position
                            position = BacktestPosition(
                                symbol=symbol,
                                entry_price=price,
                                direction=signal,
                                volume=volume,
                                sl=sl,
                                tp=tp,
                                entry_time=current_time
                            )
                            self.open_positions[symbol] = position

                # Update equity curve
                total_equity = self._calculate_current_equity(current_data)
                self.equity_curve.append(total_equity)

            # Generate final metrics
            self._calculate_final_metrics()
            self._generate_report()
            return True

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return False

    def _process_open_positions(self, current_data: dict, current_time: datetime) -> None:
        """Process all open positions"""
        positions_to_close = []

        for symbol, position in self.open_positions.items():
            current_price = current_data[symbol]['close'].iloc[0]
            
            # Check for SL/TP hits
            if position.direction == "buy":
                if current_price <= position.sl:
                    position.exit_reason = "stop_loss"
                    positions_to_close.append(symbol)
                elif current_price >= position.tp:
                    position.exit_reason = "take_profit"
                    positions_to_close.append(symbol)
            else:  # sell position
                if current_price >= position.sl:
                    position.exit_reason = "stop_loss"
                    positions_to_close.append(symbol)
                elif current_price <= position.tp:
                    position.exit_reason = "take_profit"
                    positions_to_close.append(symbol)

        # Close positions and update metrics
        for symbol in positions_to_close:
            self._close_position(symbol, current_data[symbol]['close'].iloc[0], current_time)

    def _close_position(self, symbol: str, exit_price: float, exit_time: datetime) -> None:
        """Close a position and update metrics"""
        position = self.open_positions[symbol]
        position.exit_price = exit_price
        position.exit_time = exit_time
        
        # Calculate profit/loss
        if position.direction == "buy":
            position.profit = (exit_price - position.entry_price) * position.volume
        else:
            position.profit = (position.entry_price - exit_price) * position.volume
        
        # Update metrics
        self.metrics['total_trades'] += 1
        if position.profit > 0:
            self.metrics['profitable_trades'] += 1
        self.metrics['total_profit'] += position.profit
        self.current_balance += position.profit
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.open_positions[symbol]

    def _calculate_current_equity(self, current_data: dict) -> float:
        """Calculate current equity including unrealized P/L"""
        equity = self.current_balance
        
        # Add unrealized profits/losses from open positions
        for symbol, position in self.open_positions.items():
            current_price = current_data[symbol]['close'].iloc[0]
            if position.direction == "buy":
                unrealized_pnl = (current_price - position.entry_price) * position.volume
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.volume
            equity += unrealized_pnl
        
        return equity

    def _calculate_final_metrics(self) -> None:
        """Calculate final performance metrics"""
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['profitable_trades'] / self.metrics['total_trades']
            self.metrics['avg_profit'] = self.metrics['total_profit'] / self.metrics['total_trades']
            
            # Calculate max drawdown
            peak = self.initial_balance
            max_drawdown = 0
            for equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            self.metrics['max_drawdown'] = max_drawdown
            
            # Calculate profit factor
            total_gains = sum(pos.profit for pos in self.closed_positions if pos.profit > 0)
            total_losses = abs(sum(pos.profit for pos in self.closed_positions if pos.profit < 0))
            self.metrics['profit_factor'] = total_gains / total_losses if total_losses > 0 else float('inf')

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
        # Create backtest config
        backtest_config = BackTest(
            START_DATE=datetime(2023, 12, 1, tzinfo=timezone.utc),
            END_DATE=datetime(2023, 12, 2, tzinfo=timezone.utc)
        )

        # Initialize trainer with config
        backtest_trainer = BacktestModelTrainer(
            symbols=BACKTEST_SYMBOLS,
            backtest_config=backtest_config
        )
        
        # Train models
        historical_models = backtest_trainer.train_historical_models()

        # Run backtest
        backtester = Backtester(
            symbols=BACKTEST_SYMBOLS,
            start_date=backtest_config.START_DATE,
            end_date=backtest_config.END_DATE,
            initial_balance=10000,
            models=historical_models
        )

        backtester.run_backtest()

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise

    finally:
        mt5.shutdown()
