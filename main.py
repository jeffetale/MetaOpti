# main.py

import threading
import statistics
import time
import signal
import sys
import logging
from datetime import datetime
from symbols import SYMBOLS
from config import initialize_mt5, SHUTDOWN_EVENT, mt5
from logging_config import (
    setup_comprehensive_logging,
    log_session_start,
    log_session_end,
)
from models.trading_state import trading_state
from models.trading_statistics import get_next_session_number, TradingStatistics
from trading.order_manager import OrderManager
from trading.position_manager import PositionManager
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager
from ml.predictor import MLPredictor

setup_comprehensive_logging()


class TradingBot:
    def __init__(self):
        self.trading_stats = None
        self.initial_balance = None
        self.threads = []

        # Initialize components
        self.order_manager = OrderManager()
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager(self.order_manager, self.risk_manager)

        # Create predictors for each symbol
        self.ml_predictors = {symbol: MLPredictor(symbol) for symbol in SYMBOLS}
        self.signal_generators = {
            symbol: SignalGenerator(self.ml_predictors[symbol]) for symbol in SYMBOLS
        }

    def symbol_trader(self, symbol):
        """Individual symbol trading logic"""
        logging.info(f"Starting trading thread for {symbol}")

        while not SHUTDOWN_EVENT.is_set():
            try:
                # Manage existing positions
                self.position_manager.manage_open_positions(
                    symbol, trading_state, self.trading_stats
                )

                # Check for new trading opportunities
                if (
                    not SHUTDOWN_EVENT.is_set()
                    and self.risk_manager.should_trade_symbol(symbol)
                ):
                    positions = mt5.positions_get(symbol=symbol)

                    if not positions:
                        signal, atr, potential_profit = self.signal_generators[
                            symbol
                        ].get_signal(symbol)

                        if signal == "neutral" or SHUTDOWN_EVENT.is_set():
                            continue

                        if signal and atr and potential_profit > 0:
                            state = trading_state.symbol_states[symbol]
                            volume = self.risk_manager.calculate_position_size(
                                symbol, atr, trading_state
                            )

                            success = self.order_manager.place_order(
                                symbol,
                                signal,
                                atr,
                                volume,
                                self.trading_stats,
                                is_ml_signal=True,
                            )

                            if success:
                                state.last_trade_time = datetime.now()
                            else:
                                self._handle_failed_trade(state, symbol)

                time.sleep(0.75)

            except Exception as e:
                if not SHUTDOWN_EVENT.is_set():
                    logging.error(f"Error in {symbol} trader: {e}")
                time.sleep(1)

        logging.info(f"Trading thread for {symbol} has stopped")

    def _handle_failed_trade(self, state, symbol):
        """Handle failed trade attempts"""
        state.consecutive_losses += 1
        if state.consecutive_losses >= trading_state.MAX_CONSECUTIVE_LOSSES:
            state.is_restricted = True
            logging.warning(f"{symbol} restricted due to consecutive losses")

    def initialize(self):
        """Initialize trading system"""
        session_number = get_next_session_number()
        log_session_start(session_number)

        initialize_signal_handling()

        if not initialize_mt5():
            logging.error("Failed to initialize MT5")
            return False

        account_info = mt5.account_info()
        self.initial_balance = account_info.balance
        trading_state.update_account_balance(account_info.balance) 

        # Initialize symbols
        if not self._initialize_symbols():
            return False

        self.trading_stats = TradingStatistics(SYMBOLS)
        return True

    def _initialize_symbols(self):
        """Initialize and validate trading symbols"""
        for symbol in SYMBOLS:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                logging.info(
                    f"""
                    Symbol: {symbol}
                    Filling Mode: {symbol_info.filling_mode}
                    Min Volume: {symbol_info.volume_min}
                    Max Volume: {symbol_info.volume_max}
                    Trade State: {symbol_info.trade_mode}
                """
                )

            if not mt5.symbol_select(symbol, True):
                logging.error(f"Failed to select {symbol}")
                continue

            logging.info(f"Symbol {symbol} selected successfully")

        return True

    def start_trading(self):
        """Start trading threads for each symbol"""
        for symbol in SYMBOLS:
            thread = threading.Thread(target=self.symbol_trader, args=(symbol,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def monitor_trading(self):
        """Monitor trading activity and account status"""
        try:
            while not SHUTDOWN_EVENT.is_set():
                account = mt5.account_info()
                if account:
                    self._log_account_status(account)
                time.sleep(1)

        except Exception as e:
            logging.error(f"Main loop error: {e}")

        finally:
            self.shutdown()

    def _log_account_status(self, account):
        """Log detailed account and trading status"""
        account_dict = account._asdict()
        total_profit = account_dict["profit"]

        active_symbols = sum(
            1
            for state in trading_state.symbol_states.values()
            if not state.is_restricted
        )

        avg_win_rate = statistics.mean(
            state.win_rate for state in trading_state.symbol_states.values()
        )

        logging.info(
            """
            ####### Account Status #######
            **** Balance: %s ****
            **** Equity: %s ****
            **** Profit: %s ****
            **** Active Symbols: %s ****
            **** Avg Win Rate: %.2f%% ****
        """,
            account_dict["balance"],
            account_dict["equity"],
            total_profit,
            active_symbols,
            avg_win_rate * 100,
        )

        trading_state.global_profit = total_profit

    def shutdown(self):
        """Perform clean shutdown of the trading bot"""
        logging.info("Initiating shutdown sequence...")

        SHUTDOWN_EVENT.set()

        for thread in self.threads:
            thread.join(timeout=5)

        total_profit = self._close_all_positions()
        logging.info(f"Total profit from closed positions: {total_profit}")

        final_balance = mt5.account_info().balance if mt5.account_info() else None

        if self.trading_stats:
            self.trading_stats.log_final_statistics(self.initial_balance, final_balance)

        session_number = get_next_session_number()
        log_session_end(session_number)

        mt5.shutdown()
        logging.info("MT5 connection closed")
        logging.info("Trading bot shutdown complete")

        sys.exit(0)

    def _close_all_positions(self):
        """Close all open positions with retry mechanism"""
        if not mt5.initialize():
            logging.error("MT5 not initialized during close_all_positions")
            return 0

        total_profit = 0
        max_attempts = 3

        for attempt in range(max_attempts):
            positions = mt5.positions_get()

            if not positions:
                return total_profit

            for position in positions:
                if self.order_manager.close_position(position):
                    total_profit += position.profit

            if not mt5.positions_get():
                break

            time.sleep(1)

        return total_profit


def initialize_signal_handling():
    """Set up signal handlers for clean bot shutdown"""
    signal.signal(signal.SIGINT, lambda signum, frame: signal_handler())
    signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler())


def signal_handler():
    """Handle termination signals for graceful shutdown"""
    logging.warning("Shutdown signal received. Initiating graceful bot termination...")
    SHUTDOWN_EVENT.set()


def main():
    """Main entry point for the trading bot"""
    bot = TradingBot()

    if bot.initialize():
        bot.start_trading()
        bot.monitor_trading()


if __name__ == "__main__":
    main()
