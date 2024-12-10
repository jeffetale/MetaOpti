import threading
import statistics
import time
import signal
import sys

from config import initialize_mt5, SYMBOLS, SHUTDOWN_EVENT
from trading import logging, symbol_trader, mt5, trading_state, close_position
from models import TradingStatistics

global trading_stats


def signal_handler(signum, frame):
    """
    Handle termination signals (SIGINT, SIGTERM) to initiate graceful shutdown
    """
    logging.warning("Shutdown signal received. Initiating graceful bot termination...")
    SHUTDOWN_EVENT.set()


def initialize_signal_handling():
    """
    Set up signal handlers for clean bot shutdown
    """
    # Handle Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    # Handle termination signal (SIGTERM)
    signal.signal(signal.SIGTERM, signal_handler)


def close_all_positions():
    """Close all open positions with multiple attempts and forced closures"""   
    # Verify MT5 connection before attempting to close positions
    if not mt5.initialize():
        logging.error("MT5 not initialized during close_all_positions")
        return 0

    max_attempts = 3
    total_profit = 0

    for attempt in range(max_attempts):
        logging.info(f"Attempting to close all positions (Attempt {attempt + 1})")

        # Get current positions
        positions = mt5.positions_get()

        if positions is None:
            logging.error("Failed to retrieve positions")
            continue

        if len(positions) == 0:
            logging.info("No positions to close")
            return total_profit

        # Track closures and profits
        closed_positions = 0
        unclosed_positions = []

        for position in positions:
            # Attempt to close individual position
            if close_position(position):
                total_profit += position.profit
                closed_positions += 1
            else:
                unclosed_positions.append(position)

        # Log closure status
        logging.info(
            f"Positions closed in attempt {attempt + 1}: {closed_positions}/{len(positions)}"
        )

        # Break if all positions closed
        if len(unclosed_positions) == 0:
            break

        # Wait between attempts
        time.sleep(1)

    # Log persistent positions
    if unclosed_positions:
        logging.error(
            f"Could not close {len(unclosed_positions)} positions after {max_attempts} attempts"
        )
        for position in unclosed_positions:
            logging.error(
                f"Persistent position: {position.ticket} for {position.symbol}"
            )

    return total_profit


def main():
    # Set up signal handling
    initialize_signal_handling()

    # Initialize MT5
    if not initialize_mt5():
        logging.error("Failed to initialize MT5")
        return

    # Get initial balance before starting trading
    initial_balance = mt5.account_info().balance

    # Symbol information and selection
    for symbol in SYMBOLS:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            print(f"\nSymbol: {symbol}")
            print(f"Filling Mode: {symbol_info.filling_mode}")
            print(f"Min Volume: {symbol_info.volume_min}")
            print(f"Max Volume: {symbol_info.volume_max}")
            print(f"Trade State: {symbol_info.trade_mode}")

    # Select symbols
    for symbol in SYMBOLS:
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select {symbol}")
            continue
        logging.info(f"Symbol {symbol} selected successfully")

    # Create threads for each symbol
    threads = []
    for symbol in SYMBOLS:
        thread = threading.Thread(target=symbol_trader, args=(symbol,))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    # Initialize trading statistics
    global trading_stats
    trading_stats = TradingStatistics(SYMBOLS)

    # Main monitoring loop
    try:
        while not SHUTDOWN_EVENT.is_set():
            account = mt5.account_info()
            if account:
                account_dict = account._asdict()
                total_profit = account_dict["profit"]

                # Log detailed statistics
                active_symbols = sum(
                    1
                    for state in trading_state.symbol_states.values()
                    if not state.is_restricted
                )
                avg_win_rate = statistics.mean(
                    state.win_rate for state in trading_state.symbol_states.values()
                )

                logging.info("####### Account Status #######")
                logging.info(f"**** Balance: {account_dict['balance']} ****")
                logging.info(f"**** Equity: {account_dict['equity']} ****")
                logging.info(f"**** Profit: {total_profit} ****")
                logging.info(f"**** Active Symbols: {active_symbols} ****")
                logging.info(f"**** Avg Win Rate: {avg_win_rate:.2%} ****")

                # Global state management
                trading_state.global_profit = total_profit

            # Check shutdown event every second
            time.sleep(1)

    except Exception as e:
        logging.error(f"Main loop error: {e}")

    finally:
        # Ensure shutdown happens even if an exception occurs
        shutdown(threads, initial_balance)


def shutdown(threads, initial_balance):
    """Perform clean shutdown of the trading bot"""
    logging.info("Initiating shutdown sequence...")

    # Set shutdown event to stop all trading threads
    SHUTDOWN_EVENT.set()

    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=5)  # Wait up to 5 seconds for each thread

    # Close all positions
    total_profit = close_all_positions()
    logging.info(f"Total profit from closed positions: {total_profit}")

    # Get final balance
    final_balance = mt5.account_info().balance

    # Log final statistics
    global trading_stats
    if trading_stats is None:
        trading_stats = TradingStatistics(SYMBOLS)

    trading_stats.log_final_statistics(initial_balance, final_balance)

    # Shutdown MT5 connection
    mt5.shutdown()
    logging.info("MT5 connection closed")

    # Final status log
    logging.info("Trading bot shutdown complete")

    # Explicitly exit the program
    sys.exit(0)


if __name__ == "__main__":
    main()
