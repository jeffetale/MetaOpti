# main.py

import threading, statistics, time
from config import initialize_mt5, SYMBOLS
from trading import logging, symbol_trader, mt5, trading_state

def close_all_positions():
    """Close all open positions and return total profit/loss"""
    total_profit = 0
    positions = mt5.positions_get()
    
    if positions is None:
        logging.info("No positions to close")
        return 0
        
    for position in positions:
        # Prepare the request to close position
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "close position on shutdown",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send the request
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            profit = position.profit
            total_profit += profit
            logging.info(f"Closed position {position.ticket} for {position.symbol} with profit: {profit}")
        else:
            logging.error(f"Failed to close position {position.ticket}: {result.comment}")
            
    return total_profit

def shutdown():
    """Perform clean shutdown of the trading bot"""
    logging.info("Initiating shutdown sequence...")
    
    # Close all positions
    total_profit = close_all_positions()
    logging.info(f"Total profit from closed positions: {total_profit}")
    
    # Shutdown MT5 connection
    mt5.shutdown()
    logging.info("MT5 connection closed")
    
    # Final status log
    logging.info("Trading bot shutdown complete")

def main():
    if not initialize_mt5():
        return

    for symbol in SYMBOLS:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            print(f"\nSymbol: {symbol}")
            print(f"Filling Mode: {symbol_info.filling_mode}")
            print(f"Min Volume: {symbol_info.volume_min}")
            print(f"Max Volume: {symbol_info.volume_max}")
            print(f"Trade State: {symbol_info.trade_mode}")

            
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
        logging.info(f"Started trading thread for {symbol}")

    # Main monitoring loop
    try:
        while True:
            account = mt5.account_info()
            if account:
                account_dict = account._asdict()
                total_profit = account_dict['profit']
                
                # Log detailed statistics
                active_symbols = sum(1 for state in trading_state.symbol_states.values() if not state.is_restricted)
                avg_win_rate = statistics.mean(state.win_rate for state in trading_state.symbol_states.values())
                
                logging.info(
                    f"Account Status - Balance: {account_dict['balance']}, "
                    f"Equity: {account_dict['equity']}, "
                    f"Profit: {total_profit}, "
                    f"Active Symbols: {active_symbols}, "
                    f"Avg Win Rate: {avg_win_rate:.2%}"
                )
                
                # Global state management
                trading_state.global_profit = total_profit
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("\nBot stopped by user")
        shutdown()  # Call shutdown procedure
    except Exception as e:
        logging.error(f"Main loop error: {e}")
        shutdown()  # Call shutdown procedure even on error
    finally:
        if mt5.initialize():  # Check if MT5 is still connected
            shutdown()  # Final shutdown attempt

if __name__ == "__main__":
    main()