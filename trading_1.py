# trading.py

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
import logging
from config import INITIAL_VOLUME, MIN_PROFIT_THRESHOLD, MIN_WIN_RATE, TIMEFRAME, PROFIT_LOCK_PERCENTAGE, MAX_CONSECUTIVE_LOSSES
from models import trading_state

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

def calculate_win_rate(trades):
    if not trades:
        return 0
    winning_trades = sum(1 for profit in trades if profit > 0)
    return winning_trades / len(trades)

def adjust_trading_parameters(symbol, profit):
    """Dynamically adjust trading parameters based on performance"""
    state = trading_state.symbol_states[symbol]
    state.trades_count += 1
    state.trades_history.append(profit)
    
    # Update win rate
    state.win_rate = calculate_win_rate(state.trades_history[-20:])  # Consider last 20 trades
    
    # Adjust volume based on performance
    if state.win_rate > 0.6:  # Increase volume if winning consistently
        state.volume = min(state.volume * 1.2, INITIAL_VOLUME * 2)
    elif state.win_rate < 0.4:  # Decrease volume if losing
        state.volume = max(state.volume * 0.8, INITIAL_VOLUME * 0.5)
    
    # Adjust profit threshold based on volatility
    if profit > state.profit_threshold:
        state.profit_threshold *= 1.1  # Increase threshold on good performance
    elif profit < 0:
        state.profit_threshold = max(MIN_PROFIT_THRESHOLD, state.profit_threshold * 0.9)

def should_trade_symbol(symbol):
    """Determine if we should trade a symbol based on its performance"""
    state = trading_state.symbol_states[symbol]
    
    if state.is_restricted:
        # Check if enough time has passed to retry
        if state.last_trade_time and (datetime.now() - state.last_trade_time).hours < 1:
            return False
        
        # Reset restriction if conditions improve
        if state.win_rate > MIN_WIN_RATE:
            state.is_restricted = False
            logging.info(f"{symbol} restrictions lifted due to improved performance")
            return True
        return False
    
    return True

def get_market_volatility(symbol, timeframe=mt5.TIMEFRAME_M1, periods=20):
    """Calculate current market volatility"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, periods)
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    return df['high'].max() - df['low'].min()

def calculate_indicators(df, symbol):
    """Enhanced indicator calculation with volatility consideration"""
    params = trading_state.ta_params
    
    # Basic indicators
    df['SMA_short'] = df['close'].rolling(window=params.sma_short).mean()
    df['SMA_long'] = df['close'].rolling(window=params.sma_long).mean()
    
    # Enhanced RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=params.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params.rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility indicators
    df['ATR'] = df['high'].rolling(window=params.atr_period).max() - df['low'].rolling(window=params.atr_period).min()
    df['Volatility'] = df['close'].rolling(window=10).std()
    
    # Trend strength
    df['ADX'] = calculate_adx(df)
    
    return df

def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    df['TR'] = pd.DataFrame({
        'HL': (df['high'] - df['low']).abs(),
        'HD': (df['high'] - df['close'].shift(1)).abs(),
        'LD': (df['low'] - df['close'].shift(1)).abs()
    }).max(axis=1)
    
    df['+DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
    df['-DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
    
    df['+DI'] = 100 * (df['+DM'].rolling(window=period).mean() / df['TR'].rolling(window=period).mean())
    df['-DI'] = 100 * (df['-DM'].rolling(window=period).mean() / df['TR'].rolling(window=period).mean())
    
    df['DX'] = 100 * ((df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI']))
    return df['DX'].rolling(window=period).mean()

def get_signal(symbol):
    """Enhanced signal generation with multiple confirmations"""
    if not should_trade_symbol(symbol):
        return None, None, 0
    
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 50)
    if rates is None:
        return None, None, 0
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = calculate_indicators(df, symbol)
    
    current = df.iloc[-1]
    
    # Skip if volatility is too low
    if current.Volatility < trading_state.ta_params.volatility_threshold:
        return None, None, 0
    
    # Calculate trend strength and quality
    trend_score = 0
    
    # ADX trend strength
    if current.ADX > 25:  # Strong trend
        trend_score += 2
    
    # Moving average alignment
    if current.close > current.SMA_short > current.SMA_long:
        trend_score += 1
    elif current.close < current.SMA_short < current.SMA_long:
        trend_score -= 1
    
    # RSI extremes with trend confirmation
    if current.RSI < trading_state.ta_params.rsi_oversold and trend_score > 0:
        trend_score += 2
    elif current.RSI > trading_state.ta_params.rsi_overbought and trend_score < 0:
        trend_score -= 2
    
    # Calculate potential profit based on ATR and trend strength
    potential_profit = current.ATR * abs(trend_score) * 10
    
    # Conservative mode adjustments
    if trading_state.is_conservative_mode:
        required_score = 3  # Higher requirement in conservative mode
        potential_profit *= 0.8  # Reduce expected profit for safety
    else:
        required_score = 2
    
    logging.info(f"{symbol} - Trend Score: {trend_score}, "
                f"ADX: {current.ADX:.2f}, RSI: {current.RSI:.2f}, "
                f"Potential Profit: {potential_profit:.5f}")
    
    if trend_score >= required_score:
        return "buy", current.ATR, potential_profit
    elif trend_score <= -required_score:
        return "sell", current.ATR, potential_profit
    
    return None, None, 0

def manage_open_positions(symbol):
    """Smart position management with trailing stops"""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    state = trading_state.symbol_states[symbol]
    
    for position in positions:
        current_profit = position.profit
        state.total_profit = current_profit
        
        # Update max profit if current profit is higher
        if current_profit > state.max_profit:
            state.max_profit = current_profit
        
        # Check if profit exceeds threshold
        if current_profit >= state.profit_threshold:
            trading_state.is_conservative_mode = True
            
            # Calculate trailing stop threshold
            trailing_stop = state.max_profit * PROFIT_LOCK_PERCENTAGE
            
            # Close position if profit falls below trailing stop
            if current_profit < trailing_stop:
                close_position(position)
                logging.info(f"{symbol} position closed to lock in profit: {current_profit}")
                adjust_trading_parameters(symbol, current_profit)
        
        # Check for stop loss adjustment
        elif current_profit < 0:
            # Tighten stop loss if in significant loss
            if current_profit < -state.profit_threshold * 0.5:
                modify_stop_loss(position)

def close_position(position):
    """Close an open position"""
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "close position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE

def modify_stop_loss(position):
    """Modify position's stop loss"""
    new_sl = position.price_open if position.profit < 0 else position.sl
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl": new_sl,
        "tp": position.tp
    }
    
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE

def place_order(symbol, direction, atr, volume):
    """
    Place a trading order with dynamic stop loss and take profit based on ATR
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return False
        
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get price for {symbol}")
        return False
    
    # Adjust volume to meet minimum requirements
    volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
    
    # Calculate order parameters
    sl_distance = atr * 1.5  # Stop loss at 1.5 * ATR
    tp_distance = atr * 2.5  # Take profit at 2.5 * ATR
    
    if direction == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - sl_distance
        tp = price + tp_distance
    else:  # sell
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + sl_distance
        tp = price - tp_distance
    
    # Use the default filling mode based on what we see in the symbol info
    # From your log, filling_mode is 1 for all symbols
    filling_type = mt5.ORDER_FILLING_FOK
    
    # Prepare the trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": f"python script {direction}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    
    # Send the order
    result = mt5.order_send(request)
    
    if result is None:
        logging.error(f"Order failed for {symbol}: No result returned")
        return False
        
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order failed for {symbol}: {result.comment} (Error code: {result.retcode})")
        logging.info(f"Symbol {symbol} filling mode: {symbol_info.filling_mode}")
        
        # Try alternative filling mode if first attempt fails
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Second attempt failed for {symbol}")
            return False
    
    logging.info(f"Order placed successfully for {symbol}: {direction.upper()} "
                f"Volume: {volume}, Price: {price}, SL: {sl}, TP: {tp}")
    return True

def symbol_trader(symbol):
    """Enhanced symbol trading loop"""
    while True:
        try:
            # Manage existing positions
            manage_open_positions(symbol)
            
            # Check if we should look for new trades
            if should_trade_symbol(symbol):
                positions = mt5.positions_get(symbol=symbol)
                
                if not positions:  # No open positions for this symbol
                    signal, atr, potential_profit = get_signal(symbol)
                    if signal and atr and potential_profit > 0:
                        state = trading_state.symbol_states[symbol]
                        success = place_order(symbol, signal, atr, state.volume)
                        
                        if success:
                            state.last_trade_time = datetime.now()
                        else:
                            state.consecutive_losses += 1
                            
                            if state.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                                state.is_restricted = True
                                logging.warning(f"{symbol} restricted due to consecutive losses")
            
            time.sleep(0.1)  # Check every 100ms
            
        except Exception as e:
            logging.error(f"Error in {symbol} trader: {e}")
            time.sleep(1)

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