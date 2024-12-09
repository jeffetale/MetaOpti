# trading.py

# import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
import logging
from config import INITIAL_VOLUME, MIN_PROFIT_THRESHOLD, MIN_WIN_RATE, TIMEFRAME, PROFIT_LOCK_PERCENTAGE, MAX_CONSECUTIVE_LOSSES, mt5, GAIN_THRESHOLD, LOSS_THRESHOLD
from models import trading_state
from ml_predictor import MLPredictor

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

    # Add recent trade direction to memory
    state.recent_trade_directions.append('buy' if profit > 0 else 'sell')
    
    # Limit memory size
    if len(state.recent_trade_directions) > state.trade_direction_memory_size:
        state.recent_trade_directions.pop(0)
    
    # Update consecutive losses and total profit
    state.trades_count += 1
    state.trades_history.append(profit)
    
    # Update win rate
    state.win_rate = calculate_win_rate(state.trades_history[-10:])  # Consider last 10 trades
    
    # Adjust volume based on performance
    if state.win_rate > 0.6:  # Increase volume if winning consistently
        state.volume = min(state.volume * 1.2, INITIAL_VOLUME * 2)  #double volume
    elif state.win_rate < 0.4:  # Decrease volume if losing
        state.volume = max(state.volume * 0.8, INITIAL_VOLUME * 0.5)  #half volume
    
    # Adjust profit threshold based on volatility
    if profit > state.profit_threshold:
        state.profit_threshold *= 1.1  # Increase threshold on good performance
    elif profit < 0:
        state.profit_threshold = max(MIN_PROFIT_THRESHOLD, state.profit_threshold * 0.9)

def should_trade_symbol(symbol):
    """Determine if we should trade a symbol based on its performance"""
    state = trading_state.symbol_states[symbol]

    # if state.last_trade_time:
    #     cooling_period = datetime.now() - state.last_trade_time
    #     if cooling_period.total_seconds() < 120:  # 2-minute cooling period
    #         return False
        
    # Check recent trade performance
    recent_trades = state.trades_history[-5:]  # Last 5 trades
    if recent_trades:
        recent_performance = sum(recent_trades)
        
        # If recent trades have been consistently losing
        if recent_performance < 0:
            if state.last_trade_time and (datetime.now() - state.last_trade_time).total_seconds() < 120:  # 2-minute cooling period
                logging.info(f"{symbol} trade suppressed due to recent poor performance")
                return False
        
        # Prevent trading if recent trades are too volatile
        trade_variance = max(recent_trades) - min(recent_trades)
        if trade_variance > state.profit_threshold * 2:
            logging.info(f"{symbol} trade suppressed due to high trade volatility")
            return False
    
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

def calculate_trend_score(current):
    """Calculate trend score based on traditional indicators"""
    trend_score = 0

    # ADX trend strength
    if current.ADX > 25:
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

    return trend_score


def get_signal(symbol):
    if not should_trade_symbol(symbol):
        return None, None, 0

    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 50)
    if rates is None:
        logging.warning(f"No rates available for {symbol}")
        return None, None, 0

    df = pd.DataFrame(rates)
    df = calculate_indicators(df, symbol)
    current = df.iloc[-1]

    ml_predictor = MLPredictor(symbol)
    ml_signal, ml_confidence, _ = ml_predictor.predict()

    if ml_confidence < 0.6:
        logging.info(
            f"{symbol}: ***** Neutral hold due to low ML confidence ({ml_confidence}) *****"
        )
        return "hold", 0, 0

    return ml_signal, current.ATR, current.ATR * abs(ml_confidence)


def manage_open_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    state = trading_state.symbol_states[symbol]
    for position in positions:
        current_profit = position.profit
        current_time = datetime.now()
        position_open_time = datetime.fromtimestamp(position.time)
        time_since_open = (current_time - position_open_time).total_seconds()

        # Check if ML recommends a switch or hold
        ml_predictor = MLPredictor(symbol)
        ml_signal, ml_confidence, _ = ml_predictor.predict()

        # Close and switch positions based on conditions
        if current_profit <= LOSS_THRESHOLD:
            if ml_confidence > 0.6 and ml_signal:
                logging.info(
                    f"{symbol}: Closing {position.type} due to loss and opening {ml_signal}"
                )
                close_result = close_position(position)
                if close_result:
                    open_position(symbol, ml_signal, state.volume)
                    adjust_trading_parameters(symbol, current_profit)
            else:
                logging.info(
                    f"{symbol}: Holding position as ML lacks confidence or signal is neutral"
                )

        # Optional: Lock profit if above GAIN_THRESHOLD
        elif current_profit >= GAIN_THRESHOLD:
            logging.info(f"{symbol}: Locking profit at {current_profit}")
            close_result = close_position(position)
            if close_result:
                adjust_trading_parameters(symbol, current_profit)


def open_position(symbol, direction, volume):
    """Open a new position in the given direction"""
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": (
            mt5.symbol_info_tick(symbol).ask
            if direction == "buy"
            else mt5.symbol_info_tick(symbol).bid
        ),
        "deviation": 20,
        "magic": 234000,
        "comment": f"open {direction} position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"Successfully opened {direction} position for {symbol}")
        return True
    else:
        logging.error(
            f"Failed to open {direction} position for {symbol}: {result.comment}"
        )
        return False


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
    
    logging.info(f"=======Order placed successfully for {symbol}: {direction.upper()} "
                f"Volume: {volume}, Price: {price}, SL: {sl}, TP: {tp}========")
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
            
            time.sleep(0.2)  # Check every 200ms
            
        except Exception as e:
            logging.error(f"Error in {symbol} trader: {e}")
            time.sleep(1)
