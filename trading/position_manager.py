# trading/position_manager.py

import logging
from datetime import datetime
from config import mt5, TRADING_CONFIG

from logging_config import setup_comprehensive_logging

setup_comprehensive_logging()


class PositionManager:
    def __init__(self, order_manager, risk_manager):
        self.logger = logging.getLogger(__name__)
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.trailing_stops = {}

    def manage_open_positions(self, symbol, trading_state, trading_stats=None):
        """Comprehensive position management with advanced features"""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        state = trading_state.symbol_states[symbol]

        for position in positions:
            self._check_position_age(position)
            self._manage_position_profit(position, symbol, state, trading_stats)
            self._manage_breakeven_plus(position, symbol)
            self._enhanced_trailing_stop(position, symbol)
            self._check_reversal_conditions(position, symbol, state, trading_stats)

    def _manage_breakeven_plus(self, position, symbol):
        """Move stop loss to break-even plus additional pips once in sufficient profit"""
        try:
            # Only proceed if position is in profit
            if position.profit <= 0:
                return

            # Get symbol info for pip calculations
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return

            # Calculate point value
            point = symbol_info.point
            profit_pips = position.profit / (point * position.volume)

            # If profit exceeds threshold (20 pips)
            if profit_pips >= 20:
                # Calculate break-even level plus 5 pips
                breakeven_plus = position.price_open + (
                    TRADING_CONFIG.BREAKEVEN_PLUS_PIPS * point if position.type == mt5.ORDER_TYPE_BUY else -TRADING_CONFIG.BREAKEVEN_PLUS_PIPS * point
                )

                # Only modify if new stop loss is better than existing
                if (
                    position.sl is None
                    or (
                        position.type == mt5.ORDER_TYPE_BUY
                        and breakeven_plus > position.sl
                    )
                    or (
                        position.type == mt5.ORDER_TYPE_SELL
                        and breakeven_plus < position.sl
                    )
                ):
                    self._modify_stop_loss(position, breakeven_plus)
                    self.logger.info(
                        f"""
                        ✅ Set break-even plus for {position.symbol}:
                        🎫 Ticket: {position.ticket}
                        💰 Profit Pips: {profit_pips:.1f}
                        🛑 New SL: {breakeven_plus}
                        """
                    )

        except Exception as e:
            self.logger.error(f"Error in break-even plus management: {e}")

    def _enhanced_trailing_stop(self, position, symbol):
        """Advanced trailing stop with better profit protection"""
        try:
            # Get current market price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return

            # Calculate ATR properly using true range
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 14)
            if rates is None:
                return

            import pandas as pd
            import numpy as np

            df = pd.DataFrame(rates)
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
            df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            atr = df['tr'].mean()

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            position_id = position.ticket

            # Initialize trailing data if not exists
            if position_id not in self.trailing_stops:
                self.trailing_stops[position_id] = {
                    "highest_price": current_price if position.type == mt5.ORDER_TYPE_BUY else float("inf"),
                    "lowest_price": current_price if position.type == mt5.ORDER_TYPE_SELL else float("-inf"),
                    "profit_locked": False,
                    "breakeven_set": False
                }

            trail_data = self.trailing_stops[position_id]
            
            # Calculate profit in pips
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return
            
            point = symbol_info.point
            profit_pips = position.profit / (point * position.volume)

            # Break-even logic once we have 10 pips profit
            if not trail_data["breakeven_set"] and profit_pips >= 10:
                breakeven_level = position.price_open + (2 * point if position.type == mt5.ORDER_TYPE_BUY else -2 * point)
                self._modify_stop_loss(position, breakeven_level)
                trail_data["breakeven_set"] = True
                self.logger.info(f"Set break-even stop for {symbol} position {position_id}")
                return

            # Enhanced trailing stop logic
            if position.type == mt5.ORDER_TYPE_BUY:
                if current_price > trail_data["highest_price"]:
                    trail_data["highest_price"] = current_price
                    
                    # Trail distance gets tighter as profit increases
                    if profit_pips > 20:
                        trail_distance = atr * TRADING_CONFIG.TRAILING_STOP_TIGHT_ATR  # Tighter trail for larger profits
                    elif profit_pips > 10:
                        trail_distance = atr * 1.5
                    else:
                        trail_distance = atr * TRADING_CONFIG.INITIAL_VOLUME

                    new_sl = current_price - trail_distance
                    
                    # Ensure new stop loss is better than current
                    if not position.sl or new_sl > position.sl:
                        self._modify_stop_loss(position, new_sl)
                        self.logger.info(f"Updated trailing stop for {symbol} Buy position {position_id} to {new_sl}")

            else:  # SELL position
                if current_price < trail_data["lowest_price"]:
                    trail_data["lowest_price"] = current_price
                    
                    # Trail distance gets tighter as profit increases
                    if profit_pips > 20:
                        trail_distance = atr * TRADING_CONFIG.TRAILING_STOP_TIGHT_ATR
                    elif profit_pips > 10:
                        trail_distance = atr * 1.5
                    else:
                        trail_distance = atr * TRADING_CONFIG.INITIAL_VOLUME

                    new_sl = current_price + trail_distance
                    
                    # Ensure new stop loss is better than current
                    if not position.sl or new_sl < position.sl:
                        self._modify_stop_loss(position, new_sl)
                        self.logger.info(f"Updated trailing stop for {symbol} Sell position {position_id} to {new_sl}")

        except Exception as e:
            self.logger.error(f"Error in enhanced trailing stop management: {e}")

    def _modify_stop_loss(self, position, new_sl):
        """Modify stop loss level for a position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp,  # Maintain existing take profit
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"""
                    ✅ Successfully modified stop loss:
                    🎫 Ticket: {position.ticket}
                    🛑 New SL: {new_sl}
                    💰 Current Profit: {position.profit}
                """
                )
            else:
                self.logger.warning(
                    f"""
                    ⚠️ Failed to modify stop loss:
                    🎫 Ticket: {position.ticket}
                    ❌ Error code: {result.retcode if result else 'Unknown'}
                """
                )

        except Exception as e:
            self.logger.error(f"Error modifying stop loss: {e}")

    def _check_position_age(self, position):
        """Monitor position duration and take action if needed"""
        position_age = (
            datetime.now() - datetime.fromtimestamp(position.time)
        ).total_seconds()

        if position_age >= TRADING_CONFIG.MAX_POSITION_AGE_SECONDS and position.profit < 0:
            if self.order_manager.close_position(position):
                self.logger.info(
                    f"Closed aged position {position.ticket} with negative profit"
                )

    def _manage_position_profit(self, position, symbol, state, trading_stats):
        """Monitor and manage position profit/loss"""
        if position.profit <= -15.80:
            if self.order_manager.close_position(position):
                self.logger.info(
                    f"Closed position {position.ticket} due to significant loss"
                )
                self.risk_manager.adjust_trading_parameters(
                    symbol, position.profit, state
                )
                if trading_stats:
                    trading_stats.log_trade(symbol, "close", position.profit, False)

    def _check_reversal_conditions(self, position, symbol, state, trading_stats):
        """Check and execute position reversal if conditions are met"""
        if position.profit <= TRADING_CONFIG.POSITION_REVERSAL_THRESHOLD:
            if self.order_manager.close_position(position):
                reversal_direction = (
                    "sell" if position.type == mt5.ORDER_TYPE_BUY else "buy"
                )

                # Get market conditions for reversal
                atr = self._get_market_volatility(symbol)
                if atr:
                    success = self.order_manager.place_order(
                        symbol,
                        reversal_direction,
                        atr,
                        state.volume * 1.5,  # Increase volume for reversal
                        trading_stats,
                    )

                    if success:
                        self.logger.info(f"Successfully reversed position for {symbol}")
                        if trading_stats:
                            trading_stats.log_position_reversal(symbol)

    def _get_market_volatility(self, symbol):
        """Calculate current market volatility"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20)
        if rates is None:
            return None

        import pandas as pd

        df = pd.DataFrame(rates)
        return df["high"].max() - df["low"].min()
