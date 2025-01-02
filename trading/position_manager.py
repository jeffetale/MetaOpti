# trading/position_manager.py

import logging
from datetime import datetime
from config import mt5, TRADING_CONFIG, MT5Config, update_risk_profile

from logging_config import setup_comprehensive_logging

setup_comprehensive_logging()

#update_risk_profile('AGGRESSIVE')
# update_risk_profile('MODERATE')
# update_risk_profile('CONSERVATIVE')


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

            # Calculate point value and profit in pips
            point = symbol_info.point
            profit_pips = position.profit / (point * position.volume)

            # Define the breakeven level based on profit condition
            breakeven_plus = None
            
            if profit_pips >= 20:  # Higher threshold with more protective pips
                # Calculate break-even level plus configured pips
                breakeven_plus = position.price_open + (
                    TRADING_CONFIG.BREAKEVEN_PLUS_PIPS * point if position.type == mt5.ORDER_TYPE_BUY 
                    else -TRADING_CONFIG.BREAKEVEN_PLUS_PIPS * point
                )
            elif position.profit > 10:  # Lower threshold with tighter protection
                # Calculate break-even level plus 2 pips
                breakeven_plus = position.price_open + (
                    2 * point if position.type == mt5.ORDER_TYPE_BUY 
                    else -2 * point
                )

            # Only modify if breakeven level was set and new stop loss is better than existing
            if breakeven_plus and (
                position.sl is None
                or (position.type == mt5.ORDER_TYPE_BUY and breakeven_plus > position.sl)
                or (position.type == mt5.ORDER_TYPE_SELL and breakeven_plus < position.sl)
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
        """Advanced trailing stop with more aggressive profit protection"""
        try:
            # Get current market price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return

            # Calculate ATR
            rates = mt5.copy_rates_from_pos(symbol, MT5Config.TIMEFRAME, 0, 14)
            if rates is None:
                return

            import pandas as pd
            import numpy as np

            df = pd.DataFrame(rates)
            df["tr"] = np.maximum(
                df["high"] - df["low"],
                np.maximum(
                    abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"] - df["close"].shift(1))
                )
            )
            atr = df["tr"].mean()

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            position_id = position.ticket

            # Initialize trailing data if not exists
            if position_id not in self.trailing_stops:
                self.trailing_stops[position_id] = {
                    "highest_price": current_price if position.type == mt5.ORDER_TYPE_BUY else float("inf"),
                    "lowest_price": current_price if position.type == mt5.ORDER_TYPE_SELL else float("-inf"),
                    "max_profit": 0,
                    "profit_locked": False
                }

            trail_data = self.trailing_stops[position_id]
            
            # Calculate profit in pips
            symbol_info = mt5.symbol_info(symbol)
            point = symbol_info.point
            profit_pips = position.profit / (point * position.volume)

            # Update maximum profit reached
            trail_data["max_profit"] = max(trail_data["max_profit"], position.profit)
            
            # Early profit protection - move stop loss to break even + small buffer
            if profit_pips >= 10 and not trail_data.get("breakeven_set", False):  # Reduced from 15 to 10 pips
                breakeven_level = position.price_open + (
                    2 * point if position.type == mt5.ORDER_TYPE_BUY else -2 * point
                )
                self._modify_stop_loss(position, breakeven_level)
                trail_data["breakeven_set"] = True
                return

            # Progressive trailing stop based on profit level
            if position.type == mt5.ORDER_TYPE_BUY:
                if current_price > trail_data["highest_price"]:
                    trail_data["highest_price"] = current_price
                    
                    # Tighter trailing stops as profit increases
                    if profit_pips > 30:
                        trail_distance = atr * 0.5  # Very tight trail for large profits
                    elif profit_pips > 20:
                        trail_distance = atr * 0.75  # Tighter trail for medium profits
                    elif profit_pips > 10:
                        trail_distance = atr * 1.0  # Standard trail for small profits
                    else:
                        trail_distance = atr * 1.5  # Wide trail initially

                    new_sl = current_price - trail_distance
                    
                    if not position.sl or new_sl > position.sl:
                        self._modify_stop_loss(position, new_sl)

            else:  # SELL position
                if current_price < trail_data["lowest_price"]:
                    trail_data["lowest_price"] = current_price
                    
                    if profit_pips > 30:
                        trail_distance = atr * 0.5
                    elif profit_pips > 20:
                        trail_distance = atr * 0.75
                    elif profit_pips > 10:
                        trail_distance = atr * 1.0
                    else:
                        trail_distance = atr * 1.5

                    new_sl = current_price + trail_distance
                    
                    if not position.sl or new_sl < position.sl:
                        self._modify_stop_loss(position, new_sl)

            # Profit protection - close position if profit drops significantly from peak
            max_profit = trail_data["max_profit"]
            if max_profit > 0:
                profit_drawdown = (max_profit - position.profit) / max_profit
                if profit_drawdown >= 0.75 and position.profit > 0:  # Close if we've lost 75% of max profit
                    self.order_manager.close_position(position)
                    self.logger.info(f"Closed position {position_id} to protect profits. Max profit: {max_profit}, Current profit: {position.profit}")

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
        # Close losing positions at max loss
        if position.profit <= -50.80:
            if self.order_manager.close_position(position):
                self.logger.info(f"Closed position {position.ticket} due to significant loss")
                
        # Lock in profits when they reach certain thresholds
        elif position.profit >= 30:  # Add profit taking
            if self.order_manager.close_position(position):
                self.logger.info(f"Closed position {position.ticket} to lock in profits")

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
        rates = mt5.copy_rates_from_pos(symbol, MT5Config.TIMEFRAME, 0, 20)
        if rates is None:
            return None

        import pandas as pd

        df = pd.DataFrame(rates)
        return df["high"].max() - df["low"].min()
