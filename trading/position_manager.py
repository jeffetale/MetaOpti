# trading/position_manager.py
import logging
from datetime import datetime
from config import POSITION_REVERSAL_THRESHOLD, mt5

from logging_config import setup_comprehensive_logging
setup_comprehensive_logging()

class PositionManager:
    def __init__(self, order_manager, risk_manager):
        self.logger = logging.getLogger(__name__)
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.trailing_stops = {}
        self.initial_stops = {}

    def manage_open_positions(self, symbol, trading_state, trading_stats=None):
        """Comprehensive position management with advanced features"""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        state = trading_state.symbol_states[symbol]

        for position in positions:
            position_id = position.ticket

            # Initialize tracking for new positions
            if position_id not in self.initial_stops:
                self.initial_stops[position_id] = position.sl

            self._check_position_age(position)
            self._manage_position_profit(position, symbol, state, trading_stats)
            self._manage_trailing_stop(position, symbol)
            self._check_reversal_conditions(position, symbol, state, trading_stats)

    def _manage_trailing_stop(self, position, symbol):
        """Manage trailing stop loss for all positions"""
        try:
            # Get current market price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.error(f"Cannot get tick data for {symbol}")
                return

            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Cannot get symbol info for {symbol}")
                return

            # Get minimum stop level in points
            min_stop_level = symbol_info.trade_stops_level

            # Convert to price difference
            point_value = symbol_info.point
            min_stop_distance = min_stop_level * point_value

            # Add extra buffer to minimum distance (e.g., 20% more)
            min_stop_distance = min_stop_distance * 1.2

            current_price = (
                tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            )
            position_id = position.ticket

            # Initialize trailing stop if not exists
            if position_id not in self.trailing_stops:
                self.trailing_stops[position_id] = {
                    "highest_price": (
                        position.price_current
                        if position.type == mt5.ORDER_TYPE_BUY
                        else float("inf")
                    ),
                    "lowest_price": (
                        position.price_current
                        if position.type == mt5.ORDER_TYPE_SELL
                        else float("-inf")
                    ),
                }

            new_sl = None

            if position.type == mt5.ORDER_TYPE_BUY:
                # For buy positions
                if current_price > self.trailing_stops[position_id]["highest_price"]:
                    self.trailing_stops[position_id]["highest_price"] = current_price
                    new_sl = current_price - min_stop_distance

                    # Ensure new stop loss is higher than current one
                    if position.sl is None or new_sl > position.sl:
                        # Verify stop loss isn't too close to current price
                        if (current_price - new_sl) >= min_stop_distance:
                            self._modify_stop_loss(position, new_sl)
                        else:
                            self.logger.info(
                                f"Stop loss too close to current price for {symbol}"
                            )

            else:  # SELL position
                if current_price < self.trailing_stops[position_id]["lowest_price"]:
                    self.trailing_stops[position_id]["lowest_price"] = current_price
                    new_sl = current_price + min_stop_distance

                    # Ensure new stop loss is lower than current one
                    if position.sl is None or new_sl < position.sl:
                        # Verify stop loss isn't too close to current price
                        if (new_sl - current_price) >= min_stop_distance:
                            self._modify_stop_loss(position, new_sl)
                        else:
                            self.logger.info(
                                f"Stop loss too close to current price for {symbol}"
                            )

            # Handle positions that haven't moved into profit
            if position.profit <= 0 and position.sl is None:
                self._set_initial_stop_loss(position, symbol_info, min_stop_distance)

        except Exception as e:
            self.logger.error(f"Error in trailing stop management: {e}")

    def _set_initial_stop_loss(self, position, symbol_info, min_stop_distance):
        """Set initial stop loss for positions that haven't had one set"""
        try:
            current_price = position.price_current

            # Calculate initial stop loss based on position type
            if position.type == mt5.ORDER_TYPE_BUY:
                initial_sl = current_price - (
                    min_stop_distance * 2
                )  # Double the minimum distance for initial stop
            else:
                initial_sl = current_price + (min_stop_distance * 2)

            self._modify_stop_loss(position, initial_sl)
            self.initial_stops[position.ticket] = initial_sl

        except Exception as e:
            self.logger.error(f"Error setting initial stop loss: {e}")

    def _modify_stop_loss(self, position, new_sl):
        """Modify stop loss level for a position"""
        try:
            # Round the stop loss to the symbol's digits
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info:
                new_sl = round(new_sl, symbol_info.digits)

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
                    📊 Current Price: {position.price_current}
                """
                )
            else:
                self.logger.warning(
                    f"""
                    ⚠️ Failed to modify stop loss:
                    🎫 Ticket: {position.ticket}
                    ❌ Error code: {result.retcode if result else 'Unknown'}
                    🛑 Attempted SL: {new_sl}
                    📊 Current Price: {position.price_current}
                """
                )

        except Exception as e:
            self.logger.error(f"Error modifying stop loss: {e}")

    def _check_position_age(self, position):
        """Monitor position duration and take action if needed"""
        position_age = (
            datetime.now() - datetime.fromtimestamp(position.time)
        ).total_seconds()

        if position_age >= 30 and position.profit < 0:
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
        if position.profit <= POSITION_REVERSAL_THRESHOLD:
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
                        state.volume * 1.5,
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
