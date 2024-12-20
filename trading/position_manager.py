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
        self.trailing_stops = {}  # Store highest/lowest prices for trailing stops
        self.initial_stops = {}  # Store initial stop losses
        self.profit_locks = {}  # Store profit levels where we've locked in gains

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
                self._initialize_position_tracking(position, symbol)

            self._check_position_age(position)
            self._manage_position_profit(position, symbol, state, trading_stats)
            self._manage_advanced_stop_loss(position, symbol)
            self._check_reversal_conditions(position, symbol, state, trading_stats)

    def _initialize_position_tracking(self, position, symbol):
        """Initialize tracking for a new position"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Cannot get symbol info for {symbol}")
                return

            position_id = position.ticket

            # Initialize tracking dictionaries
            self.initial_stops[position_id] = position.sl
            self.trailing_stops[position_id] = {
                "highest_price": position.price_open,
                "lowest_price": position.price_open,
                "initial_price": position.price_open,
                "atr": self._calculate_atr(symbol),  # Get ATR for dynamic stops
            }
            self.profit_locks[position_id] = {"level": 0, "locked_price": None}

        except Exception as e:
            self.logger.error(f"Error initializing position tracking: {e}")

    def _calculate_atr(self, symbol, period=14, timeframe=mt5.TIMEFRAME_M5):
        """Calculate Average True Range for dynamic stop loss"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
            if rates is None:
                return None

            import pandas as pd
            import numpy as np

            df = pd.DataFrame(rates)

            # Calculate True Range
            df["high_low"] = df["high"] - df["low"]
            df["high_close"] = np.abs(df["high"] - df["close"].shift(1))
            df["low_close"] = np.abs(df["low"] - df["close"].shift(1))
            df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

            # Calculate ATR
            atr = df["tr"].mean()
            return atr

        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return None

    def _manage_advanced_stop_loss(self, position, symbol):
        """Advanced stop loss management with dynamic adjustment"""
        try:
            position_id = position.ticket
            position_data = self.trailing_stops[position_id]
            atr = position_data["atr"]

            if not atr:
                return

            # Get current market price and symbol info
            tick = mt5.symbol_info_tick(symbol)
            symbol_info = mt5.symbol_info(symbol)
            if not tick or not symbol_info:
                return

            current_price = (
                tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            )
            min_stop_distance = symbol_info.trade_stops_level * symbol_info.point * 1.2

            # Update price tracking
            if position.type == mt5.ORDER_TYPE_BUY:
                position_data["highest_price"] = max(
                    position_data["highest_price"], current_price
                )
            else:
                position_data["lowest_price"] = min(
                    position_data["lowest_price"], current_price
                )

            # Calculate price movement from entry
            price_movement = abs(current_price - position_data["initial_price"])
            movement_in_atrs = price_movement / atr if atr else 0

            # Implement tiered stop loss strategy
            new_sl = None
            if position.profit > 0:
                # Position is profitable - implement trailing stop based on ATR and profit level
                if position.type == mt5.ORDER_TYPE_BUY:
                    # Different stop distances based on profit levels
                    if (
                        position.profit >= 0.5
                        and self.profit_locks[position_id]["level"] < 0.5
                    ):
                        # First level of profit locking
                        new_sl = position_data["initial_price"] + (
                            min_stop_distance * 1.5
                        )
                        self.profit_locks[position_id] = {
                            "level": 0.5,
                            "locked_price": new_sl,
                        }
                    elif (
                        position.profit >= 5
                        and self.profit_locks[position_id]["level"] < 1
                    ):
                        # Lock in profits at first target
                        new_sl = position_data["initial_price"] + (
                            min_stop_distance * 2
                        )
                        self.profit_locks[position_id] = {
                            "level": 1,
                            "locked_price": new_sl,
                        }
                    elif (
                        position.profit >= 10
                        and self.profit_locks[position_id]["level"] < 2
                    ):
                        # Move stop to higher level at second target
                        new_sl = position_data["highest_price"] - (atr * 0.5)
                        self.profit_locks[position_id] = {
                            "level": 2,
                            "locked_price": new_sl,
                        }
                    elif position.profit >= 20:
                        # Tighter trailing stop at higher profits
                        new_sl = position_data["highest_price"] - (atr * 0.3)
                else:  # SELL position
                    if (
                        position.profit >= 0.5
                        and self.profit_locks[position_id]["level"] < 0.5
                    ):
                        new_sl = position_data["initial_price"] - (
                            min_stop_distance * 1.5
                        )
                        self.profit_locks[position_id] = {
                            "level": 0.5,
                            "locked_price": new_sl,
                        }
                    elif (
                        position.profit >= 5
                        and self.profit_locks[position_id]["level"] < 1
                    ):
                        new_sl = position_data["initial_price"] - (
                            min_stop_distance * 2
                        )
                        self.profit_locks[position_id] = {
                            "level": 1,
                            "locked_price": new_sl,
                        }
                    elif (
                        position.profit >= 10
                        and self.profit_locks[position_id]["level"] < 2
                    ):
                        new_sl = position_data["lowest_price"] + (atr * 0.5)
                        self.profit_locks[position_id] = {
                            "level": 2,
                            "locked_price": new_sl,
                        }
                    elif position.profit >= 20:
                        new_sl = position_data["lowest_price"] + (atr * 0.3)
            else:
                # Position is not yet profitable - use wider stops based on ATR
                if not position.sl:
                    # Set initial stop loss at 2 ATR for new positions
                    if position.type == mt5.ORDER_TYPE_BUY:
                        new_sl = position_data["initial_price"] - (atr * 2)
                    else:
                        new_sl = position_data["initial_price"] + (atr * 2)

            # Validate and apply new stop loss
            if new_sl is not None:
                # Ensure stop loss respects minimum distance
                if position.type == mt5.ORDER_TYPE_BUY:
                    if (current_price - new_sl) >= min_stop_distance:
                        if position.sl is None or new_sl > position.sl:
                            self._modify_stop_loss(position, new_sl)
                else:
                    if (new_sl - current_price) >= min_stop_distance:
                        if position.sl is None or new_sl < position.sl:
                            self._modify_stop_loss(position, new_sl)

        except Exception as e:
            self.logger.error(f"Error in advanced stop loss management: {e}")

    def _modify_stop_loss(self, position, new_sl):
        """Modify stop loss level for a position"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info:
                new_sl = round(new_sl, symbol_info.digits)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp,
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"""
                    ✅ Modified stop loss:
                    🎫 Ticket: {position.ticket}
                    🛑 New SL: {new_sl}
                    💰 Profit: {position.profit}
                    📊 Price: {position.price_current}
                """
                )
            else:
                self.logger.warning(
                    f"""
                    ⚠️ Failed to modify stop loss:
                    🎫 Ticket: {position.ticket}
                    ❌ Error: {result.retcode if result else 'Unknown'}
                    🛑 Attempted SL: {new_sl}
                    📊 Price: {position.price_current}
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
