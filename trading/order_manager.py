# trading/order_manager.py

import logging
from config import mt5, TRADING_CONFIG, update_risk_profile

from logging_config import setup_comprehensive_logging
setup_comprehensive_logging()

# update_risk_profile('AGGRESSIVE')
# update_risk_profile('MODERATE')
# update_risk_profile('CONSERVATIVE')


class OrderManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎮 Initialized OrderManager")

    def _validate_symbol_info(self, symbol_info, symbol):
        """Validate symbol information before placing an order"""
        if symbol_info is None:
            self.logger.error(f"❌ Failed to get symbol info for {symbol}")
            return False

        if not symbol_info.visible:
            self.logger.error(f"🚫 {symbol} is not visible in market watch")
            return False

        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            self.logger.error(f"⛔ {symbol} is not available for full trading")
            return False

        self.logger.info(f"✅ Symbol {symbol} validated successfully")
        return True

    def _get_valid_filling_mode(self, symbol):
        """Get valid filling mode for symbol based on execution mode"""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"❌ Cannot get symbol info for {symbol}")
            return None

        filling_mode = symbol_info.filling_mode

        self.logger.info(f"🔍 Symbol {symbol} supports filling modes: {filling_mode}")

        return mt5.ORDER_FILLING_IOC

    def _validate_account_money(self, symbol, volume, direction, tick):
        """Validate if account has enough money for the trade"""
        account_info = mt5.account_info()
        if not account_info:
            self.logger.error("Failed to get account info")
            return False

        # Get symbol specification
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"❌ Failed to get symbol info for {symbol}")
            return False

        # Calculate margin required for the trade
        price = tick.ask if direction == "buy" else tick.bid
        margin = self._calculate_margin(symbol_info, volume, price)

        self.logger.info(
            f"""
            💰 Margin Check for {symbol}:
            📈 Direction: {direction}
            📊 Volume: {volume}
            💱 Required Margin: {margin:.2f}
            💵 Free Margin: {account_info.margin_free:.2f}
            💶 Balance: {account_info.balance:.2f}
            💷 Equity: {account_info.equity:.2f}
            """
        )

        # Check if we have enough free margin
        if margin > account_info.margin_free:
            self.logger.error(
                f"""
                ⚠️ Insufficient margin for trade {symbol}:
                💸 Required: {margin:.2f}
                💰 Available: {account_info.margin_free:.2f}
                """
            )
            return False

        self.logger.info("✅ Account has sufficient margin for trade")
        return True

    def _calculate_margin(self, symbol_info, volume, price):
        """Calculate required margin for trade based on leverage"""
        try:
            # Get account leverage
            account_info = mt5.account_info()
            if not account_info:
                self.logger.error("Failed to get account info for leverage calculation")
                return float("inf")

            leverage = account_info.leverage
            if leverage == 0:  # Protect against division by zero
                self.logger.error("Account leverage is 0, using default 100")
                leverage = 100

            contract_size = symbol_info.trade_contract_size

            # Basic margin calculation
            margin = (price * volume * contract_size) / leverage

            # Apply currency conversion if needed
            if symbol_info.currency_profit != account_info.currency:
                conversion_rate = self._get_conversion_rate(symbol_info.currency_profit)
                margin *= conversion_rate
                self.logger.info(
                    f"💱 Applied currency conversion rate: {conversion_rate}"
                )

            # Log detailed margin calculation
            self.logger.debug(
                f"""
                💰 Margin Calculation Details:
                💵 Price: {price}
                📊 Volume: {volume}
                📈 Contract Size: {contract_size}
                💪 Leverage: {leverage}
                💱 Base Margin: {margin:.2f}
                🏦 Account Currency: {account_info.currency}
                💰 Symbol Currency: {symbol_info.currency_profit}
                """
            )

            return margin

        except Exception as e:
            self.logger.error(f"Error in margin calculation: {e}")
            return float("inf")

    def _get_conversion_rate(self, currency):
        """Get conversion rate to account currency"""
        account_currency = mt5.account_info().currency
        if currency == account_currency:
            return 1

        # Try direct conversion pair
        conversion_symbol = f"{currency}{account_currency}"
        rate = mt5.symbol_info_tick(conversion_symbol)
        if rate:
            self.logger.info(f"💱 Found direct conversion rate for {conversion_symbol}")
            return rate.bid

        # Try inverse pair
        conversion_symbol = f"{account_currency}{currency}"
        rate = mt5.symbol_info_tick(conversion_symbol)
        if rate:
            self.logger.info(
                f"💱 Found inverse conversion rate for {conversion_symbol}"
            )
            return 1 / rate.ask

        self.logger.warning(f"⚠️ Could not find conversion rate for {currency}")
        return 1

    def _validate_tick_info(self, tick, symbol):
        """Validate tick information before placing an order"""
        if not tick:
            self.logger.error(f"❌ Failed to get tick data for {symbol}")
            return False

        if tick.bid == 0 or tick.ask == 0:
            self.logger.error(f"❌ Invalid bid/ask prices for {symbol}")
            return False

        self.logger.debug(
            f"""
            ✅ Tick validation passed for {symbol}:
            💰 Bid: {tick.bid}
            💰 Ask: {tick.ask}
            """
        )
        return True

    def _adjust_volume(self, volume, symbol_info):
        """Adjust trading volume to match symbol limits"""
        original_volume = volume
        volume = max(volume, symbol_info.volume_min)
        volume = min(volume, symbol_info.volume_max)
        volume = round(volume, 2)

        if volume != original_volume:
            self.logger.info(
                f"""
                📊 Volume adjusted:
                Original: {original_volume}
                Adjusted: {volume}
                """
            )
        return volume

    def _calculate_order_parameters(self, direction, tick, atr):
        """Calculate order price, stop loss, and take profit levels"""
        spread = tick.ask - tick.bid

        if direction == "buy":
            price = tick.ask
            sl = price - (atr * TRADING_CONFIG.SL_ATR_MULTIPLIER)
            tp = price + (atr * TRADING_CONFIG.TP_ATR_MULTIPLIER)
        else:
            price = tick.bid
            sl = price + (atr * TRADING_CONFIG.SL_ATR_MULTIPLIER)
            tp = price - (atr * TRADING_CONFIG.TP_ATR_MULTIPLIER)

        self.logger.info(
            f"""
            📊 Order Parameters:
            💰 Price: {price}
            🛑 Stop Loss: {sl}
            🎯 Take Profit: {tp}
            📈 Spread: {spread}
            """
        )
        return price, sl, tp

    def _create_order_request(self, symbol, direction, volume, price, sl, tp, filling_type):
        """Create an order request with specified parameters"""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": TRADING_CONFIG.PRICE_DEVIATION_POINTS, 
            "magic": TRADING_CONFIG.ORDER_MAGIC_NUMBER,
            "comment": "single",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        self.logger.debug(f"📝 Created order request: {request}")
        return request

    def _calculate_safe_volume(self, symbol, direction, price, available_margin):
        """Calculate safe trading volume based on available margin and risk parameters"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Cannot get symbol info for {symbol}")
                return None

            account_info = mt5.account_info()
            if not account_info:
                self.logger.error("Cannot get account info")
                return None

            # Get symbol trading parameters
            contract_size = symbol_info.trade_contract_size
            margin_rate = 1 / account_info.leverage
            min_volume = symbol_info.volume_min
            max_volume = symbol_info.volume_max
            volume_step = symbol_info.volume_step

            # Calculate maximum possible volume based on available margin
            # Using 50% of available margin instead of 90% for more aggressive trading
            max_margin_volume = (available_margin * TRADING_CONFIG.MARGIN_USAGE_LIMIT) / (
                price * contract_size * margin_rate
            )

            # Calculate volume based on equity (risk management)
            # Increased from 0.2 (20%) to 0.4 (40%) of equity
            equity_based_volume = (account_info.equity * TRADING_CONFIG.EQUITY_RISK_PER_TRADE) / (price * contract_size)

            # Set minimum target volume (can be adjusted based on your preference)
            min_target_volume = TRADING_CONFIG.MIN_TARGET_VOLUME

            # Choose volume - take the maximum of our minimum target and calculated volumes
            calculated_volume = max(
                min_target_volume,
                min(max_margin_volume, equity_based_volume)
            )

            # Ensure volume is within symbol's limits
            calculated_volume = min(calculated_volume, max_volume)
            calculated_volume = max(calculated_volume, min_volume)

            # Round to the nearest volume step
            steps = round(calculated_volume / volume_step)
            safe_volume = steps * volume_step

            # Final validation against symbol limits
            if safe_volume < min_volume:
                safe_volume = min_volume
            elif safe_volume > max_volume:
                safe_volume = max_volume

            self.logger.info(
                f"""
                💭 Volume Calculation for {symbol}:
                💰 Available Margin: {available_margin}
                📊 Initial Calculated Volume: {calculated_volume}
                🔒 Final Safe Volume: {safe_volume}
                📏 Volume Constraints:
                    Min: {min_volume}
                    Max: {max_volume}
                    Step: {volume_step}
                📈 Contract Size: {contract_size}
                💱 Margin Rate: {margin_rate}
            """
            )

            return safe_volume

        except Exception as e:
            self.logger.error(f"Error calculating safe volume: {e}")
            return None

    def _validate_volume(self, volume, symbol_info):
        """Validate if volume meets symbol requirements"""
        if volume < symbol_info.volume_min:
            self.logger.error(f"Volume {volume} below minimum {symbol_info.volume_min}")
            return False

        if volume > symbol_info.volume_max:
            self.logger.error(f"Volume {volume} above maximum {symbol_info.volume_max}")
            return False

        # Check if volume is multiple of step
        steps = round(volume / symbol_info.volume_step)
        adjusted_volume = steps * symbol_info.volume_step
        if abs(adjusted_volume - volume) > 1e-8:  # Float comparison with tolerance
            self.logger.error(
                f"Volume {volume} not multiple of step {symbol_info.volume_step}"
            )
            return False

        return True

    def place_order(
        self, symbol, direction, atr, volume, trading_stats=None, is_ml_signal=False
    ):
        """Enhanced order placement with proper volume calculation and validation"""
        self.logger.info(
            f"""
            🎯 Starting order placement:
            Symbol: {symbol}
            Direction: {direction}
            Initial Volume: {volume}
            ML Signal: {is_ml_signal}
        """
        )

        symbol_info = mt5.symbol_info(symbol)
        if not self._validate_symbol_info(symbol_info, symbol):
            return False

        tick = mt5.symbol_info_tick(symbol)
        if not self._validate_tick_info(tick, symbol):
            return False

        # Get account info for margin calculation
        account_info = mt5.account_info()
        if not account_info:
            self.logger.error("Failed to get account info")
            return False

        # Calculate safe volume
        safe_volume = self._calculate_safe_volume(
            symbol,
            direction,
            tick.ask if direction == "buy" else tick.bid,
            account_info.margin_free,
        )

        if safe_volume is None:
            return False

        # Validate final volume
        if not self._validate_volume(safe_volume, symbol_info):
            self.logger.error(f"Invalid final volume {safe_volume} for {symbol}")
            return False

        # Get valid filling mode
        filling_mode = self._get_valid_filling_mode(symbol)
        if not filling_mode:
            return False

        # Calculate order parameters
        price, sl, tp = self._calculate_order_parameters(direction, tick, atr)

        # Create and send order
        request = self._create_order_request(
            symbol, direction, safe_volume, price, sl, tp, filling_mode
        )

        result = self._send_order(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self._log_successful_order(symbol, direction, safe_volume, price, sl, tp)

            if trading_stats:
                trading_stats.log_trade(
                    symbol=symbol,
                    direction=direction,
                    profit=0,  # Initial profit is 0
                    is_ml_signal=is_ml_signal,
                )

            return True

        return False

    def place_hedged_order(self, symbol: str, direction: str, volume: float, atr: float, sl_distance: float = None, tp_distance: float = None, market_context: dict = None) -> object:
        try:
            self.logger.info(f"Attempting to place order - Symbol: {symbol}, Direction: {direction}, Volume: {volume}")
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Symbol info not found for {symbol}")
                return None
                
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.error(f"Tick info not found for {symbol}")
                return None
                
            self.logger.info(f"Current tick - Bid: {tick.bid}, Ask: {tick.ask}")
            
            # Calculate entry price
            entry_price = tick.ask if direction == "buy" else tick.bid
            
            # Use provided distances or calculate from ATR
            sl_distance = sl_distance if sl_distance is not None else atr * TRADING_CONFIG.SL_ATR_MULTIPLIER
            tp_distance = tp_distance if tp_distance is not None else atr * TRADING_CONFIG.TP_ATR_MULTIPLIER
            
            self.logger.info(f"Calculated levels - SL Distance: {sl_distance}, TP Distance: {tp_distance}")
            
            if direction == "buy":
                sl = entry_price - sl_distance
                tp = entry_price + tp_distance
            else:
                sl = entry_price + sl_distance
                tp = entry_price - tp_distance
                
            self.logger.info(f"Final levels - Entry: {entry_price}, SL: {sl}, TP: {tp}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": sl,
                "tp": tp,
                "deviation": TRADING_CONFIG.PRICE_DEVIATION_POINTS,
                "magic": TRADING_CONFIG.ORDER_MAGIC_NUMBER,
                "comment": "hedged",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            self.logger.info(f"Sending order request: {request}")
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Order placed successfully")
                return result
                
            self.logger.error(f"Order failed with retcode: {result.retcode if result else 'No result'}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing hedged order: {str(e)}", exc_info=True)
            return None

    def modify_position_sl_tp(self, ticket: int, new_sl: float = None, new_tp: float = None) -> bool:
        """Modify position's stop loss and/or take profit with improved validation"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.error(f"Position {ticket} not found")
                return False
                
            position = position[0]
            symbol_info = mt5.symbol_info(position.symbol)
            if not symbol_info:
                self.logger.error(f"Symbol info not found for {position.symbol}")
                return False

            current_price = mt5.symbol_info_tick(position.symbol)
            if not current_price:
                self.logger.error(f"Unable to get current price for {position.symbol}")
                return False

            # Validate stop loss levels
            if new_sl is not None:
                if position.type == mt5.ORDER_TYPE_BUY:
                    if new_sl >= current_price.bid:
                        self.logger.error(f"Invalid stop loss level for buy position: {new_sl} >= {current_price.bid}")
                        return False
                else:  # SELL position
                    if new_sl <= current_price.ask:
                        self.logger.error(f"Invalid stop loss level for sell position: {new_sl} <= {current_price.ask}")
                        return False

            # Add minimal spread check to prevent too frequent modifications
            min_spread = symbol_info.point * symbol_info.spread
            
            # Only proceed if the change is significant enough
            if new_sl is not None and position.sl != 0 and abs(new_sl - position.sl) <= min_spread:
                self.logger.info(f"Stop loss modification too small (within spread): {abs(new_sl - position.sl)} <= {min_spread}")
                return False
                
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": position.symbol,  # Added symbol field
                "sl": new_sl if new_sl is not None else position.sl,  # Keep existing SL if not modifying
                "tp": new_tp if new_tp is not None else position.tp,  # Keep existing TP if not modifying
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            self.logger.info(f"Sending modify request: {request}")
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Successfully modified position {ticket}")
                return True
            else:
                self.logger.error(f"Failed to modify position {ticket}. Result: {result.retcode if result else 'No result'}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error modifying position SL/TP: {str(e)}", exc_info=True)
            return False

    def _send_order(self, request):
        """Send order to MT5 and handle the response"""
        self.logger.info("🚀 Sending order to MT5...")
        result = mt5.order_send(request)

        if result is None:
            self.logger.error(f"❌ Failed to send order: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"❌ Order failed. Error code: {result.retcode}")
            return None

        self.logger.info("✅ Order sent successfully")
        return result

    def _log_successful_order(self, symbol, direction, volume, price, sl, tp):
        """Log details of successful order placement"""
        self.logger.info(
            f"""
            🎉 Order placed successfully:
            🏷️ Symbol: {symbol}
            📈 Direction: {direction}
            📊 Volume: {volume}
            💰 Price: {price}
            🛑 SL: {sl}
            🎯 TP: {tp}
            """
        )

    def _update_trading_stats(self, trading_stats, symbol, direction, result):
        """Update trading statistics after successful order"""
        if trading_stats:
            trading_stats.update_order_stats(symbol, direction, result.volume)
            self.logger.info(f"📊 Updated trading statistics for {symbol}")

    def close_position(self, position):
        """Close an open trading position with advanced error handling and logging."""
        try:
            if not mt5.initialize():
                self.logger.error(
                    f"❌ MT5 not initialized when trying to close position {position.ticket}"
                )
                return False

            # Get current market tick information
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                self.logger.error(
                    f"❌ Could not get tick information for {position.symbol}"
                )
                return False

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": (
                    mt5.ORDER_TYPE_SELL
                    if position.type == mt5.ORDER_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY
                ),
                "price": (
                    tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
                ),
                "deviation": 50,  # Increased deviation for better fill probability
                "magic": 234000,
                "comment": "close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            self.logger.debug(
                f"""
                📝 Close position request for {position.ticket}:
                {request}
                """
            )

            # Send order with timeout
            result = mt5.order_send(request)

            # Error checking
            if result is None:
                self.logger.error(
                    f"❌ Failed to send close order for position {position.ticket}. Returned None."
                )
                return False

            # Check return code
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"""
                    ✅ Successfully closed position:
                    🎫 Ticket: {position.ticket}
                    💰 Profit: {position.profit}
                    📊 Volume: {position.volume}
                    """
                )
                return True
            else:
                self.logger.warning(
                    f"""
                    ⚠️ Failed to close position {position.ticket}:
                    Return code: {result.retcode}
                    Comment: {result.comment}
                    """
                )
                return False

        except Exception as e:
            self.logger.error(
                f"""
                ❌ Exception when closing position {position.ticket}:
                Error: {str(e)}
                """
            )
            return False
