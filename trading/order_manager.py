# trading/order_manager.py

import logging
from config import mt5

from logging_config import setup_comprehensive_logging
setup_comprehensive_logging()


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
            sl = price - (atr * 2)
            tp = price + (atr * 3)
        else:
            price = tick.bid
            sl = price + (atr * 2)
            tp = price - (atr * 3)

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
            "deviation": 20,  # Increased deviation helps with execution
            "magic": 234000,
            "comment": "python",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        
        self.logger.debug(f"📝 Created order request: {request}")
        return request

    def place_order(
        self, symbol, direction, atr, volume, trading_stats=None, is_ml_signal=False
    ):
        """Enhanced order placement with proper filling mode and money validation"""
        self.logger.info(
            f"""
            🎯 Starting order placement:
            Symbol: {symbol}
            Direction: {direction}
            Volume: {volume}
            ML Signal: {is_ml_signal}
            """
        )

        symbol_info = mt5.symbol_info(symbol)
        if not self._validate_symbol_info(symbol_info, symbol):
            return False

        tick = mt5.symbol_info_tick(symbol)
        if not self._validate_tick_info(tick, symbol):
            return False

        # Validate account has enough money before adjusting volume
        if not self._validate_account_money(symbol, volume, direction, tick):
            return False

        # Adjust volume based on symbol limits
        volume = self._adjust_volume(volume, symbol_info)

        # Get valid filling mode for this symbol
        filling_mode = self._get_valid_filling_mode(symbol)
        if not filling_mode:
            return False

        # Calculate order parameters
        price, sl, tp = self._calculate_order_parameters(direction, tick, atr)

        # Create and send order with proper filling mode
        request = self._create_order_request(
            symbol, direction, volume, price, sl, tp, filling_mode
        )

        # Log order request details
        self.logger.info(
            f"""
            📋 Order Details:
            🏷️ Symbol: {symbol}
            📈 Direction: {direction}
            📊 Volume: {volume}
            💰 Price: {price}
            🛑 SL: {sl}
            🎯 TP: {tp}
            ⚙️ Filling Mode: {filling_mode}
            """
        )

        result = self._send_order(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self._log_successful_order(symbol, direction, volume, price, sl, tp)
            if trading_stats:
                self._update_trading_stats(trading_stats, symbol, direction, result)
            return True

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
        """Close an open trading position with advanced error handling and logging.

        Args:
            position: MT5 position object to be closed

        Returns:
            bool: True if position was closed successfully, False otherwise
        """
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
