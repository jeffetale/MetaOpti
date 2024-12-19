# trading/order_manager.py

import logging
from config import mt5


class OrderManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _validate_symbol_info(self, symbol_info, symbol):
        """Validate symbol information before placing an order"""
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return False

        if not symbol_info.visible:
            self.logger.error(f"{symbol} is not visible")
            return False

        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            self.logger.error(f"{symbol} is not available for full trading")
            return False

        return True

    def _get_valid_filling_mode(self, symbol):
        """Get valid filling mode for symbol based on execution mode"""
        filling_mode = mt5.symbol_info(symbol).filling_mode

        # Log available filling modes for debugging
        self.logger.info(f"Symbol {symbol} supports filling modes: {filling_mode}")

        # Check execution type
        execution_mode = mt5.symbol_info(symbol).trade_mode

        if execution_mode == mt5.SYMBOL_TRADE_EXECUTION_MARKET:
            # Market execution - use what's allowed by symbol
            if filling_mode & mt5.SYMBOL_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            elif filling_mode & mt5.SYMBOL_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            else:
                self.logger.error(f"No valid filling mode found for {symbol}")
                return None
        else:
            # For other execution types, RETURN is always allowed
            return mt5.ORDER_FILLING_RETURN

    def _validate_account_money(self, symbol, volume, direction, tick):
        """Validate if account has enough money for the trade"""
        account_info = mt5.account_info()
        if not account_info:
            self.logger.error("Failed to get account info")
            return False

        # Get symbol specification
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return False

        # Calculate margin required for the trade
        price = tick.ask if direction == "buy" else tick.bid
        margin = self._calculate_margin(symbol_info, volume, price)

        # Log margin requirements
        self.logger.info(
            f"""
            Margin Check for {symbol}:
            Direction: {direction}
            Volume: {volume}
            Required Margin: {margin:.2f}
            Free Margin: {account_info.margin_free:.2f}
            Balance: {account_info.balance:.2f}
            Equity: {account_info.equity:.2f}
        """
        )

        # Check if we have enough free margin
        if margin > account_info.margin_free:
            self.logger.error(
                f"""
                Insufficient margin for trade:
                Required: {margin:.2f}
                Available: {account_info.margin_free:.2f}
            """
            )
            return False

        return True

    def _calculate_margin(self, symbol_info, volume, price):
        """Calculate required margin for trade"""
        contract_size = symbol_info.trade_contract_size
        margin_initial = (
            symbol_info.margin_initial if symbol_info.margin_initial != 0 else 1
        )

        # Basic margin calculation
        margin = price * volume * contract_size * margin_initial

        # Convert to account currency if needed
        if symbol_info.currency_profit != mt5.account_info().currency:
            conversion_rate = self._get_conversion_rate(symbol_info.currency_profit)
            margin *= conversion_rate

        return margin

    def _get_conversion_rate(self, currency):
        """Get conversion rate to account currency"""
        account_currency = mt5.account_info().currency
        if currency == account_currency:
            return 1

        # Try direct conversion pair
        conversion_symbol = f"{currency}{account_currency}"
        rate = mt5.symbol_info_tick(conversion_symbol)
        if rate:
            return rate.bid

        # Try inverse pair
        conversion_symbol = f"{account_currency}{currency}"
        rate = mt5.symbol_info_tick(conversion_symbol)
        if rate:
            return 1 / rate.ask

        self.logger.warning(f"Could not find conversion rate for {currency}")
        return 1

    def _validate_tick_info(self, tick, symbol):
        """Validate tick information before placing an order"""
        if not tick:
            self.logger.error(f"Failed to get tick data for {symbol}")
            return False

        if tick.bid == 0 or tick.ask == 0:
            self.logger.error(f"Invalid bid/ask prices for {symbol}")
            return False

        return True

    def _adjust_volume(self, volume, symbol_info):
        """Adjust trading volume to match symbol limits"""
        volume = max(volume, symbol_info.volume_min)
        volume = min(volume, symbol_info.volume_max)
        return round(volume, 2)

    def _calculate_order_parameters(self, direction, tick, atr):
        """Calculate order price, stop loss, and take profit levels"""
        spread = tick.ask - tick.bid

        if direction == "buy":
            price = tick.ask
            sl = price - (atr * 2)
            tp = price + (atr * 3)
        else:  # sell
            price = tick.bid
            sl = price + (atr * 2)
            tp = price - (atr * 3)

        return price, sl, tp

    def _create_order_request(self, symbol, direction, volume, price, sl, tp, filling_type):
        """Create an order request with specified parameters"""
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 234000,
            "comment": "python",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

    def _send_order(self, request):
        """Send order to MT5 and handle the response"""
        result = mt5.order_send(request)
        if result is None:
            self.logger.error(f"Failed to send order: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed. Error code: {result.retcode}")
            return None

        return result

    def _log_successful_order(self, symbol, direction, volume, price, sl, tp):
        """Log details of successful order placement"""
        self.logger.info(
            f"""
            Order placed successfully:
            Symbol: {symbol}
            Direction: {direction}
            Volume: {volume}
            Price: {price}
            SL: {sl}
            TP: {tp}
            """
        )

    def _update_trading_stats(self, trading_stats, symbol, direction, result):
        """Update trading statistics after successful order"""
        if trading_stats:
            trading_stats.update_order_stats(symbol, direction, result.volume)

    def place_order(
        self, symbol, direction, atr, volume, trading_stats=None, is_ml_signal=False
    ):
        """Enhanced order placement with proper filling mode and money validation"""
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
            Sending Order:
            Symbol: {symbol}
            Direction: {direction}
            Volume: {volume}
            Price: {price}
            SL: {sl}
            TP: {tp}
            Filling Mode: {filling_mode}
        """
        )

        result = self._send_order(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self._log_successful_order(symbol, direction, volume, price, sl, tp)
            if trading_stats:
                self._update_trading_stats(trading_stats, symbol, direction, result)
            return True

        return False
