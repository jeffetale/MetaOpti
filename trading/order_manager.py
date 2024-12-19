# trading/order_manager.py

import logging
from config import mt5

class OrderManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def place_order(
        self, symbol, direction, atr, volume, trading_stats=None, is_ml_signal=False
    ):
        """Enhanced order placement with multiple filling modes and error handling"""
        symbol_info = mt5.symbol_info(symbol)
        if not self._validate_symbol_info(symbol_info, symbol):
            return False

        tick = mt5.symbol_info_tick(symbol)
        if not self._validate_tick_info(tick, symbol):
            return False

        # Adjust volume and calculate order parameters
        volume = self._adjust_volume(volume, symbol_info)
        price, sl, tp = self._calculate_order_parameters(direction, tick, atr)

        # Try different filling modes
        for filling_type in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC]:
            request = self._create_order_request(
                symbol, direction, volume, price, sl, tp, filling_type
            )
            result = self._send_order(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self._log_successful_order(symbol, direction, volume, price, sl, tp)
                if trading_stats:
                    self._update_trading_stats(trading_stats, symbol, direction, result)
                return True

        return False

    def close_position(self, position):
        """Enhanced position closing with retry mechanism"""
        try:
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                self.logger.error(f"Failed to get tick data for {position.symbol}")
                return False

            request = self._create_close_request(position, tick)

            # Try closing with different filling modes
            for filling_type in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC]:
                request["type_filling"] = filling_type
                result = mt5.order_send(request)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"Successfully closed position {position.ticket}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error closing position {position.ticket}: {e}")
            return False

    def modify_position(self, position, new_sl=None, new_tp=None):
        """Modify existing position parameters"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl if new_sl is not None else position.sl,
            "tp": new_tp if new_tp is not None else position.tp,
        }

        result = mt5.order_send(request)
        success = result and result.retcode == mt5.TRADE_RETCODE_DONE

        if success:
            self.logger.info(
                f"Modified position {position.ticket}: SL={new_sl}, TP={new_tp}"
            )
        else:
            self.logger.error(f"Failed to modify position {position.ticket}")

        return success
