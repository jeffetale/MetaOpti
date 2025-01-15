# hedging/hedging_manager.py

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, List, Optional
from config import mt5


@dataclass
class HedgedPosition:
    main_ticket: int
    hedge_ticket: int
    symbol: str
    main_direction: str
    main_volume: float
    hedge_volume: float
    current_phase: str  # 'HEDGED', 'SINGLE', 'TRAILING'
    entry_time: datetime
    trailing_stop: Optional[float] = None
    trailing_activation_price: Optional[float] = None


class HedgingManager:
    def __init__(self, order_manager, ml_predictor):
        self.logger = logging.getLogger(__name__)
        self.order_manager = order_manager
        self.ml_predictor = ml_predictor
        self.hedged_positions: Dict[str, List[HedgedPosition]] = {}
        self.MAX_POSITIONS = 8
        self.TRAILING_ACTIVATION_PERCENT = 0.5  # Activate trailing after 0.5% profit
        self.TRAILING_STOP_PERCENT = 0.3  # Trailing stop at 0.3% below peak

    def manage_hedged_positions(self, symbol: str) -> None:
        """Main method to manage hedged positions for a symbol"""
        if symbol not in self.hedged_positions:
            self.hedged_positions[symbol] = []

        # Check if we can open new positions
        if len(self.hedged_positions[symbol]) < self.MAX_POSITIONS:
            self._try_open_hedged_position(symbol)

        # Manage existing positions
        self._manage_existing_positions(symbol)

    def _try_open_hedged_position(self, symbol: str) -> None:
        """Attempt to open a new hedged position pair with dynamic sizing"""
        # Get ML directional confidence
        buy_confidence, sell_confidence = self.ml_predictor.get_directional_confidence()

        # Only proceed if we have strong enough conviction
        min_confidence = 0.55
        if max(buy_confidence, sell_confidence) < min_confidence:
            return

        # Calculate position sizes based on confidence
        base_volume = 0.1  # Base trading volume
        buy_volume, sell_volume = self.ml_predictor.get_position_sizing(
            base_volume, buy_confidence, sell_confidence
        )

        # Determine main and hedge positions based on confidence
        main_direction = "buy" if buy_confidence > sell_confidence else "sell"
        main_volume = buy_volume if main_direction == "buy" else sell_volume
        hedge_volume = sell_volume if main_direction == "buy" else buy_volume

        atr = self._calculate_atr(symbol)
        if not atr:
            return

        # Open main position
        main_result = self._open_main_position(symbol, main_direction, main_volume, atr)

        if main_result:
            # Get the actual position info for the main position
            main_position = mt5.positions_get(ticket=main_result.order)
            if not main_position:
                self.logger.error(
                    f"Could not get main position info after order placement"
                )
                return

            main_position = main_position[0]  # Get the first position object

            # Open hedge position
            hedge_result = self._open_hedge_position(
                symbol,
                "sell" if main_direction == "buy" else "buy",
                hedge_volume,
                atr,
                main_position,
            )

            if hedge_result:
                # Calculate trailing activation price using the main position
                activation_price = self._calculate_trailing_activation_price(
                    main_result.order, main_direction
                )

                self.hedged_positions[symbol].append(
                    HedgedPosition(
                        main_ticket=main_result.order,  # Use order number from result
                        hedge_ticket=hedge_result.order,  # Use order number from result
                        symbol=symbol,
                        main_direction=main_direction,
                        main_volume=main_volume,
                        hedge_volume=hedge_volume,
                        current_phase="HEDGED",
                        entry_time=datetime.now(),
                        trailing_activation_price=activation_price,
                    )
                )
            else:
                # Close main position if hedge fails
                self.order_manager.close_position(main_position)

    def _open_main_position(
        self, symbol: str, direction: str, volume: float, atr: float
    ):
        """Open main position with wider stops"""
        return self.order_manager.place_hedged_order(
            symbol=symbol,
            direction=direction,
            volume=volume,
            atr=atr * 2.0,  # Using wider ATR multiplier for main position
        )

    def _open_hedge_position(
        self, symbol: str, direction: str, volume: float, atr: float, main_position
    ):
        """Open hedge position with tighter stops"""
        return self.order_manager.place_hedged_order(
            symbol=symbol,
            direction=direction,
            volume=volume,
            atr=atr * 1.5,  # Using tighter ATR multiplier for hedge
        )

    def _open_position(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl_distance: float,
        tp_distance: float,
    ):
        """Generic position opening with calculated levels"""
        tick = mt5.symbol_info_tick(symbol)

        entry_price = tick.ask if direction == "buy" else tick.bid

        if direction == "buy":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        return self.order_manager.place_hedged_order(
            symbol=symbol, direction=direction, volume=volume, sl=sl, tp=tp
        )

    def _manage_existing_positions(self, symbol: str) -> None:
        """Manage existing hedged positions with improved trailing stop logic"""
        positions_to_remove = []

        for hedged_pos in self.hedged_positions[symbol]:
            main_pos = mt5.positions_get(ticket=hedged_pos.main_ticket)
            hedge_pos = mt5.positions_get(ticket=hedged_pos.hedge_ticket)

            # Get the actual position objects if they exist
            main_pos = main_pos[0] if main_pos else None
            hedge_pos = hedge_pos[0] if hedge_pos else None

            if not main_pos or not hedge_pos:
                # One position closed, manage the remaining one
                surviving_pos = main_pos if main_pos else hedge_pos

                if surviving_pos:
                    if hedged_pos.current_phase == "HEDGED":
                        # Convert to trailing stop mode
                        self._convert_to_trailing_stop(surviving_pos, hedged_pos)
                    elif hedged_pos.current_phase == "TRAILING":
                        # Update trailing stop
                        self._update_trailing_stop(surviving_pos, hedged_pos)
                else:
                    positions_to_remove.append(hedged_pos)

        # Clean up closed positions
        for pos in positions_to_remove:
            self.hedged_positions[symbol].remove(pos)

    def _calculate_trailing_activation_price(
        self, position_ticket: int, direction: str
    ) -> float:
        """Calculate price at which trailing stop gets activated"""
        # Get the actual position info from MT5
        position = mt5.positions_get(ticket=position_ticket)
        if not position or not position[0]:
            self.logger.error(
                f"Could not get position info for ticket {position_ticket}"
            )
            return 0.0

        position = position[0]  # Get the first position object
        entry_price = position.price_open
        activation_move = entry_price * self.TRAILING_ACTIVATION_PERCENT / 100

        return (
            entry_price + activation_move
            if direction == "buy"
            else entry_price - activation_move
        )

    def _convert_to_trailing_stop(
        self, position, hedged_position: HedgedPosition
    ) -> None:
        """Convert position to trailing stop mode with dynamic stop calculation"""
        current_price = mt5.symbol_info_tick(position.symbol).bid
        stop_distance = current_price * self.TRAILING_STOP_PERCENT / 100

        if position.type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - stop_distance
        else:
            new_sl = current_price + stop_distance

        # Modify position
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": new_sl,
            "tp": None,  # Remove TP for trailing
        }

        if mt5.order_send(request).retcode == mt5.TRADE_RETCODE_DONE:
            hedged_position.current_phase = "TRAILING"
            hedged_position.trailing_stop = new_sl

    def _update_trailing_stop(self, position, hedged_position: HedgedPosition) -> None:
        """Update trailing stop with improved risk management"""
        current_price = mt5.symbol_info_tick(position.symbol).bid
        stop_distance = current_price * self.TRAILING_STOP_PERCENT / 100

        if position.type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - stop_distance
            if new_sl > hedged_position.trailing_stop:
                self._modify_sl_tp(position, new_sl, None)
                hedged_position.trailing_stop = new_sl
        else:
            new_sl = current_price + stop_distance
            if new_sl < hedged_position.trailing_stop:
                self._modify_sl_tp(position, new_sl, None)
                hedged_position.trailing_stop = new_sl

    def _modify_sl_tp(self, position, new_sl: float, new_tp: float) -> None:
        """Modify position's stop loss and take profit levels"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": new_sl,
            "tp": new_tp,
        }
        mt5.order_send(request)

    def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range for dynamic stop levels"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
        if rates is None:
            return None

        import numpy as np

        high = rates["high"]
        low = rates["low"]
        close = np.roll(rates["close"], 1)

        tr1 = np.abs(high - low)
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return float(np.mean(tr))
