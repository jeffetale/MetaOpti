# models/position_manager.py

from hedging.models import HedgedPosition
import logging
from config import mt5
from logging_config import EmojiLogger
from hedging.config import HedgingConfig

class PositionManager:
    def __init__(self, order_manager):
        self.logger = logging.getLogger(__name__)
        self.order_manager = order_manager

    def _calculate_position_size(
        self, symbol: str, base_volume: float, confidence: float
    ) -> float:
        """Calculate position size based on performance and market conditions"""
        try:
            # Get symbol info for volume constraints
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"Symbol info not found for {symbol}"
                    )
                )
                return 0.0

            # Calculate volume with previous logic
            winning_streak_factor = 1.0 + (
                0.1 * self.symbol_stats[symbol]["winning_streak"]
            )
            success_rate_factor = (
                1.0 + (self.symbol_stats[symbol]["success_rate"] - 0.5)
                if self.symbol_stats[symbol]["success_rate"] > 0.5
                else 1.0
            )
            risk_factor = (
                max(0.5, 1.0 - (abs(self.current_drawdown) / self.max_drawdown))
                if self.max_drawdown != 0
                else 1.0
            )
            confidence_factor = (confidence - self.MIN_CONFIDENCE_THRESHOLD) / (
                1 - self.MIN_CONFIDENCE_THRESHOLD
            )

            # Calculate raw volume
            raw_volume = (
                base_volume
                * winning_streak_factor
                * success_rate_factor
                * risk_factor
                * confidence_factor
            )

            # Normalize volume to symbol constraints
            steps = round(raw_volume / symbol_info.volume_step)
            normalized_volume = steps * symbol_info.volume_step

            # Ensure volume is within bounds
            final_volume = max(
                symbol_info.volume_min, min(normalized_volume, symbol_info.volume_max)
            )

            return final_volume

        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error calculating position size: {str(e)}"
                )
            )
            return 0.0

    def _calculate_trailing_activation_price(
        self, position_ticket: int, direction: str
    ) -> float:
        """Calculate price at which trailing stop gets activated"""
        # Get the actual position info from MT5
        position = mt5.positions_get(ticket=position_ticket)
        if not position or not position[0]:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR,
                    f"Could not get position info for ticket {position_ticket}",
                )
            )
            return 0.0

        position = position[0]  # Get the first position object
        entry_price = position.price_open
        activation_move = entry_price * (self.BASE_TRAILING_ACTIVATION * 1.5) / 100

        return (
            entry_price + activation_move
            if direction == "buy"
            else entry_price - activation_move
        )

    def _update_trailing_stop(self, position, hedged_position: HedgedPosition) -> None:
        """Update trailing stop with improved risk management"""
        try:
            if not position:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, "Position not found for trailing stop update"
                    )
                )
                return

            current_price = mt5.symbol_info_tick(position.symbol).bid
            stop_distance = current_price * self.BASE_TRAILING_STOP / 100

            self.logger.info(
                EmojiLogger.format_message(
                    EmojiLogger.TRAILING,
                    f"Updating trailing stop - Symbol: {position.symbol} | Current Price: {current_price:.5f} | Current SL: {position.sl:.5f}",
                )
            )

            # Initialize trailing_stop if None
            if hedged_position.trailing_stop is None:
                hedged_position.trailing_stop = (
                    position.sl
                    if position.sl
                    else (
                        current_price - stop_distance
                        if position.type == mt5.ORDER_TYPE_BUY
                        else current_price + stop_distance
                    )
                )

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

            if self._modify_sl_tp(position, new_sl, None):
                self.logger.info(
                    EmojiLogger.format_message(
                        EmojiLogger.SUCCESS,
                        f"Trailing stop updated - New SL: {new_sl:.5f} | Distance: {stop_distance:.5f}",
                    )
                )
            else:
                self.logger.warning(
                    EmojiLogger.format_message(
                        EmojiLogger.WARNING,
                        f"Failed to update trailing stop for ticket {position.ticket}",
                    )
                )

        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error in trailing stop management: {str(e)}"
                )
            )

    def _handle_stopped_position(
        self, hedged_pos, surviving_pos, closed_pos, symbol: str
    ):
        """Recovery strategy with dynamic position sizing and risk management"""
        try:
            # Calculate the loss from the closed position
            loss_amount = abs(closed_pos.profit)

            self.logger.info(
                EmojiLogger.format_message(
                    EmojiLogger.ALERT,
                    f"Position stopped out - Symbol: {symbol} | Loss: {loss_amount:.2f} | Ticket: {closed_pos.ticket}",
                )
            )

            # Get current market context
            market_context = self._get_market_context(symbol)
            self.logger.info(
                EmojiLogger.format_message(
                    EmojiLogger.MARKET,
                    f"Recovery market context - Regime: {market_context.market_regime} | Volatility: {market_context.volatility:.4f}",
                )
            )

            # Calculate base recovery volume with dynamic scaling
            base_recovery_volume = surviving_pos.volume

            # Scale recovery volume based on market conditions
            volume_multiplier = 1.0

            # Increase size more in trending markets
            if (
                market_context.market_regime == "TRENDING"
                and market_context.trend_strength > 25
            ):
                volume_multiplier *= 1.8

            # Reduce size in volatile markets
            elif market_context.market_regime == "VOLATILE":
                volume_multiplier *= 0.8

            # Consider winning streak
            if self.symbol_stats[symbol]["winning_streak"] > 2:
                volume_multiplier *= 1.2

            # Consider volatility
            if (
                market_context.volatility
                < self.symbol_stats[symbol]["volatility_history"][-20:].mean()
            ):
                volume_multiplier *= 1.2

            # Calculate final recovery volume
            recovery_volume = base_recovery_volume * volume_multiplier

            # Apply maximum position size limit
            max_allowed_volume = (
                surviving_pos.volume * 2.5
            )  # Never more than 2.5x original
            recovery_volume = min(recovery_volume, max_allowed_volume)

            self.logger.info(
                EmojiLogger.format_message(
                    EmojiLogger.INFO,
                    f"Recovery volume: {recovery_volume:.2f} | Base volume: {base_recovery_volume:.2f}",
                )
            )

            # Calculate dynamic stop loss based on ATR and market regime
            atr = self._calculate_atr(symbol)
            sl_multiplier = {"TRENDING": 1.5, "RANGING": 1.0, "VOLATILE": 0.8}.get(
                market_context.market_regime, 1.0
            )

            sl_distance = atr * sl_multiplier if atr else None

            # Calculate profit target based on risk-reward ratio
            risk_reward_ratio = 2.0  # Minimum 2:1 reward-to-risk
            tp_distance = sl_distance * risk_reward_ratio if sl_distance else None

            # Place recovery order
            recovery_direction = (
                "buy" if surviving_pos.type == mt5.ORDER_TYPE_BUY else "sell"
            )

            recovery_result = self.order_manager.place_hedged_order(
                symbol=symbol,
                direction=recovery_direction,
                volume=recovery_volume,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
                market_context=market_context,
            )

            if recovery_result:
                self.logger.info(
                    EmojiLogger.format_message(
                        EmojiLogger.INFO,
                        f"Placed recovery order: {recovery_result.order} "
                        f"Volume: {recovery_volume}, "
                        f"Market Regime: {market_context.market_regime}, "
                        f"Multiplier: {volume_multiplier}",
                    )
                )
                hedged_pos.recovery_ticket = recovery_result.order

                # Update position tracking
                hedged_pos.increased_position = True
                hedged_pos.current_phase = "RECOVERY"

            return True

        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error in recovery handling: {str(e)}"
                )
            )
            return False

    def _execute_partial_close(
        self, position, volume: float, hedged_pos: HedgedPosition
    ) -> None:
        """Execute a partial position close"""
        try:
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": volume,
                "type": (
                    mt5.ORDER_TYPE_SELL
                    if position.type == mt5.ORDER_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY
                ),
                "price": (
                    mt5.symbol_info_tick(position.symbol).bid
                    if position.type == mt5.ORDER_TYPE_BUY
                    else mt5.symbol_info_tick(position.symbol).ask
                ),
                "deviation": self.PRICE_DEVIATION_POINTS,
                "magic": position.magic,
                "comment": "partial_tp",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(close_request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"Partial close executed: {volume} lots at level {result.price}"
                )

        except Exception as e:
            self.logger.error(f"Error executing partial close: {str(e)}")
