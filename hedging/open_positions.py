# hedging/open_positions.py

import numpy as np
import logging
from datetime import datetime
from config import mt5
from hedging.config import HedgingConfig
from ml.predictor import MLPredictor
from logging_config import EmojiLogger
from hedging.hedging_manager import HedgedPosition
from hedging.config import HedgingConfig

class OpenPositions:
    def __init__(self, ml_predictor: MLPredictor, order_manager) -> None:
        self.ml_predictor = ml_predictor
        self.logger = logging.getLogger(__name__)
        self.MIN_CONFIDENCE_THRESHOLD = HedgingConfig.MIN_CONFIDENCE_THRESHOLD
        self.BASE_VOLUME = HedgingConfig.BASE_VOLUME
        self.VOLUME_MULTIPLIER = HedgingConfig.VOLUME_MULTIPLIER
        self.hedged_positions = {}              
        self.order_manager = order_manager

    def _try_open_hedged_position(self, symbol: str) -> None:
            try:
                # Get ML directional confidence
                buy_confidence, sell_confidence = (
                    self.ml_predictor.get_directional_confidence()
                )
                self.logger.info(
                    EmojiLogger.format_message(
                        EmojiLogger.AI,
                        f"Directional confidence analysis for {symbol} - Buy: {buy_confidence:.2f}, Sell: {sell_confidence:.2f}",
                    )
                )
                # Check confidence threshold
                min_confidence = self.MIN_CONFIDENCE_THRESHOLD
                if max(buy_confidence, sell_confidence) < min_confidence:
                    self.logger.info(
                        EmojiLogger.format_message(
                            EmojiLogger.AI,
                            f"Insufficient confidence for {symbol}. Required: {min_confidence:.2f}",
                        )
                    )
                    return

                # Get symbol information for volume validation
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    self.logger.error(f"Failed to get symbol info for {symbol}")
                    return

                # Get volume constraints
                min_volume = symbol_info.volume_min
                max_volume = symbol_info.volume_max
                volume_step = symbol_info.volume_step

                # Calculate position sizes with confidence-based scaling
                base_volume = max(min_volume, self.BASE_VOLUME)
                confidence_difference = abs(buy_confidence - sell_confidence)
                volume_multiplier = 1 + (confidence_difference * self.VOLUME_MULTIPLIER)

                main_direction = "buy" if buy_confidence > sell_confidence else "sell"

                # Calculate and normalize main volume
                raw_main_volume = base_volume * volume_multiplier
                steps = round(raw_main_volume / volume_step)
                main_volume = steps * volume_step
                main_volume = max(min_volume, min(main_volume, max_volume))

                # hedge at base volume
                hedge_volume = base_volume

                self.logger.info(
                    EmojiLogger.format_message(
                        EmojiLogger.INFO,
                        f"""Position sizing details:
                        Main Volume: {main_volume:.3f}
                        Hedge Volume: {hedge_volume:.3f}
                        Confidence Diff: {confidence_difference:.3f}
                        Min Volume: {min_volume:.3f}
                        Volume Step: {volume_step:.3f}"""
                    )
                )

                # Get ATR
                atr = self._calculate_atr(symbol)
                self.logger.info(f"Calculated ATR: {atr}")
                if not atr:
                    self.logger.error("ATR calculation failed")
                    return

                # Market Context
                market_context = self._get_market_context(symbol)
                self.logger.info(
                    EmojiLogger.format_message(
                        EmojiLogger.MARKET,
                        f"Market Context - Regime: {market_context.market_regime} | Volatility: {market_context.volatility:.4f} | Trend Strength: {market_context.trend_strength:.2f}",
                    )
                )

                # Open main position
                main_result = self._open_main_position(
                    symbol, main_direction, main_volume, atr
                )

                if not main_result:
                    self.logger.error(
                        EmojiLogger.format_message(
                            EmojiLogger.ERROR, "Failed to open main position"
                        )
                    )
                    return

                self.logger.info(
                    EmojiLogger.format_message(
                        EmojiLogger.SUCCESS,
                        f"Main position opened - Ticket: {main_result.order} | Direction: {main_direction.upper()}",
                    )
                )

                if main_result:
                    main_position = mt5.positions_get(ticket=main_result.order)

                    if not main_position:
                        self.logger.error(
                            EmojiLogger.format_message(
                                EmojiLogger.ERROR, "Failed to get main position"
                            )
                        )
                        return

                    main_position = main_position[0]

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

                # Calculate dynamic take profit and stop loss levels based on recent price action
                entry_price = main_position.price_open

                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
                if rates is not None:
                    recent_lows = [rate["low"] for rate in rates]
                    recent_highs = [rate["high"] for rate in rates]
                    avg_range = np.mean([h - l for h, l in zip(recent_highs, recent_lows)])

                    atr_stop_multiplier = 1.5  # Stop loss is 1.5 of ATR
                    tp_range_multiplier = 0.3

                    # Set stop losses and take profit levels
                    if main_direction == "buy":
                        stop_loss = entry_price - (atr * atr_stop_multiplier)  # Tight stop below entry
                        tp1 = entry_price + (avg_range * 0.3)  # First TP at 30% of avg range
                        tp2 = entry_price + (avg_range * 0.6) # Second TP at 60% of avg range
                        tp3 = entry_price + (avg_range * 0.9)  # Third TP at 90% of avg range
                    else:
                        stop_loss = entry_price + (atr * atr_stop_multiplier)  # Tight stop above entry
                        tp1 = entry_price - (avg_range * 0.3)
                        tp2 = entry_price - (avg_range * 0.6)
                        tp3 = entry_price - (avg_range * 0.9)

                    # Calculate potential loss at stop loss
                    potential_loss = abs(entry_price - stop_loss) * main_volume

                    # Log risk metrics
                    self.logger.info(
                        EmojiLogger.format_message(
                            EmojiLogger.WARNING,
                            f"""Risk Metrics:
                            Stop Loss: {stop_loss:.5f}
                            Potential Loss: {potential_loss:.2f}
                            Risk per ATR: {atr_stop_multiplier:.2f}"""
                        )
                    )

                    hedged_pos = HedgedPosition(
                        main_ticket=main_result.order,
                        hedge_ticket=hedge_result.order,
                        symbol=symbol,
                        main_direction=main_direction,
                        main_volume=main_volume,
                        hedge_volume=hedge_volume,
                        current_phase="HEDGED",
                        entry_time=datetime.now(),
                        stop_loss=stop_loss,
                        tp_levels=[tp1, tp2, tp3],
                        tp_volumes = [
                        main_volume * 0.6,  # Take more profit at first level (increased from 0.4)
                        main_volume * 0.25,  # Keep middle level the same
                        main_volume * 0.15,  # Reduce last level (reduced from 0.3)
                            ],
                        trailing_activation_price=activation_price,
                    )

                    self.hedged_positions[symbol].append(hedged_pos)
                    self.logger.info(
                        EmojiLogger.format_message(
                            EmojiLogger.SUCCESS,
                            f"Hedged position opened - Main Ticket: {main_result.order} | Hedge Ticket: {hedge_result.order}",
                        )
                    )
                else:
                    # Close main position if hedge fails
                    self.order_manager.close_position(main_position)
                    self.logger.error("Failed to open main position")

            except Exception as e:
                self.logger.error(
                    f"Error in _try_open_hedged_position: {str(e)}", exc_info=True
                )

    def _open_main_position(
        self, symbol: str, direction: str, volume: float, atr: float
    ):
        """Open main position with wider stops"""
        try:
            result = self.order_manager.place_hedged_order(
                symbol=symbol,
                direction=direction,
                volume=volume,
                atr=atr
                * 3.0,  # Using wider ATR multiplier for main position  
            )
            return result
        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error opening main position: {str(e)}"
                )
            )
            return None

    def _open_hedge_position(
        self, symbol: str, direction: str, volume: float, atr: float, main_position
    ):
        """Open hedge position with tighter stops"""
        try:
            result = self.order_manager.place_hedged_order(
                symbol=symbol,
                direction=direction,
                volume=volume,
                atr=atr * 2.0,  # Using tighter ATR multiplier for hedge
            )
            return result
        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error opening hedge position: {str(e)}"
                )
            )
            return None