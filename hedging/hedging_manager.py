# hedging/hedging_manager.py

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import mt5
from logging_config import setup_comprehensive_logging, EmojiLogger

setup_comprehensive_logging()


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
    profit_target_hit: bool = False
    increased_position: bool = False
    winning_streak: int = 0
    entry_volatility: float = 0.0
    last_increase_time: Optional[datetime] = None
    increase_count: int = 0
    tp_levels: List[float] = field(default_factory=list)
    tp_volumes: List[float] = field(default_factory=list)
    stop_loss: Optional[float] = None

@dataclass
class MarketContext:
    volatility: float
    trend_strength: float
    recent_volume: float
    avg_spread: float
    market_regime: str  # 'TRENDING', 'RANGING', 'VOLATILE'


class HedgingManager:
    def __init__(self, order_manager, ml_predictor):
        self.logger = logging.getLogger(__name__)
        self.order_manager = order_manager
        self.ml_predictor = ml_predictor
        self.hedged_positions: Dict[str, List[HedgedPosition]] = {}
        self.symbol_stats: Dict[str, Dict] = {}  # Track symbol-specific performance

        # Dynamic configuration
        self.BASE_VOLUME = 0.1
        self.VOLUME_MULTIPLIER = 5.0
        self.MAX_POSITIONS = 15
        self.BASE_TRAILING_ACTIVATION = 0.8  # Activate trailing stop after 0.8 % profit
        self.BASE_TRAILING_STOP = 0.5  # Set trailing stop at 0.5 % profit
        self.BASE_PROFIT_TARGET = 20.0  # Profit targets at 20 usd
        self.POSITION_INCREASE_FACTOR = 1.5  # Increase position size by 50%
        self.MAX_LOSS_PER_PAIR = -40.0  # Stop losses at -40 usd
        self.TRAILING_ACTIVATION_PERCENT = 3.0  # Activate trailing at 3% profit
        self.LOSS_TIGHTENING_PERCENT = 5.0  # Tighten stops at 5% loss
        self.POSITION_INCREASE_THRESHOLD = 3.0  # Increase position size at 3% profit
        self.STOP_TIGHTENING_FACTOR = 0.7  # How much to tighten stops (70% closer)
        self.MAX_WINNING_STREAK_FACTOR = (
            2.5  # Max lot size multiplier limit for winning streak
        )
        self.VOLATILITY_ADJUSTMENT_FACTOR = 1.2  #
        self.MIN_CONFIDENCE_THRESHOLD = (
            0.57  # Minimum model confidence for opening positions
        )
        self.MIN_PROFIT_TICKS = 10  # Minimum profit in ticks before considering exit
        self.PROFIT_LOCK_THRESHOLD = (
            0.3  # Lock in profits when position reaches 0.3% gain
        )
        self.PARTIAL_CLOSE_THRESHOLD = 0.5  # Start partial closes at 0.5% gain
        self.PARTIAL_CLOSE_RATIO = 0.5  # Close 50% of position on first target
        self.TREND_CONTINUATION_THRESHOLD = (
            0.7  # Confidence needed to maintain position
        )
        self.MAX_ADVERSE_MOVE = 0.2  # Maximum allowable adverse move before closing

        # Performance tracking
        self.winning_trades = 0
        self.total_trades = 0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

    def _initialize_symbol_stats(self, symbol: str) -> None:
        """Initialize or update symbol-specific statistics"""
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {
                "winning_streak": 0,
                "losing_streak": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "volatility_history": [],
                "success_rate": 0.0,
                "total_trades": 0,
            }

    def manage_hedged_positions(self, symbol: str) -> None:
        if symbol not in self.hedged_positions:
            self.hedged_positions[symbol] = []

        if len(self.hedged_positions[symbol]) < self.MAX_POSITIONS:
            self._try_open_hedged_position(symbol)

        self._manage_existing_positions(symbol)

    def _determine_market_regime(
        self, volatility: float, trend_strength: float, volume: float
    ) -> str:
        """Classify current market regime"""
        if trend_strength > 25 and volume > self.symbol_stats.get("avg_volume", 0):
            return "TRENDING"
        elif volatility > 1.5 * self.symbol_stats.get("avg_volatility", 0):
            return "VOLATILE"
        else:
            return "RANGING"

    def _calculate_dynamic_profit_target(
        self, symbol: str, market_context: MarketContext
    ) -> float:
        """Calculate profit target based on market conditions"""
        base_target = self.BASE_PROFIT_TARGET

        # Adjust for volatility
        volatility_multiplier = (
            market_context.volatility
            / self.symbol_stats[symbol]["volatility_history"][-20:].mean()
        )

        # Adjust for market regime
        regime_multiplier = {
            "TRENDING": 1.5,  # More room to run in trends
            "VOLATILE": 0.8,  # Tighter targets in volatile conditions
            "RANGING": 1.0,  # Standard targets in ranging markets
        }.get(market_context.market_regime, 1.0)

        # Adjust for winning streak
        streak_multiplier = min(
            1 + (self.symbol_stats[symbol]["winning_streak"] * 0.1),
            self.MAX_WINNING_STREAK_FACTOR,
        )

        return (
            base_target * volatility_multiplier * regime_multiplier * streak_multiplier
        )

    def _calculate_position_size(
        self, symbol: str, base_volume: float, confidence: float
    ) -> float:
        """Calculate position size based on performance and market conditions"""
        stats = self.symbol_stats[symbol]

        # Base scaling factors
        winning_streak_factor = 1.0 + (0.1 * stats["winning_streak"])
        success_rate_factor = (
            1.0 + (stats["success_rate"] - 0.5) if stats["success_rate"] > 0.5 else 1.0
        )

        # Risk-based scaling
        risk_factor = (
            max(0.5, 1.0 - (abs(self.current_drawdown) / self.max_drawdown))
            if self.max_drawdown != 0
            else 1.0
        )

        # Confidence scaling
        confidence_factor = (confidence - self.MIN_CONFIDENCE_THRESHOLD) / (
            1 - self.MIN_CONFIDENCE_THRESHOLD
        )

        final_volume = (
            base_volume
            * winning_streak_factor
            * success_rate_factor
            * risk_factor
            * confidence_factor
        )

        calculated_volume = super()._calculate_position_size(
            symbol, base_volume, confidence
        )
        logging.info(
            f"""
            Position size calculation for {symbol}:
            Base volume: {base_volume}
            Confidence: {confidence}
            Final volume: {calculated_volume}
            """
        )

        return round(min(final_volume, base_volume * self.MAX_WINNING_STREAK_FACTOR), 2)

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
                    stop_loss=stop_loss,  # Added stop loss
                    tp_levels=[tp1, tp2, tp3],
                    tp_volumes = [
                    main_volume * 0.5,  # Take more profit at first level (increased from 0.4)
                    main_volume * 0.3,  # Keep middle level the same
                    main_volume * 0.2,  # Reduce last level (reduced from 0.3)
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

    def _manage_existing_positions(self, symbol: str) -> None:
        for hedged_pos in self.hedged_positions[symbol]:
            try:
                main_pos = mt5.positions_get(ticket=hedged_pos.main_ticket)
                if not main_pos:
                    continue

                main_pos = main_pos[0]
                current_price = (
                    mt5.symbol_info_tick(symbol).bid
                    if main_pos.type == mt5.ORDER_TYPE_BUY
                    else mt5.symbol_info_tick(symbol).ask
                )

                # Check if we've hit any take profit levels
                for i, (tp_level, tp_volume) in enumerate(
                    zip(hedged_pos.tp_levels, hedged_pos.tp_volumes)
                ):
                    if (
                        main_pos.type == mt5.ORDER_TYPE_BUY
                        and current_price >= tp_level
                    ) or (
                        main_pos.type == mt5.ORDER_TYPE_SELL
                        and current_price <= tp_level
                    ):

                        # Close partial position
                        if tp_volume > 0:
                            self._execute_partial_close(main_pos, tp_volume, hedged_pos)
                            hedged_pos.tp_volumes[i] = 0  # Mark this level as taken

                            # Check if we should open a new position
                            if self._should_reenter_position(symbol, main_pos.type):
                                self._open_continuation_position(
                                    symbol, main_pos.type, tp_volume
                                )

                # Lock in profits with trailing stop
                if (
                    self._calculate_profit_percent(main_pos)
                    >= self.PROFIT_LOCK_THRESHOLD
                ):
                    self._update_trailing_stop(main_pos, hedged_pos)

                # Protect against adverse moves
                if self._calculate_profit_percent(main_pos) >= self.MAX_ADVERSE_MOVE:
                    new_sl = current_price * (
                        0.999 if main_pos.type == mt5.ORDER_TYPE_BUY else 1.001
                    )
                    self.order_manager.modify_position_sl_tp(
                        main_pos.ticket, new_sl=new_sl
                    )

            except Exception as e:
                self.logger.error(f"Error managing position: {str(e)}")

    def _should_reenter_position(self, symbol: str, position_type) -> bool:
        """Determine if we should open a new position after taking profit"""
        try:
            # Get ML model confidence
            buy_confidence, sell_confidence = (
                self.ml_predictor.get_directional_confidence()
            )

            # Get market context
            market_context = self._get_market_context(symbol)

            # Check if trend is still valid
            confidence = (
                buy_confidence
                if position_type == mt5.ORDER_TYPE_BUY
                else sell_confidence
            )

            return (
                confidence > self.TREND_CONTINUATION_THRESHOLD
                and market_context.market_regime == "TRENDING"
                and market_context.trend_strength > 25
            )

        except Exception as e:
            self.logger.error(f"Error checking reentry conditions: {str(e)}")
            return False

    def _open_continuation_position(
        self, symbol: str, position_type, volume: float
    ) -> None:
        """Open a new position in the same direction after taking profit"""
        try:
            direction = "buy" if position_type == mt5.ORDER_TYPE_BUY else "sell"

            # Calculate new position parameters
            atr = self._calculate_atr(symbol)
            if not atr:
                return

            market_context = self._get_market_context(symbol)

            # Place new position with tighter stops since we're in profit
            result = self.order_manager.place_hedged_order(
                symbol=symbol,
                direction=direction,
                volume=volume,
                atr=atr * 0.8,  # Tighter stop for continuation trade
                market_context=market_context,
            )

            if result:
                self.logger.info(f"Opened continuation position: {result.order}")

        except Exception as e:
            self.logger.error(f"Error opening continuation position: {str(e)}")

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
                "deviation": self.order_manager.PRICE_DEVIATION_POINTS,
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

    def _calculate_profit_percent(self, position) -> float:
        """Calculate profit as a percentage of position value"""
        if not position:
            return 0.0

        contract_size = mt5.symbol_info(position.symbol).trade_contract_size
        position_value = position.price_open * position.volume * contract_size
        return (position.profit / position_value) * 100 if position_value != 0 else 0

    def _increase_position_size(
        self, symbol: str, hedged_pos, main_pos, hedge_pos
    ) -> None:
        """Increase position size for profitable positions"""
        if hedged_pos.last_increase_time:
            time_since_last = datetime.now() - hedged_pos.last_increase_time
            if time_since_last.total_seconds() < 600:  # 10 min minimum
                return
        try:
            # Determine which position is profitable
            profitable_pos = (
                main_pos
                if (
                    main_pos
                    and self._calculate_profit_percent(main_pos)
                    >= self.POSITION_INCREASE_THRESHOLD
                )
                else hedge_pos
            )

            if not profitable_pos:
                return

            # Calculate increase factor based on current profit
            profit_percent = self._calculate_profit_percent(profitable_pos)
            increase_factor = 1.5  # Base increase

            if profit_percent >= 10:
                increase_factor = 2.0
            if profit_percent >= 20:
                increase_factor = 2.5

            new_volume = profitable_pos.volume * increase_factor

            # Get current market context and ATR
            market_context = self._get_market_context(symbol)
            atr = self._calculate_atr(symbol)

            if not atr:
                self.logger.error("Could not calculate ATR for position increase")
                return

            # Open new position in same direction
            direction = "buy" if profitable_pos.type == mt5.ORDER_TYPE_BUY else "sell"
            result = self.order_manager.place_hedged_order(
                symbol=symbol,
                direction=direction,
                volume=new_volume,
                atr=atr,
                market_context=market_context,
            )

            if result:
                hedged_pos.increased_position = True
                self.logger.info(
                    f"Successfully increased position size for {symbol} to {new_volume}"
                )

        except Exception as e:
            self.logger.error(f"Error increasing position size: {str(e)}")

    def _tighten_stop_loss(self, position) -> None:
        """Tighten stop loss for losing positions"""
        try:
            if not position or not position.sl:
                return

            current_price = (
                mt5.symbol_info_tick(position.symbol).bid
                if position.type == mt5.ORDER_TYPE_BUY
                else mt5.symbol_info_tick(position.symbol).ask
            )

            # Calculate tighter stop loss
            if position.type == mt5.ORDER_TYPE_BUY:
                current_stop_distance = position.sl - position.price_open
                new_stop_distance = current_stop_distance * self.STOP_TIGHTENING_FACTOR
                new_sl = current_price - new_stop_distance
                if new_sl > position.sl:
                    self.order_manager.modify_position_sl_tp(
                        position.ticket, new_sl=new_sl
                    )
            else:
                current_stop_distance = position.price_open - position.sl
                new_stop_distance = current_stop_distance * self.STOP_TIGHTENING_FACTOR
                new_sl = current_price + new_stop_distance
                if new_sl < position.sl:
                    self.order_manager.modify_position_sl_tp(
                        position.ticket, new_sl=new_sl
                    )

        except Exception as e:
            self.logger.error(f"Error tightening stop loss: {str(e)}")

    def _handle_max_loss_exceeded(self, hedged_pos, main_pos, hedge_pos, symbol: str):
        """Handling of maximum loss situations with recovery strategy"""
        try:
            # Calculate current ATR for new positions
            atr = self._calculate_atr(symbol)
            if not atr:
                self.logger.error("Could not calculate ATR for recovery position")
                return

            # Get market context
            market_context = self._get_market_context(symbol)
            self.logger.info(
                EmojiLogger.format_message(
                    EmojiLogger.MARKET,
                    f"Market Context - Regime: {market_context.market_regime} | Volatility: {market_context.volatility:.4f} | Trend Strength: {market_context.trend_strength:.2f}",
                )
            )

            # Determine which position is losing more
            main_profit = main_pos.profit if main_pos else 0
            hedge_profit = hedge_pos.profit if hedge_pos else 0

            losing_pos = main_pos if main_profit < hedge_profit else hedge_pos
            winning_pos = hedge_pos if main_profit < hedge_profit else main_pos

            # Close losing position
            if losing_pos:
                self.order_manager.close_position(losing_pos)

            if winning_pos:
                # Calculate recovery volume
                recovery_volume = winning_pos.volume * 1.5

                # Place recovery order with ATR-based stops
                winning_direction = (
                    "buy" if winning_pos.type == mt5.ORDER_TYPE_BUY else "sell"
                )
                recovery_result = self.order_manager.place_hedged_order(
                    symbol=symbol,
                    direction=winning_direction,
                    volume=recovery_volume,
                    atr=atr,
                    market_context=market_context,
                )

                if recovery_result:
                    self.logger.info(f"Placed recovery order: {recovery_result.order}")

        except Exception as e:
            self.logger.error(
                f"Error in _handle_max_loss_exceeded: {str(e)}", exc_info=True
            )

    def _apply_trailing_stop(self, position, hedged_pos) -> None:
        """Apply trailing stop to a profitable position"""
        try:
            if not position:
                return

            symbol_info = mt5.symbol_info(position.symbol)
            current_price = (
                mt5.symbol_info_tick(position.symbol).bid
                if position.type == mt5.ORDER_TYPE_BUY
                else mt5.symbol_info_tick(position.symbol).ask
            )

            # Calculate trailing stop distance (1/3 of current profit)
            price_delta = abs(current_price - position.price_open)
            trailing_distance = price_delta * 0.33

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - trailing_distance
                if not position.sl or new_sl > position.sl:
                    self.order_manager.modify_position_sl_tp(
                        position.ticket, new_sl=new_sl
                    )
            else:
                new_sl = current_price + trailing_distance
                if not position.sl or new_sl < position.sl:
                    self.order_manager.modify_position_sl_tp(
                        position.ticket, new_sl=new_sl
                    )

        except Exception as e:
            self.logger.error(f"Error applying trailing stop: {str(e)}")

    def _calculate_pair_profit(self, main_pos, hedge_pos):
        """Calculate total profit for a pair of positions"""
        main_profit = main_pos.profit if main_pos else 0
        hedge_profit = hedge_pos.profit if hedge_pos else 0
        return main_profit + hedge_profit

    def _handle_profit_target_reached(
        self, hedged_pos, main_pos, hedge_pos, symbol, market_context
    ):
        """Profit target handling with sophisticated exit criteria"""
        try:
            self.logger.info(
                EmojiLogger.format_message(
                    EmojiLogger.PROFIT,
                    f"Profit target reached for {symbol} - Main Profit: {main_pos.profit:.2f} | Hedge Profit: {hedge_pos.profit:.2f}",
                )
            )

            # Determine which position is profitable
            profitable_pos = (
                main_pos if (main_pos and main_pos.profit > 0) else hedge_pos
            )
            losing_pos = hedge_pos if profitable_pos == main_pos else main_pos

            if profitable_pos and losing_pos:
                # Update statistics
                self.symbol_stats[symbol]["winning_streak"] += 1
            self.winning_trades += 1
            self.total_trades += 1

            # Calculate new position size with dynamic scaling
            base_increase = self.POSITION_INCREASE_FACTOR
            streak_bonus = min(0.1 * self.symbol_stats[symbol]["winning_streak"], 0.5)
            volatility_adj = max(
                0.5, 1.0 - (market_context.volatility / hedged_pos.entry_volatility)
            )

            new_volume = (
                profitable_pos.volume * (base_increase + streak_bonus) * volatility_adj
            )

            # Close losing position
            self.order_manager.close_position(losing_pos)

            # Open new position with improved parameters
            direction = "buy" if profitable_pos.type == mt5.ORDER_TYPE_BUY else "sell"

            if market_context.market_regime != "VOLATILE":
                self.order_manager.place_hedged_order(
                    symbol=symbol,
                    direction=direction,
                    volume=new_volume,
                    atr=self._calculate_atr(symbol),
                    market_context=market_context,
                )

            hedged_pos.profit_target_hit = True
            hedged_pos.increased_position = True
            hedged_pos.current_phase = "TRAILING"

            # Update risk parameters
            self._update_risk_parameters(symbol, True, market_context)

        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error handling profit target: {str(e)}"
                )
            )

    def _calculate_trailing_activation_price(
        self, position_ticket: int, direction: str
    ) -> float:
        """Calculate price at which trailing stop gets activated"""
        # Get the actual position info from MT5
        position = mt5.positions_get(ticket=position_ticket)
        if not position or not position[0]:
            self.logger.error(EmojiLogger.format_message(
                EmojiLogger.ERROR, f"Could not get position info for ticket {position_ticket}"
            ))
            return 0.0

        position = position[0]  # Get the first position object
        entry_price = position.price_open
        activation_move = entry_price * (self.BASE_TRAILING_ACTIVATION * 1.5) / 100

        return (
            entry_price + activation_move
            if direction == "buy"
            else entry_price - activation_move
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

    def _convert_to_trailing_stop(
        self, position, hedged_position: HedgedPosition
    ) -> None:
        """Convert position to trailing stop mode with dynamic stop calculation"""
        current_price = mt5.symbol_info_tick(position.symbol).bid
        stop_distance = current_price * self.BASE_TRAILING_STOP / 100

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
        try:
            current_price = mt5.symbol_info_tick(position.symbol).bid
            stop_distance = current_price * self.BASE_TRAILING_STOP / 100

            self.logger.info(
                EmojiLogger.format_message(
                    EmojiLogger.TRAILING,
                    f"Updating trailing stop - Symbol: {position.symbol} | Current Price: {current_price:.5f} | Current SL: {position.sl:.5f}",
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

    def _update_risk_parameters(
        self, symbol: str, is_winning: bool, market_context: MarketContext
    ) -> None:
        """Update risk parameters based on performance and market conditions"""
        stats = self.symbol_stats[symbol]

        if is_winning:
            stats["winning_streak"] += 1
            stats["losing_streak"] = 0
            # Gradually increase risk tolerance
            self.MAX_POSITIONS = min(self.MAX_POSITIONS + 1, 15)
            self.MAX_LOSS_PER_PAIR *= 1.1  # Allow slightly larger drawdown
        else:
            stats["losing_streak"] += 1
            stats["winning_streak"] = 0
            # Reduce risk exposure
            self.MAX_POSITIONS = max(self.MAX_POSITIONS - 1, 5)
            self.MAX_LOSS_PER_PAIR *= 0.9  # Tighten drawdown limit

        # Update volatility history
        stats["volatility_history"].append(market_context.volatility)
        if len(stats["volatility_history"]) > 100:
            stats["volatility_history"].pop(0)

        # Update success rate
        stats["total_trades"] += 1
        stats["success_rate"] = self.winning_trades / max(self.total_trades, 1)

    def _calculate_adx(self, symbol: str, period: int = 14) -> float:
        """Calculate Average Directional Index for trend strength with proper error handling"""
        try:
            # Validate inputs
            if not symbol or period <= 0:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Invalid inputs for ADX calculation: symbol={symbol}, period={period}"))
                return 0.0

            # Verify symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Symbol {symbol} not found in MT5"))
                return 0.0

            # Get enough data for calculation
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period * 2 + 1)
            if rates is None or len(rates) < period * 2:
                self.logger.warning(EmojiLogger.format_message(EmojiLogger.WARNING, f"Insufficient data for ADX calculation for {symbol}"))
                return 0.0

            # Convert to numpy arrays
            try:
                high = np.array([rate["high"] for rate in rates], dtype=np.float64)
                low = np.array([rate["low"] for rate in rates], dtype=np.float64)
                close = np.array([rate["close"] for rate in rates], dtype=np.float64)
            except (KeyError, ValueError) as e:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Error converting price data: {str(e)}"))
                return 0.0

            # Validate price data
            if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"NaN values found in price data"))
                return 0.0

            # Ensure arrays are the correct length
            high = high[
                :-1
            ]  # Remove last element to match length after differentiation
            low = low[:-1]
            close = close[:-1]

            # Calculate True Range
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                try:
                    hl = high[i] - low[i]
                    hc = abs(high[i] - close[i - 1])
                    lc = abs(low[i] - close[i - 1])
                    tr[i] = max(hl, hc, lc)
                except IndexError as e:
                    self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Index error in TR calculation: {str(e)}"))
                    return 0.0

            # Calculate Directional Movement
            plus_dm = np.zeros(len(high))
            minus_dm = np.zeros(len(high))

            for i in range(1, len(high)):
                up_move = high[i] - high[i - 1]
                down_move = low[i - 1] - low[i]

                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                else:
                    plus_dm[i] = 0

                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
                else:
                    minus_dm[i] = 0

            # Calculate smoothed averages
            plus_period = plus_dm[-period:].mean()
            minus_period = minus_dm[-period:].mean()

            # Calculate DI with validation
            try:
                tr_period = tr[-period:].mean()
                if tr_period == 0:
                    self.logger.warning(EmojiLogger.format_message(EmojiLogger.WARNING, f"TR period is zero"))
                    return 0.0

                plus_di = plus_period / tr_period * 100 if tr_period > 0 else 0
                minus_di = minus_period / tr_period * 100 if tr_period > 0 else 0

                # Validate DI values
                if not (0 <= plus_di <= 100) or not (0 <= minus_di <= 100):
                    self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Invalid DI values: +DI={plus_di}, -DI={minus_di}"))
                    return 0.0

                # Calculate ADX
                dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0

                return float(dx)

            except Exception as e:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Error in DI/ADX calculation: {str(e)}"))
                return 0.0

        except Exception as e:
            self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Error calculating ADX for {symbol}: {str(e)}"))
            return 0.0

    def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate ATR with comprehensive error handling"""
        try:
            # Validate inputs
            if not symbol or period <= 0:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Invalid inputs for ATR calculation: symbol={symbol}, period={period}"))
                return None

            # Verify symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Symbol {symbol} not found in MT5"))
                return None

            # Get price data with timeout handling
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
            if rates is None or len(rates) < period + 1:
                self.logger.warning(EmojiLogger.format_message(EmojiLogger.WARNING, f"Insufficient data for ATR calculation for {symbol}"))
                return None

            # Convert to numpy arrays with validation
            try:
                high = np.array([rate["high"] for rate in rates], dtype=np.float64)
                low = np.array([rate["low"] for rate in rates], dtype=np.float64)
                close = np.array([rate["close"] for rate in rates], dtype=np.float64)
            except (KeyError, ValueError) as e:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Error converting price data for ATR: {str(e)}"))
                return None

            # Validate price data
            if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"NaN values found in price data for ATR"))
                return None

            # Calculate true range with validation
            tr = np.zeros(len(high))
            for i in range(1, len(tr)):
                try:
                    hl = high[i] - low[i]
                    hc = abs(high[i] - close[i - 1])
                    lc = abs(low[i] - close[i - 1])
                    tr[i] = max(hl, hc, lc)
                except IndexError as e:
                    self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Index error in ATR calculation: {str(e)}"))
                    return None

            # Calculate ATR with validation
            try:
                atr = tr[1:].mean()  # Exclude the first zero value

                # Validate ATR value
                if np.isnan(atr) or atr <= 0:
                    self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Invalid ATR value calculated: {atr}"))
                    return None

                return float(atr)

            except Exception as e:
                self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Error in final ATR calculation: {str(e)}"))
                return None

        except Exception as e:
            self.logger.error(EmojiLogger.format_message(EmojiLogger.ERROR, f"Error calculating ATR for {symbol}: {str(e)}"))
            return None

    def _get_market_context(self, symbol: str) -> MarketContext:
        """Analyze current market conditions with proper error handling"""
        try:
            # Calculate recent volatility
            atr = self._calculate_atr(symbol)
            current_tick = mt5.symbol_info_tick(symbol)

            if atr is None or current_tick is None:
                self.logger.warning(f"Unable to get market context for {symbol}")
                return MarketContext(
                    volatility=0.0,
                    trend_strength=0.0,
                    recent_volume=0.0,
                    avg_spread=0.0,
                    market_regime="RANGING",  # Default to ranging when data is insufficient
                )

            volatility = atr / current_tick.bid if current_tick.bid != 0 else 0.0

            # Calculate trend strength using ADX
            adx = self._calculate_adx(symbol)

            # Get recent volume with error handling
            try:
                volume_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
                recent_volume = (
                    float(np.mean([rate["tick_volume"] for rate in volume_data]))
                    if volume_data is not None
                    else 0.0
                )
            except Exception as e:
                self.logger.warning(f"Error calculating recent volume: {str(e)}")
                recent_volume = 0.0

            # Calculate average spread
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                avg_spread = 0.0
            else:
                avg_spread = symbol_info.spread * symbol_info.point

            # Determine market regime with safe defaults
            market_regime = self._determine_market_regime(
                volatility, adx, recent_volume
            )

            return MarketContext(
                volatility=volatility,
                trend_strength=adx,
                recent_volume=recent_volume,
                avg_spread=avg_spread,
                market_regime=market_regime,
            )

        except Exception as e:
            self.logger.error(f"Error getting market context for {symbol}: {str(e)}")
            return MarketContext(
                volatility=0.0,
                trend_strength=0.0,
                recent_volume=0.0,
                avg_spread=0.0,
                market_regime="RANGING",
            )
