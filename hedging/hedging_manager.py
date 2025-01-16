# hedging/hedging_manager.py

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import mt5
from logging_config import setup_comprehensive_logging

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
        self.MAX_POSITIONS = 10
        self.BASE_TRAILING_ACTIVATION = 0.5  # Activate trailing stop after 0.5 % profit
        self.BASE_TRAILING_STOP = 0.3
        self.BASE_PROFIT_TARGET = 10.0
        self.POSITION_INCREASE_FACTOR = 1.5
        self.MAX_LOSS_PER_PAIR = -5.0
        self.MAX_WINNING_STREAK_FACTOR = 2.5
        self.VOLATILITY_ADJUSTMENT_FACTOR = 1.2
        self.MIN_CONFIDENCE_THRESHOLD = 0.57

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

        calculated_volume = super()._calculate_position_size(symbol, base_volume, confidence)
        logging.info(f"""
            Position size calculation for {symbol}:
            Base volume: {base_volume}
            Confidence: {confidence}
            Final volume: {calculated_volume}
            """)

        return round(min(final_volume, base_volume * self.MAX_WINNING_STREAK_FACTOR), 2)

    def _try_open_hedged_position(self, symbol: str) -> None:
        try:
            # Get ML directional confidence
            buy_confidence, sell_confidence = self.ml_predictor.get_directional_confidence()
            self.logger.info(f"Directional confidence - Buy: {buy_confidence}, Sell: {sell_confidence}")

            # Check confidence threshold
            min_confidence = self.MIN_CONFIDENCE_THRESHOLD
            if max(buy_confidence, sell_confidence) < min_confidence:
                self.logger.info(f"Insufficient confidence. Required: {min_confidence}")
                return

            # Calculate position sizes
            base_volume = 0.1
            main_direction = "buy" if buy_confidence > sell_confidence else "sell"
            main_volume = base_volume if main_direction == "buy" else base_volume
            hedge_volume = base_volume if main_direction == "sell" else base_volume

            self.logger.info(f"Calculated volumes - Main: {main_volume}, Hedge: {hedge_volume}")

            # Get ATR
            atr = self._calculate_atr(symbol)
            self.logger.info(f"Calculated ATR: {atr}")
            if not atr:
                self.logger.error("ATR calculation failed")
                return

            # Open main position with error handling
            self.logger.info(f"Attempting to open main position for {symbol}")
            main_result = self._open_main_position(symbol, main_direction, main_volume, atr)

            if main_result:
                self.logger.info(f"Main position opened successfully: {main_result}")
                main_position = mt5.positions_get(ticket=main_result.order)

                if not main_position:
                    self.logger.error("Could not get main position info after order placement")
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
                self.logger.error("Failed to open main position")

        except Exception as e:
            self.logger.error(f"Error in _try_open_hedged_position: {str(e)}", exc_info=True)

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

    def _manage_existing_positions(self, symbol: str) -> None:
        """Position management with recovery strategies"""
        positions_to_remove = []
        
        for hedged_pos in self.hedged_positions[symbol]:
            main_pos = mt5.positions_get(ticket=hedged_pos.main_ticket)
            hedge_pos = mt5.positions_get(ticket=hedged_pos.hedge_ticket)
            
            main_pos = main_pos[0] if main_pos else None
            hedge_pos = hedge_pos[0] if hedge_pos else None
            
            # Calculate total profit of the pair
            total_profit = self._calculate_pair_profit(main_pos, hedge_pos)
            
            # Handle cases where one position is closed
            if bool(main_pos) != bool(hedge_pos):  # XOR - one position exists, other doesn't
                surviving_pos = main_pos if main_pos else hedge_pos
                closed_pos = hedge_pos if main_pos else main_pos
                
                if not hasattr(hedged_pos, 'recovery_in_progress'):
                    hedged_pos.recovery_in_progress = True
                    success = self._handle_stopped_position(hedged_pos, surviving_pos, closed_pos, symbol)
                    if not success:
                        positions_to_remove.append(hedged_pos)
                    continue
            
            # Handle maximum loss exceeded
            if total_profit <= self.MAX_LOSS_PER_PAIR:
                self._handle_max_loss_exceeded(hedged_pos, main_pos, hedge_pos, symbol)
                positions_to_remove.append(hedged_pos)
                continue
                
            # Handle profit target reached
            if total_profit >= self.BASE_PROFIT_TARGET and not hedged_pos.profit_target_hit:
                self._handle_profit_target_reached(hedged_pos, main_pos, hedge_pos, symbol)
                continue
                
            # Regular position management
            if not main_pos and not hedge_pos:
                positions_to_remove.append(hedged_pos)
                
        # Clean up closed positions
        for pos in positions_to_remove:
            self.hedged_positions[symbol].remove(pos)

    def _calculate_pair_profit(self, main_pos, hedge_pos):
        """Calculate total profit for a pair of positions"""
        main_profit = main_pos.profit if main_pos else 0
        hedge_profit = hedge_pos.profit if hedge_pos else 0
        return main_profit + hedge_profit

    def _handle_max_loss_exceeded(self, hedged_pos, main_pos, hedge_pos, symbol: str):
        """Enhanced handling of maximum loss situations with recovery strategy"""
        try:
            # Determine which position is losing more
            main_profit = main_pos.profit if main_pos else 0
            hedge_profit = hedge_pos.profit if hedge_pos else 0
            
            losing_pos = main_pos if main_profit < hedge_profit else hedge_pos
            winning_pos = hedge_pos if main_profit < hedge_profit else main_pos
            
            # Close the losing position
            if losing_pos:
                self.order_manager.close_position(losing_pos)
                
            # Get market context for new positions
            market_context = self._get_market_context(symbol)
            
            # Calculate recovery parameters
            loss_amount = abs(losing_pos.profit)
            point = mt5.symbol_info(symbol).point
            recovery_pips = loss_amount / (losing_pos.volume * point)
            
            # Open new position in winning direction with increased volume
            winning_direction = "buy" if winning_pos.type == mt5.ORDER_TYPE_BUY else "sell"
            recovery_volume = winning_pos.volume * 2  # Double the volume for faster recovery
            
            # Calculate tight stop loss
            atr = self._calculate_atr(symbol)
            tight_sl_distance = atr * 0.25 if atr else None  # Very tight SL
            
            # Place recovery order
            recovery_result = self.order_manager.place_hedged_order(
                symbol=symbol,
                direction=winning_direction,
                volume=recovery_volume,
                sl_distance=tight_sl_distance,
                tp_distance=recovery_pips * 1.3,  # Set TP to recover losses plus 30%
                market_context=market_context
            )
            
            if recovery_result:
                self.logger.info(f"Placed loss recovery order: {recovery_result.order}")
                
            # Update risk parameters
            self._update_risk_parameters(symbol, False, market_context)
            
        except Exception as e:
            self.logger.error(f"Error in _handle_max_loss_exceeded: {str(e)}", exc_info=True)

    def _handle_profit_target_reached(
        self, hedged_pos, main_pos, hedge_pos, symbol, market_context
    ):
        """Enhanced profit target handling with sophisticated exit criteria"""
        # Determine which position is profitable
        profitable_pos = main_pos if (main_pos and main_pos.profit > 0) else hedge_pos
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
        activation_move = entry_price * self.BASE_TRAILING_ACTIVATION / 100

        return (
            entry_price + activation_move
            if direction == "buy"
            else entry_price - activation_move
        )
        
    def _handle_stopped_position(self, hedged_pos, surviving_pos, closed_pos, symbol: str):
        """Handle recovery strategy when one position of a hedged pair gets stopped out"""
        try:
            # Calculate the loss from the closed position
            loss_amount = abs(closed_pos.profit)
            
            # Get current market context
            market_context = self._get_market_context(symbol)
            
            # Calculate required pip value to recover loss
            point = mt5.symbol_info(symbol).point
            pip_value = loss_amount / (surviving_pos.volume * point)
            
            # Adjust surviving position's take profit to recover losses
            new_tp = None
            if surviving_pos.type == mt5.ORDER_TYPE_BUY:
                new_tp = surviving_pos.price_open + (pip_value * point * 1.2)  # Add 20% buffer
            else:
                new_tp = surviving_pos.price_open - (pip_value * point * 1.2)
                
            # Modify surviving position
            self._modify_sl_tp(surviving_pos, surviving_pos.sl, new_tp)
            
            # Open recovery position in profitable direction
            recovery_volume = surviving_pos.volume * 1.5  # Increase volume for faster recovery
            recovery_direction = "buy" if surviving_pos.type == mt5.ORDER_TYPE_BUY else "sell"
            
            # Calculate tight stop loss (using 1/3 of normal ATR)
            atr = self._calculate_atr(symbol)
            tight_sl_distance = atr * 0.33 if atr else None
            
            # Place recovery order with tight SL and higher TP
            recovery_result = self.order_manager.place_hedged_order(
                symbol=symbol,
                direction=recovery_direction,
                volume=recovery_volume,
                sl_distance=tight_sl_distance,
                tp_distance=pip_value * 1.5,  # Set higher TP for additional profit
                market_context=market_context
            )
            
            if recovery_result:
                self.logger.info(f"Placed recovery order: {recovery_result.order}")
                hedged_pos.recovery_ticket = recovery_result.order
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in _handle_stopped_position: {str(e)}", exc_info=True)
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
        current_price = mt5.symbol_info_tick(position.symbol).bid
        stop_distance = current_price * self.BASE_TRAILING_STOP / 100

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
            # Get enough data for calculation
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period * 2 + 1)
            if rates is None or len(rates) < period * 2:
                self.logger.warning(f"Insufficient data for ADX calculation for {symbol}")
                return 0.0

            # Convert to numpy arrays
            high = np.array([rate['high'] for rate in rates])
            low = np.array([rate['low'] for rate in rates])
            close = np.array([rate['close'] for rate in rates])

            # Ensure arrays are the correct length
            high = high[:-1]  # Remove last element to match length after differentiation
            low = low[:-1]
            close = close[:-1]

            # Calculate True Range
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)

            # Calculate Directional Movement
            plus_dm = np.zeros(len(high))
            minus_dm = np.zeros(len(high))

            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]

                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                else:
                    plus_dm[i] = 0

                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
                else:
                    minus_dm[i] = 0

            # Calculate smoothed averages
            tr_period = tr[-period:].mean()
            plus_period = plus_dm[-period:].mean()
            minus_period = minus_dm[-period:].mean()

            # Avoid division by zero
            if tr_period == 0:
                return 0.0

            # Calculate DI+ and DI-
            plus_di = (plus_period / tr_period) * 100
            minus_di = (minus_period / tr_period) * 100

            # Calculate DX and ADX
            dx = 0.0
            if (plus_di + minus_di) != 0:
                dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

            return float(dx)

        except Exception as e:
            self.logger.error(f"Error calculating ADX for {symbol}: {str(e)}")
            return 0.0

    def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range with proper error handling"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
            if rates is None or len(rates) < period + 1:
                self.logger.warning(f"Insufficient data for ATR calculation for {symbol}")
                return None

            # Convert to numpy arrays
            high = np.array([rate['high'] for rate in rates])
            low = np.array([rate['low'] for rate in rates])
            close = np.array([rate['close'] for rate in rates])

            # Calculate true range
            tr = np.zeros(len(high))
            for i in range(1, len(tr)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)

            # Calculate ATR
            atr = tr[1:].mean()  # Exclude the first zero value

            return float(atr)

        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {str(e)}")
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
                    market_regime="RANGING"  # Default to ranging when data is insufficient
                )

            volatility = atr / current_tick.bid if current_tick.bid != 0 else 0.0

            # Calculate trend strength using ADX
            adx = self._calculate_adx(symbol)

            # Get recent volume with error handling
            try:
                volume_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
                recent_volume = float(np.mean([rate['tick_volume'] for rate in volume_data])) if volume_data is not None else 0.0
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
            market_regime = self._determine_market_regime(volatility, adx, recent_volume)

            return MarketContext(
                volatility=volatility,
                trend_strength=adx,
                recent_volume=recent_volume,
                avg_spread=avg_spread,
                market_regime=market_regime
            )

        except Exception as e:
            self.logger.error(f"Error getting market context for {symbol}: {str(e)}")
            return MarketContext(
                volatility=0.0,
                trend_strength=0.0,
                recent_volume=0.0,
                avg_spread=0.0,
                market_regime="RANGING"
            )
