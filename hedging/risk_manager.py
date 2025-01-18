# hedging/risk_manager.py

from typing import Dict
import logging
from hedging.models import HedgedPosition, MarketContext
from config import mt5
from logging_config import EmojiLogger
from hedging.config import HedgingConfig


class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.symbol_stats: Dict[str, Dict] = {}

    def _check_position_safety(self, hedged_pos: HedgedPosition) -> None:
        """Monitor and manage risk for hedged positions, especially main positions going against the market"""
        try:
            main_pos = mt5.positions_get(ticket=hedged_pos.main_ticket)
            hedge_pos = mt5.positions_get(ticket=hedged_pos.hedge_ticket)

            if not main_pos or not hedge_pos:
                return

            main_pos = main_pos[0]
            hedge_pos = hedge_pos[0]

            # Calculate profit ratios
            main_profit_ratio = main_pos.profit / (
                main_pos.volume * 100
            )  # Normalize by position size
            hedge_profit_ratio = hedge_pos.profit / (hedge_pos.volume * 100)

            # Calculate position loss percentage
            main_pos_loss_percent = (
                main_pos.profit / (main_pos.price_open * main_pos.volume * 100)
            ) * 100

            # Define risk thresholds
            PROFIT_DISPARITY_THRESHOLD = 2.0  # Hedge is making 2x more profit per lot
            MAX_LOSS_THRESHOLD = -0.5  # Maximum -0.5% loss on main position
            QUICK_EXIT_THRESHOLD = -1.0  # Emergency exit at -1% loss

            # Case 1: Position exceeds max loss threshold but not yet at quick exit level
            if MAX_LOSS_THRESHOLD >= main_pos_loss_percent > QUICK_EXIT_THRESHOLD:
                current_price = (
                    mt5.symbol_info_tick(main_pos.symbol).ask
                    if main_pos.type == mt5.ORDER_TYPE_BUY
                    else mt5.symbol_info_tick(main_pos.symbol).bid
                )

                # Set very tight stop loss to prevent further losses
                new_sl = self._calculate_defensive_stop(
                    main_pos, current_price, aggressive=True
                )
                self.order_manager.modify_position_sl_tp(main_pos.ticket, new_sl=new_sl)

                self.logger.warning(
                    f"Max loss threshold reached for main position {main_pos.ticket} "
                    f"Loss: {main_pos_loss_percent:.2f}% | Setting tight stop at: {new_sl:.5f}"
                )

            # Case 2: Hedge is profiting significantly while main is losing
            elif (
                hedge_profit_ratio > abs(main_profit_ratio) * PROFIT_DISPARITY_THRESHOLD
                and main_pos.profit < 0
            ):

                current_price = (
                    mt5.symbol_info_tick(main_pos.symbol).ask
                    if main_pos.type == mt5.ORDER_TYPE_BUY
                    else mt5.symbol_info_tick(main_pos.symbol).bid
                )

                # Set moderately tight stop loss
                new_sl = self._calculate_defensive_stop(
                    main_pos, current_price, aggressive=False
                )
                self.order_manager.modify_position_sl_tp(main_pos.ticket, new_sl=new_sl)

                self.logger.warning(
                    f"Setting defensive stop loss for losing main position {main_pos.ticket} "
                    f"Current loss: {main_pos.profit:.2f} | New SL: {new_sl:.5f}"
                )

            # Case 3: Emergency exit if loss exceeds threshold
            elif main_pos_loss_percent <= QUICK_EXIT_THRESHOLD:
                self.logger.warning(
                    f"Emergency closing main position {main_pos.ticket} "
                    f"Loss exceeded quick exit threshold: {main_pos_loss_percent:.2f}%"
                )
                self.order_manager.close_position(main_pos)

                # Adjust hedge position if needed
                if hedge_pos.profit > 0:
                    # Move stop loss to secure some profit
                    new_hedge_sl = self._calculate_profit_lock_level(hedge_pos)
                    self.order_manager.modify_position_sl_tp(
                        hedge_pos.ticket, new_sl=new_hedge_sl
                    )

        except Exception as e:
            self.logger.error(
                f"Error in position safety check: {str(e)}", exc_info=True
            )

    def _calculate_defensive_stop(
        self, position, current_price: float, aggressive: bool = False
    ) -> float:
        """Calculate defensive stop loss level for losing positions"""
        atr = self._calculate_atr(position.symbol)
        if not atr:
            return position.sl  # Keep existing stop if ATR calculation fails

        # Use tighter stop distance if aggressive mode is enabled
        stop_distance = atr * (
            0.3 if aggressive else 0.5
        )  # Tighter stop for aggressive mode

        if position.type == mt5.ORDER_TYPE_BUY:
            return max(current_price - stop_distance, position.sl)
        else:
            return min(current_price + stop_distance, position.sl)

    def _calculate_profit_lock_level(self, position) -> float:
        """Calculate level to lock in profits for hedge positions"""
        profit_ticks = position.profit / (
            position.volume * 100
        )  # Approximate ticks of profit
        lock_ratio = 0.6  # Lock in 60% of current profit

        if position.type == mt5.ORDER_TYPE_BUY:
            return position.price_open + (profit_ticks * lock_ratio)
        else:
            return position.price_open - (profit_ticks * lock_ratio)

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
