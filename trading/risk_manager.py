# trading/risk_manager.py

import logging
from datetime import datetime, timedelta
from config import MIN_PROFIT_THRESHOLD, MIN_WIN_RATE, INITIAL_VOLUME
from models.trading_state import trading_state


class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def should_trade_symbol(self, symbol):
        """Determine if a symbol should be traded based on risk parameters"""
        state = trading_state.symbol_states[symbol]

        # Check if symbol is restricted
        if state.is_restricted:
            return False

        # Check consecutive losses
        if state.consecutive_losses >= trading_state.MAX_CONSECUTIVE_LOSSES:
            return False

        # Check win rate
        if state.trades_count > 10 and state.win_rate < MIN_WIN_RATE:
            # Allow trading again after cooling period
            cooling_period = datetime.now() - state.last_trade_time
            if cooling_period < timedelta(minutes=30):
                return False

        # Check recent performance
        if state.trades_history:
            recent_trades = state.trades_history[-3:]
            if sum(1 for profit in recent_trades if profit < 0) >= 2:
                # Two or more losses in last three trades
                cooling_period = datetime.now() - state.last_trade_time
                if cooling_period < timedelta(minutes=15):
                    return False

        # Check global account risk
        if trading_state.global_profit < -100:  # $100 maximum drawdown
            return False

        return True

    def calculate_win_rate(self, trades):
        """Calculate win rate from trade history"""
        if not trades:
            return 0
        winning_trades = sum(1 for profit in trades if profit > 0)
        return winning_trades / len(trades)

    def adjust_trading_parameters(self, symbol, profit, trading_state):
        """Dynamically adjust trading parameters based on performance"""
        state = trading_state.symbol_states[symbol]

        # Update trade history
        state.trades_count += 1
        state.trades_history.append(profit)
        state.recent_trade_directions.append("buy" if profit > 0 else "sell")

        # Reset consecutive losses if profitable
        if profit > 0:
            state.consecutive_losses = 0
        else:
            state.consecutive_losses += 1

        # Maintain trade direction memory size
        if len(state.recent_trade_directions) > state.trade_direction_memory_size:
            state.recent_trade_directions.pop(0)

        # Update performance metrics
        state.win_rate = self.calculate_win_rate(state.trades_history[-10:])

        # Volume adjustment
        self._adjust_volume(state)

        # Profit threshold adjustment
        self._adjust_profit_threshold(state, profit)

        return state

    def _adjust_volume(self, state):
        """Adjust trading volume based on performance"""
        if state.win_rate > 0.6:
            state.volume = min(state.volume * 1.2, INITIAL_VOLUME * 2)
        elif state.win_rate < 0.4:
            state.volume = max(state.volume * 0.8, INITIAL_VOLUME * 0.5)

    def _adjust_profit_threshold(self, state, profit):
        """Adjust profit threshold based on recent performance"""
        if profit > state.profit_threshold:
            state.profit_threshold *= 1.1
        elif profit < 0:
            state.profit_threshold = max(
                MIN_PROFIT_THRESHOLD, state.profit_threshold * 0.9
            )

    def calculate_position_size(self, symbol, atr, trading_state):
        """Calculate optimal position size based on risk parameters"""
        state = trading_state.symbol_states[symbol]
        risk_per_trade = min(
            0.02 * trading_state.account_balance, 100
        )  # 2% risk or max $100

        if state.win_rate < MIN_WIN_RATE:
            risk_per_trade *= 0.5

        return risk_per_trade / (atr * 5)  # Using 5x ATR for stop loss

    def calculate_risk_reward(self, entry_price, stop_loss, take_profit):
        """Calculate risk-reward ratio for a trade"""
        risk = abs(entry_price - stop_loss)
        reward = abs(entry_price - take_profit)
        return reward / risk if risk != 0 else 0
