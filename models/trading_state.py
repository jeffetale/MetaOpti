# models/trading_state.py

from collections import defaultdict
from config import TRADING_CONFIG

class SymbolState:
    def __init__(self):
        self.trades_history = []
        self.consecutive_losses = 0
        self.total_profit = 0
        self.max_profit = 0
        self.is_restricted = False
        self.last_trade_time = None
        self.win_rate = 0
        self.volume = TRADING_CONFIG.INITIAL_VOLUME
        self.trades_count = 0
        self.profit_threshold = TRADING_CONFIG.MIN_PROFIT_THRESHOLD
        self.recent_trade_directions = []  # Track last few trade directions
        self.trade_direction_memory_size = 5  # Remember last 5 trades
        self.neutral_start_time = None


class TAParams:
    # Dynamic parameters that adjust based on market conditions
    def __init__(self):
        self.sma_short = 5
        self.sma_long = 10
        self.rsi_period = 7
        self.rsi_overbought = 80
        self.rsi_oversold = 20
        self.atr_period = 7
        self.atr_multiplier = 1.5
        self.volatility_threshold = 0.0001


class TradingState:
    # Class-level constants for risk management
    MAX_CONSECUTIVE_LOSSES = TRADING_CONFIG.MAX_CONSECUTIVE_LOSSES
    MIN_WIN_RATE = TRADING_CONFIG.MIN_WIN_RATE
    NEUTRAL_HOLD_DURATION = TRADING_CONFIG.NEUTRAL_HOLD_DURATION

    def __init__(self):
        self.symbol_states = defaultdict(SymbolState)
        self.global_profit = 0
        self.is_conservative_mode = False
        self.ta_params = TAParams()
        self.manual_intervention_detected = False
        self.manual_intervention_cooldown = 0
        self.last_manual_intervention_time = None
        self.account_balance = 0 

    def update_account_balance(self, new_balance):
        """Update the current account balance"""
        self.account_balance = new_balance

    def should_enter_conservative_mode(self):
        """Determine if trading should switch to conservative mode"""
        return (
            self.global_profit < -50  # Enter conservative mode at $50 drawdown
            or any(
                state.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES
                for state in self.symbol_states.values()
            )
        )

    def reset_symbol_state(self, symbol):
        """Reset a symbol's trading state"""
        self.symbol_states[symbol] = SymbolState()


trading_state = TradingState()
