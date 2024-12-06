# models.py

from collections import defaultdict
from config import INITIAL_VOLUME, MIN_PROFIT_THRESHOLD

class SymbolState:
    def __init__(self):
        self.trades_history = []
        self.consecutive_losses = 0
        self.total_profit = 0
        self.max_profit = 0
        self.is_restricted = False
        self.last_trade_time = None
        self.win_rate = 0
        self.volume = INITIAL_VOLUME
        self.trades_count = 0
        self.profit_threshold = MIN_PROFIT_THRESHOLD
        self.recent_trade_directions = []  # Track last few trade directions
        self.trade_direction_memory_size = 3  # Remember last 3 trades

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
    def __init__(self):
        self.symbol_states = defaultdict(SymbolState)
        self.global_profit = 0
        self.is_conservative_mode = False
        self.ta_params = TAParams()
        self.manual_intervention_detected = False
        self.manual_intervention_cooldown = 0
        self.last_manual_intervention_time = None

trading_state = TradingState()