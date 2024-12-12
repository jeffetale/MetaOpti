# models.py

import numpy as np
from collections import defaultdict
from config import INITIAL_VOLUME, MIN_PROFIT_THRESHOLD, get_next_session_number
import logging
from datetime import datetime

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
    def __init__(self):
        self.symbol_states = defaultdict(SymbolState)
        self.global_profit = 0
        self.is_conservative_mode = False
        self.ta_params = TAParams()
        self.manual_intervention_detected = False
        self.manual_intervention_cooldown = 0
        self.last_manual_intervention_time = None


class TradingStatistics:
    def __init__(self, symbols, currency='$'):
        self.symbols = symbols
        self.currency = currency
        self.reset_statistics()

    def reset_statistics(self):
        """Reset all trading statistics to initial state"""
        self.total_trades = {symbol: 0 for symbol in self.symbols}
        self.buy_trades = {symbol: 0 for symbol in self.symbols}
        self.sell_trades = {symbol: 0 for symbol in self.symbols}

        self.symbol_profits = {symbol: [] for symbol in self.symbols}
        self.ml_signals_count = 0
        self.calculated_signals_count = 0

        self.position_reversals = {symbol: 0 for symbol in self.symbols}

        # Comprehensive tracking arrays
        self.profit_array = []
        self.symbol_profit_array = {symbol: [] for symbol in self.symbols}
        self.max_cumulative_profit = 0
        self.min_cumulative_profit = 0
        self.cumulative_profit_history = []
        self.symbol_cumulative_profits = {symbol: 0 for symbol in self.symbols}

    def log_trade(self, symbol, direction, profit, is_ml_signal):
        """Log individual trade details"""
        self.total_trades[symbol] += 1
        if direction == "buy":
            self.buy_trades[symbol] += 1
        else:
            self.sell_trades[symbol] += 1

        self.symbol_profits[symbol].append(profit)
        self.profit_array.append(profit)
        self.symbol_profit_array[symbol].append(profit)

        # Signal type tracking
        if is_ml_signal:
            self.ml_signals_count += 1
        else:
            self.calculated_signals_count += 1

        # Update cumulative profit tracking
        current_cumulative_profit = sum(self.profit_array + [profit])
        self.cumulative_profit_history.append(current_cumulative_profit)

        # Update max and min cumulative profits
        self.max_cumulative_profit = max(
            self.max_cumulative_profit, current_cumulative_profit
        )
        self.min_cumulative_profit = min(
            self.min_cumulative_profit, current_cumulative_profit
        )

        self.symbol_cumulative_profits[symbol] += profit

    def log_position_reversal(self, symbol):
        """Log position reversals for a symbol"""
        self.position_reversals[symbol] += 1

    def get_statistics(self):
        """Generate comprehensive trading statistics"""
        stats = {
            "peak_single_symbol_profit": (
                max(self.profit_array) if self.profit_array else 0
            ),
            "lowest_single_symbol_profit": (
                min(self.profit_array) if self.profit_array else 0
            ),
            "cumulative_total_profit": (
                sum(self.profit_array) if self.profit_array else 0
            ),
            "total_trades_by_symbol": self.total_trades,
            "buy_trades_by_symbol": self.buy_trades,
            "sell_trades_by_symbol": self.sell_trades,
            "highest_symbol_profit": {
                symbol: max([p for p in profits if p > 0], default=0)
                for symbol, profits in self.symbol_profit_array.items()
            },
            "lowest_symbol_profit": {
                symbol: min([p for p in profits if p < 0], default=0)
                for symbol, profits in self.symbol_profit_array.items()
            },
            "max_cumulative_profit": f"{self.currency}{self.max_cumulative_profit:.2f}",
            "min_cumulative_profit": f"{self.currency}{self.min_cumulative_profit:.2f}",
            "most_traded_symbol": max(self.total_trades, key=self.total_trades.get),
            "total_ml_signals": self.ml_signals_count,
            "total_calculated_signals": self.calculated_signals_count,
            "position_reversals_by_symbol": self.position_reversals,
            "average_profit": np.mean(self.profit_array) if self.profit_array else 0,
            "total_trades_count": sum(self.total_trades.values()),
        }

        stats.update(
            {
                "symbol_cumulative_profits": {
                    symbol: f"{self.currency}{profit:.2f}"
                    for symbol, profit in self.symbol_cumulative_profits.items()
                }
            }
        )

        symbol_profits = {
            symbol: [f"{self.currency}{p:.2f}" for p in profits]
            for symbol, profits in self.symbol_profit_array.items()
        }        

        stats.update(
            {
                "symbol_profits": symbol_profits,
                "profit_array": [f"{self.currency}{p:.2f}" for p in self.profit_array],
            }
        )

        return stats

    def log_final_statistics(self, initial_balance, final_balance):
        """Log final trading session statistics"""
        final_stats = self.get_statistics()
        final_stats["initial_balance"] = f"{self.currency}{initial_balance:.2f}"
        final_stats["final_balance"] = f"{self.currency}{final_balance:.2f}"
        final_stats["total_account_change"] = (
            f"{self.currency}{final_balance - initial_balance:.2f}"
        )

        session_number = get_next_session_number()

        # Log to file using the existing logging configuration
        logging.info(f"\n===== TRADING SESSION {session_number} =====")
        for key, value in final_stats.items():
            logging.info(f"{key}: {value}")

        # formatted output for easier reading
        stats_str = (
            f"{'='*20} TRADING SESSION {session_number} {'='*20}\n"
            f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"💰 Initial Balance: {self.currency}{initial_balance:.2f}\n"
            f"💸 Final Balance:   {self.currency}{final_balance:.2f}\n"
            f"📈 Total Account Change: {self.currency}{final_balance - initial_balance:.2f}\n"
            f"{'-'*50}\n"
            f"🔍 Detailed Statistics:\n"
        )

        # Add more visual representation
        for key, value in final_stats.items():
            stats_str += f"• {key.replace('_', ' ').title()}: {value}\n"

        stats_str += f"{'='*50}\n\n"

        try:
            with open("trading_session_stats.txt", "a", encoding="utf-8") as f:
                f.write(stats_str)
        except Exception as e:
            logging.error(f"Failed to append statistics to file: {e}")

        return final_stats

trading_state = TradingState()
trading_stats = None
