# models/trading_statistics.py

import numpy as np
import logging 
from datetime import datetime
import os

from logging_config import setup_comprehensive_logging
setup_comprehensive_logging()

def get_next_session_number(filename="session_tracker.txt"):
    try:
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(
                    "1,{},{}".format(datetime.now().strftime("%Y-%m-%d"), "Started")
                )
            return 1

        with open(filename, "r") as f:
            lines = f.readlines()

            # Check if file is empty
            if not lines:
                with open(filename, "w") as f:
                    f.write(
                        "1,{},{}".format(datetime.now().strftime("%Y-%m-%d"), "Started")
                    )
                return 1

            last_line = lines[-1].strip().split(",")
            current_session = int(last_line[0])

        next_session = current_session + 1

        with open(filename, "a") as f:
            f.write(
                "\n{},{},{}".format(
                    next_session, datetime.now().strftime("%Y-%m-%d"), "Started"
                )
            )

        return next_session

    except Exception as e:
        logging.error(f"Error managing session counter: {e}")
        return 1


class TradingStatistics:
    def __init__(self, symbols, currency="$"):
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

trading_stats = None
