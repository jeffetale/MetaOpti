# hedging/market_analysis.py

import numpy as np
from typing import Optional, Dict
import logging
from config import mt5
from hedging.models import MarketContext
from logging_config import EmojiLogger


class MarketAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.symbol_stats: Dict[str, Dict] = {}

    def _calculate_adx(self, symbol: str, period: int = 14) -> float:
        """Calculate Average Directional Index for trend strength with proper error handling"""
        try:
            # Validate inputs
            if not symbol or period <= 0:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR,
                        f"Invalid inputs for ADX calculation: symbol={symbol}, period={period}",
                    )
                )
                return 0.0

            # Verify symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"Symbol {symbol} not found in MT5"
                    )
                )
                return 0.0

            # Get enough data for calculation
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period * 2 + 1)
            if rates is None or len(rates) < period * 2:
                self.logger.warning(
                    EmojiLogger.format_message(
                        EmojiLogger.WARNING,
                        f"Insufficient data for ADX calculation for {symbol}",
                    )
                )
                return 0.0

            # Convert to numpy arrays
            try:
                high = np.array([rate["high"] for rate in rates], dtype=np.float64)
                low = np.array([rate["low"] for rate in rates], dtype=np.float64)
                close = np.array([rate["close"] for rate in rates], dtype=np.float64)
            except (KeyError, ValueError) as e:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"Error converting price data: {str(e)}"
                    )
                )
                return 0.0

            # Validate price data
            if (
                np.any(np.isnan(high))
                or np.any(np.isnan(low))
                or np.any(np.isnan(close))
            ):
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"NaN values found in price data"
                    )
                )
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
                    self.logger.error(
                        EmojiLogger.format_message(
                            EmojiLogger.ERROR,
                            f"Index error in TR calculation: {str(e)}",
                        )
                    )
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
                    self.logger.warning(
                        EmojiLogger.format_message(
                            EmojiLogger.WARNING, f"TR period is zero"
                        )
                    )
                    return 0.0

                plus_di = plus_period / tr_period * 100 if tr_period > 0 else 0
                minus_di = minus_period / tr_period * 100 if tr_period > 0 else 0

                # Validate DI values
                if not (0 <= plus_di <= 100) or not (0 <= minus_di <= 100):
                    self.logger.error(
                        EmojiLogger.format_message(
                            EmojiLogger.ERROR,
                            f"Invalid DI values: +DI={plus_di}, -DI={minus_di}",
                        )
                    )
                    return 0.0

                # Calculate ADX
                dx = (
                    abs(plus_di - minus_di) / (plus_di + minus_di) * 100
                    if (plus_di + minus_di) > 0
                    else 0
                )

                return float(dx)

            except Exception as e:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"Error in DI/ADX calculation: {str(e)}"
                    )
                )
                return 0.0

        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error calculating ADX for {symbol}: {str(e)}"
                )
            )
            return 0.0

    def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate ATR with comprehensive error handling"""
        try:
            # Validate inputs
            if not symbol or period <= 0:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR,
                        f"Invalid inputs for ATR calculation: symbol={symbol}, period={period}",
                    )
                )
                return None

            # Verify symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"Symbol {symbol} not found in MT5"
                    )
                )
                return None

            # Get price data with timeout handling
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
            if rates is None or len(rates) < period + 1:
                self.logger.warning(
                    EmojiLogger.format_message(
                        EmojiLogger.WARNING,
                        f"Insufficient data for ATR calculation for {symbol}",
                    )
                )
                return None

            # Convert to numpy arrays with validation
            try:
                high = np.array([rate["high"] for rate in rates], dtype=np.float64)
                low = np.array([rate["low"] for rate in rates], dtype=np.float64)
                close = np.array([rate["close"] for rate in rates], dtype=np.float64)
            except (KeyError, ValueError) as e:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR,
                        f"Error converting price data for ATR: {str(e)}",
                    )
                )
                return None

            # Validate price data
            if (
                np.any(np.isnan(high))
                or np.any(np.isnan(low))
                or np.any(np.isnan(close))
            ):
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"NaN values found in price data for ATR"
                    )
                )
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
                    self.logger.error(
                        EmojiLogger.format_message(
                            EmojiLogger.ERROR,
                            f"Index error in ATR calculation: {str(e)}",
                        )
                    )
                    return None

            # Calculate ATR with validation
            try:
                atr = tr[1:].mean()  # Exclude the first zero value

                # Validate ATR value
                if np.isnan(atr) or atr <= 0:
                    self.logger.error(
                        EmojiLogger.format_message(
                            EmojiLogger.ERROR, f"Invalid ATR value calculated: {atr}"
                        )
                    )
                    return None

                return float(atr)

            except Exception as e:
                self.logger.error(
                    EmojiLogger.format_message(
                        EmojiLogger.ERROR, f"Error in final ATR calculation: {str(e)}"
                    )
                )
                return None

        except Exception as e:
            self.logger.error(
                EmojiLogger.format_message(
                    EmojiLogger.ERROR, f"Error calculating ATR for {symbol}: {str(e)}"
                )
            )
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
