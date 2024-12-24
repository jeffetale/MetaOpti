# trading/signal_generator.py

import logging
import pandas as pd
from datetime import datetime
from config import TIMEFRAME, mt5, TRADING_CONFIG, MT5Config
from models.trading_state import trading_state
from ml.features.technical_indicators import TechnicalIndicators

from logging_config import setup_comprehensive_logging
setup_comprehensive_logging()

class SignalGenerator:
    def __init__(self, ml_predictor):
        self.logger = logging.getLogger(__name__)
        self.ml_predictor = ml_predictor
        self.logger.info(
            f"🎯 Initialized SignalGenerator with predictor: {ml_predictor.__class__.__name__}"
        )

    def get_signal(self, symbol):
        """Generate trading signals with multiple confirmation methods"""
        self.logger.debug(f"🔄 Starting signal generation for {symbol}")

        if not self._should_trade_symbol(symbol):
            self.logger.info(
                f"⛔ {symbol} trading restricted - skipping signal generation"
            )
            return None, None, 0

        df = self._get_market_data(symbol)
        if df is None:
            self.logger.warning(f"❌ Failed to fetch market data for {symbol}")
            return None, None, 0

        self.logger.debug(f"📊 Retrieved {len(df)} data points for {symbol}")

        # Calculate indicators and get current market state
        current_state = self._calculate_indicators(df)
        if current_state is None:
            self.logger.warning(f"⚠️ Failed to calculate indicators for {symbol}")
            return None, None, 0

        try:
            # Get ML prediction using the ml_predictor instance
            self.logger.debug(f"🤖 Getting ML prediction for {symbol}")
            ml_signal, ml_confidence, ml_predicted_return = self.ml_predictor.predict()

            self.logger.info(
                f"""💡 {symbol} ML Prediction:
                Signal={ml_signal}
                Confidence={ml_confidence:.4f}
                Predicted Return={ml_predicted_return:.4f}"""
            )

            # Check for neutral state
            if self._check_neutral_state(symbol, ml_confidence, ml_predicted_return):
                self.logger.info(
                    f"""⚖️ {symbol} entered NEUTRAL state:
                    Confidence: {ml_confidence:.4f}
                    Predicted Return: {ml_predicted_return:.4f}"""
                )
                return "neutral", current_state["ATR"], 0

            # Check if symbol is in neutral hold
            if not self._check_neutral_hold(symbol):
                self.logger.debug(f"⏸️ {symbol} in neutral hold period")
                return None, None, 0

            # Trade Direction Repetition Prevention
            if not self._check_trade_direction_valid(symbol, ml_signal):
                self.logger.info(
                    f"🚫 {symbol} - {ml_signal} signal suppressed due to recent trade direction"
                )
                return None, None, 0

            # Generate technical signals
            tech_signal = self._generate_technical_signal(current_state)
            self.logger.debug(f"📈 {symbol} Technical Signal: {tech_signal}")

            # Combine signals for final decision
            final_signal, potential_profit = self._combine_signals(
                symbol,
                current_state,
                ml_signal,
                ml_confidence,
                ml_predicted_return,
                tech_signal,
            )

            # Final decision logging
            self.logger.info(
                f"""✨ {symbol} Final Decision:
                Signal={final_signal}
                Potential Profit={potential_profit:.4f}"""
            )

            return final_signal, current_state["ATR"], potential_profit

        except Exception as e:
            self.logger.error(
                f"💥 Signal generation error for {symbol}: {str(e)}", exc_info=True
            )
            return None, None, 0

    def _generate_technical_signal(self, current_state):
        """Generate trading signal based on technical indicators"""
        # RSI signals
        rsi = current_state["RSI"]
        rsi_signal = "buy" if rsi < 30 else "sell" if rsi > 70 else None

        # MACD signals
        macd = current_state["MACD"]
        macd_signal = "buy" if macd > 0 else "sell" if macd < 0 else None

        # Stochastic signals
        stoch = current_state["Stochastic"]
        stoch_signal = "buy" if stoch < 20 else "sell" if stoch > 80 else None

        # Williams %R signals
        williams = current_state["Williams_R"]
        williams_signal = (
            "buy" if williams < -80 else "sell" if williams > -20 else None
        )

        # Count signals in each direction
        buy_signals = sum(
            1
            for signal in [rsi_signal, macd_signal, stoch_signal, williams_signal]
            if signal == "buy"
        )
        sell_signals = sum(
            1
            for signal in [rsi_signal, macd_signal, stoch_signal, williams_signal]
            if signal == "sell"
        )

        # Determine overall technical signal
        if buy_signals > sell_signals and buy_signals >= 2:
            return "buy"
        elif sell_signals > buy_signals and sell_signals >= 2:
            return "sell"
        return None

    def _combine_signals(
        self,
        symbol,
        current_state,
        ml_signal,
        ml_confidence,
        ml_predicted_return,
        tech_signal,
    ):
        """Combine ML and technical signals for final trading decision with balanced buy/sell handling"""
        # Basic signal strength scores
        ml_strength = 0
        tech_strength = 0

        # Calculate ML signal strength with balanced thresholds
        if ml_signal:
            # For sell signals, transform confidence to be relative to 0.5
            # e.g., a sell confidence of 0.2 becomes 0.8 (1 - 0.2)
            adjusted_confidence = ml_confidence if ml_signal == "buy" else (1 - ml_confidence)
            
            if adjusted_confidence >= 0.6:
                ml_strength = adjusted_confidence * 2  # Scale up to 0-2 range
                if abs(ml_predicted_return) > 0.001:  # absolute value for return
                    ml_strength *= 1.2

        # Calculate technical signal strength
        if tech_signal:
            # RSI extremes increase technical strength
            rsi = current_state["RSI"]
            if (tech_signal == "buy" and rsi < 25) or (tech_signal == "sell" and rsi > 75):
                tech_strength += 1

            # MACD divergence increases technical strength
            macd = current_state["MACD"]
            if (tech_signal == "buy" and macd > 0) or (tech_signal == "sell" and macd < 0):
                tech_strength += 0.5

            # Stochastic extremes increase technical strength
            stoch = current_state["Stochastic"]
            if (tech_signal == "buy" and stoch < 15) or (tech_signal == "sell" and stoch > 85):
                tech_strength += 0.5

        # Conservative mode adjustments
        required_strength = 2.5 if trading_state.is_conservative_mode else 2.0

        # Calculate total signal strength
        total_strength = ml_strength + tech_strength

        # Calculate potential profit based on signal strength and ATR
        potential_profit = current_state["ATR"] * total_strength * 10
        if trading_state.is_conservative_mode:
            potential_profit *= 0.8

        # Final signal determination with balanced thresholds
        if total_strength >= required_strength:
            # Signals must agree or ML confidence must be very high
            if ml_signal == tech_signal or (
                adjusted_confidence > 0.6 and abs(ml_predicted_return) > 0.002
            ):
                return ml_signal, potential_profit
            elif adjusted_confidence > 0.7:  # Very high ML confidence can override tech signal
                return ml_signal, potential_profit

        return "neutral", 0

    def _check_neutral_hold(self, symbol):
        """Check if symbol should remain in neutral hold"""
        state = trading_state.symbol_states[symbol]
        if state.neutral_start_time:
            neutral_duration = (
                datetime.now() - state.neutral_start_time
            ).total_seconds()
            if neutral_duration < TRADING_CONFIG.NEUTRAL_HOLD_DURATION:
                self.logger.info(f"{symbol} still in neutral hold")
                return False
            state.neutral_start_time = None
        return True

    def _check_trade_direction_valid(self, symbol, ml_signal):
        """Check if trade direction is valid based on recent trades"""
        state = trading_state.symbol_states[symbol]
        if state.recent_trade_directions:
            recent_direction_count = state.recent_trade_directions.count(ml_signal)
            if recent_direction_count >= 2:
                self.logger.info(
                    f"{symbol} suppressing {ml_signal} due to recent similar trades"
                )
                return False
        return True

    def _log_analysis(
        self,
        symbol,
        ml_signal,
        ml_confidence,
        ml_predicted_return,
        tech_signal,
        final_signal,
    ):
        """Log detailed analysis of signal generation"""
        self.logger.info(
            f"""
            ******** {symbol} Analysis ********
            ML Signal: {ml_signal}
            ML Confidence: {ml_confidence:.2f}
            ML Predicted Return: {ml_predicted_return:.5f}
            Technical Signal: {tech_signal}
            Final Signal: {final_signal}
            """
        )

    def _should_trade_symbol(self, symbol):
        """Determine if trading should occur for a symbol based on performance"""
        state = trading_state.symbol_states[symbol]

        # Check recent performance
        recent_trades = state.trades_history[-3:]
        if recent_trades:
            if sum(recent_trades) < 0:
                cooling_period = (
                    datetime.now() - state.last_trade_time
                ).total_seconds()
                if cooling_period < 120:
                    return False

        return not state.is_restricted

    def _get_market_data(self, symbol):
        """Fetch and prepare market data"""
        rates = mt5.copy_rates_from_pos(symbol, MT5Config.TIMEFRAME, 0, 100)
        if rates is None:
            self.logger.warning(f"No rates available for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def _calculate_indicators(self, df):
        """Calculate technical indicators including ATR"""
        try:
            # Calculate ATR and other indicators using TechnicalIndicators class
            current_state = {
                'ATR': TechnicalIndicators.atr(df).iloc[-1],
                'RSI': TechnicalIndicators.rsi(df['close']).iloc[-1],
                'MACD': TechnicalIndicators.macd(df['close']).iloc[-1],
                'Stochastic': TechnicalIndicators.stochastic(df).iloc[-1],
                'Williams_R': TechnicalIndicators.williams_r(df).iloc[-1],
                # Add other indicators as needed
            }            
            return current_state
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def _check_neutral_state(self, symbol, ml_confidence, ml_predicted_return):
        """Check if symbol should enter neutral state with balanced thresholds"""
        # For sell signals, check confidence relative to 0.5
        adjusted_confidence = ml_confidence
        if ml_confidence < 0.5:  # sell signal
            adjusted_confidence = 1 - ml_confidence
            
        if (
            adjusted_confidence <= TRADING_CONFIG.NEUTRAL_CONFIDENCE_THRESHOLD
            or abs(ml_predicted_return) < TRADING_CONFIG.MIN_PREDICTED_RETURN # absolute value for return
        ):
            trading_state.symbol_states[symbol].neutral_start_time = datetime.now()
            return True
        return False
