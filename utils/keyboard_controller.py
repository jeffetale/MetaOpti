# utils/keyboard_controller.py

from pynput import keyboard
import threading
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from config import mt5


@dataclass
class KeyboardState:
    shift_pressed: bool = False
    alt_pressed: bool = False
    last_action_time: Optional[datetime] = None
    sound_enabled: bool = True
    new_trades_enabled: bool = True


class KeyboardController:
    def __init__(self, trading_bot):
        self.logger = logging.getLogger(__name__)
        self.trading_bot = trading_bot
        self.state = KeyboardState()
        self.keyboard_thread = None
        self.running = False
        self.listener = None

    def start(self):
        """Start keyboard monitoring in a separate thread"""
        self.running = True
        self.listener = keyboard.Listener(
            on_press=self._handle_key_press,
            on_release=self._handle_key_release
        )
        self.listener.start()
        self.logger.info("Keyboard controller started")
        self._print_controls_help()

    def stop(self):
        """Stop keyboard monitoring"""
        self.running = False
        if self.listener:
            self.listener.stop()
        self.logger.info("Keyboard controller stopped")

    def _print_controls_help(self):
        """Print available keyboard controls"""
        help_text = """
        🎮 Trading Bot Keyboard Controls 🎮
        ================================
        Shift + Alt + T: Close all profitable positions
        Shift + Alt + L: Close all losing positions
        Shift + Alt + M: Toggle sound alerts
        Shift + Alt + N: Toggle new trade opening
        Shift + Alt + Q: Safe shutdown
        
        Note: All controls require Shift + Alt combination
        to prevent accidental activation
        ================================
        """
        print(help_text)

    def _handle_key_press(self, key):
        """Handle key press events"""
        try:
            if key == keyboard.Key.shift:
                self.state.shift_pressed = True
            elif key == keyboard.Key.alt:
                self.state.alt_pressed = True
            elif self.state.shift_pressed and self.state.alt_pressed and hasattr(key, 'char'):
                self._handle_command(key.char.lower())
        except AttributeError:
            pass

    def _handle_key_release(self, key):
        """Handle key release events"""
        if key == keyboard.Key.shift:
            self.state.shift_pressed = False
        elif key == keyboard.Key.alt:
            self.state.alt_pressed = False

    def _handle_command(self, key):
        """Process keyboard commands"""
        # Prevent rapid-fire actions
        now = datetime.now()
        if (
            self.state.last_action_time
            and (now - self.state.last_action_time).total_seconds() < 1
        ):
            return

        self.state.last_action_time = now

        try:
            if key == 't':
                self._close_profitable_positions()
            elif key == 'l':
                self._close_losing_positions()
            elif key == 'm':
                self._toggle_sound_alerts()
            elif key == 'n':
                self._toggle_new_trades()
            elif key == 'q':
                self._initiate_shutdown()
        except Exception as e:
            self.logger.error(f"Error handling keyboard command: {e}")

    def _close_profitable_positions(self):
        """Close all profitable positions"""
        try:
            positions = mt5.positions_get()
            if positions:
                closed_count = 0
                total_profit = 0
                for position in positions:
                    if position.profit > 0:
                        if self.trading_bot.order_manager.close_position(position):
                            closed_count += 1
                            total_profit += position.profit

                self.logger.info(
                    f"Closed {closed_count} profitable positions. Total profit: ${total_profit:.2f}"
                )
        except Exception as e:
            self.logger.error(f"Error closing profitable positions: {e}")

    def _close_losing_positions(self):
        """Close all losing positions"""
        try:
            positions = mt5.positions_get()
            if positions:
                closed_count = 0
                total_loss = 0
                for position in positions:
                    if position.profit < 0:
                        if self.trading_bot.order_manager.close_position(position):
                            closed_count += 1
                            total_loss += position.profit

                self.logger.info(
                    f"Closed {closed_count} losing positions. Total loss: ${total_loss:.2f}"
                )
        except Exception as e:
            self.logger.error(f"Error closing losing positions: {e}")

    def _toggle_sound_alerts(self):
        """Toggle sound alerts"""
        self.state.sound_enabled = not self.state.sound_enabled
        if hasattr(self.trading_bot, "alerts") and self.trading_bot.alerts:
            self.trading_bot.alerts.enabled = self.state.sound_enabled
        status = "enabled" if self.state.sound_enabled else "disabled"
        self.logger.info(f"Sound alerts {status}")

    def _toggle_new_trades(self):
        """Toggle new trade opening"""
        self.state.new_trades_enabled = not self.state.new_trades_enabled
        status = "enabled" if self.state.new_trades_enabled else "disabled"
        self.logger.info(f"New trade opening {status}")

    def _initiate_shutdown(self):
        """Initiate safe shutdown of the trading bot"""
        self.logger.info("Safe shutdown initiated via keyboard command")
        self.trading_bot.shutdown()
