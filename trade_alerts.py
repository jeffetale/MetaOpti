# trade_alerts.py

import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from playsound import playsound
from config import mt5, SOUNDS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TradeAlerts:
    def __init__(self, sound_dir: Optional[str] = None):
        self.shutdown_flag = threading.Event()
        self.position_states: Dict[int, Dict] = {}
        

        # Sounds directory
        self.sound_dir = sound_dir if sound_dir else SOUNDS_DIR

        if not self.sound_dir.exists():
            self.sound_dir.mkdir(parents=True, exist_ok=True)

        # Define threshold configurations
        self.thresholds = {
            "negative_to_positive": {
                "condition": lambda old, new: old.profit < -10 and new.profit >= 0,
                "sound": "recovery.mp3",
                "message": "Position recovered from negative to positive!",
            },
            "profit_milestone": {
                "condition": lambda old, new: old.profit < 20 and new.profit >= 20,
                "sound": "profit.mp3",
                "message": "Position reached +$20 profit!",
            },
            "high_profit": {
                "condition": lambda old, new: old.profit < 50 and new.profit >= 50,
                "sound": "high_profit.mp3",
                "message": "Position reached +$50 profit milestone!",
            },
            "loss_warning": {
                "condition": lambda old, new: old.profit > -30 and new.profit <= -30,
                "sound": "warning.mp3",
                "message": "Warning: Position loss exceeds -$30!",
            },
            "critical_loss": {
                "condition": lambda old, new: old.profit > -50 and new.profit <= -50,
                "sound": "danger.mp3",
                "message": "ALERT: Critical loss level reached (-$50)!",
            },
            "breakeven": {
                "condition": lambda old, new: abs(old.profit) >= 5
                and abs(new.profit) < 1,
                "sound": "breakeven.mp3",
                "message": "Position returned to breakeven level",
            },
            'positive_to_negative': {  
                'condition': lambda old, new: old.profit >= 50 and new.profit < 0,
                'sound': 'positive_to_negative.mp3',
                'message': 'Position dropped from +$50 profit to negative!'
            },
        }

    def _play_sound(self, sound_file: str):
        """Play sound file with error handling"""
        try:
            sound_path = self.sound_dir / sound_file
            if sound_path.exists():
                playsound(str(sound_path))
            else:
                logger.warning(f"Sound file not found: {sound_path}")
        except Exception as e:
            logger.error(f"Error playing sound {sound_file}: {e}")

    def _check_thresholds(self, old_position, new_position):
        """Check if any thresholds have been crossed"""
        for threshold_name, config in self.thresholds.items():
            try:
                if config["condition"](old_position, new_position):
                    logger.info(f"{config['message']} Ticket: {new_position.ticket}")
                    self._play_sound(config["sound"])
            except Exception as e:
                logger.error(f"Error checking {threshold_name} threshold: {e}")

    def _monitor_positions(self):
        """Monitor positions for threshold crossings"""
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return

        while not self.shutdown_flag.is_set():
            try:
                # Get all current positions
                positions = mt5.positions_get()
                if positions is None:
                    logger.warning("No positions retrieved")
                    time.sleep(1)
                    continue

                current_tickets = set()

                # Check each position
                for position in positions:
                    ticket = position.ticket
                    current_tickets.add(ticket)

                    # Get previous state
                    prev_state = self.position_states.get(ticket)

                    if prev_state:
                        # Check for threshold crossings
                        self._check_thresholds(prev_state, position)

                    # Update state
                    self.position_states[ticket] = position

                # Clean up closed positions
                closed_tickets = set(self.position_states.keys()) - current_tickets
                for ticket in closed_tickets:
                    prev_position = self.position_states[ticket]
                    if prev_position.profit > 0:
                        self._play_sound("profit_close.mp3")
                        logger.info(
                            f"Position {ticket} closed with profit: ${prev_position.profit:.2f}"
                        )
                    else:
                        self._play_sound("loss_close.mp3")
                        logger.info(
                            f"Position {ticket} closed with loss: ${prev_position.profit:.2f}"
                        )
                    del self.position_states[ticket]

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                time.sleep(1)

    def start(self):
        """Start the position monitoring thread"""
        logger.info("Starting trade alerts monitor...")
        self.monitor_thread = threading.Thread(target=self._monitor_positions)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """Stop the position monitoring thread"""
        logger.info("Stopping trade alerts monitor...")
        self.shutdown_flag.set()
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=5)
        mt5.shutdown()


def main():
    """Standalone execution"""
    try:
        alerts = TradeAlerts()
        alerts.start()

        logger.info("Trade alerts monitor running. Press Ctrl+C to stop.")

        # Keep the main thread running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down trade alerts monitor...")
        alerts.stop()


if __name__ == "__main__":
    main()
