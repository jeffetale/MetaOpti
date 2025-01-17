# logging_config.py

import logging
import colorlog


class EmojiLogger:
    # Trading Status Emojis
    POSITION_OPEN = "🟢"
    POSITION_CLOSE = "🔴"
    PROFIT = "💰"
    LOSS = "📉"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"
    HEDGE = "🔄"
    TRAILING = "🎯"
    VOLATILITY = "📊"
    INCREASE = "⬆️"
    DECREASE = "⬇️"
    MARKET = "📈"
    SUCCESS = "✅"
    ALERT = "🚨"
    TIME = "⏰"
    AI = "🤖"

    @staticmethod
    def format_message(emoji: str, message: str) -> str:
        return f"{emoji} {message}"


def setup_comprehensive_logging():
    """
    Comprehensive logging configuration with:
    - Colored console output
    - File logging
    - Emoji-enhanced formatting
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to prevent duplicate logging
    logger.handlers.clear()

    # Console Handler with Color
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "🕒 %(log_color)s%(levelname)8s%(reset)s | %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    # File Handler with UTF-8 encoding
    file_handler = logging.FileHandler("trading_bot.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "🕒 %(asctime)s.%(msecs)03d | %(levelname)8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def log_session_start(session_number):
    """Log the start of a trading session with a visually appealing marker"""
    session_start_marker = f"""
{'='*50}
🚀 TRADING SESSION {session_number} STARTED 🚀
{'='*50}
"""
    logging.info(session_start_marker)


def log_session_end(session_number):
    """Log the end of a trading session with a summary marker"""
    session_end_marker = f"""
{'='*50}
🏁 TRADING SESSION {session_number} ENDED 🏁
{'='*50}
"""
    logging.info(session_end_marker)


def log_critical_event(event_type, message):
    """Log critical events with emphasis"""
    critical_marker = f"""
{'!'*50}
🚨 {event_type.upper()} 🚨
{message}
{'!'*50}
"""
    logging.critical(critical_marker)
