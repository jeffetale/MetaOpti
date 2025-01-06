# config.py

# from mt5linux import MetaTrader5
import MetaTrader5 as mt5
import logging, threading
from pathlib import Path
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import platform

from logging_config import setup_comprehensive_logging

setup_comprehensive_logging()

load_dotenv()

# Absolute path of the project root directory
ROOT_DIR = Path(__file__).resolve().parent

# All project paths relative to ROOT_DIR
SOUNDS_DIR = ROOT_DIR / "sounds"
MODEL_SAVE_DIR= ROOT_DIR / "ml" / "ml_models"
BACKTEST_MODEL_SAVE_DIR = ROOT_DIR / "backtest" / "ml_models"

# Create directories if they don't exist
SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# MT5 paths based on operating system
MT5_PATHS = {
    "Windows": [
        Path(r"C:\Program Files\MetaTrader 5\terminal64.exe"),
        Path(r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"),
    ],
    "OSX/Linux": [
        Path.home() / ".wine/drive_c/Program Files/MetaTrader 5/terminal64.exe",
        Path.home() / ".mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe",
        Path("/opt/metaquotes/terminal64.exe"),
    ],
}


# MT5 Configuration
class MT5Config:
    PASSWORD = os.getenv("PASSWORD")
    SERVER = os.getenv("SERVER")
    ACCOUNT_NUMBER = 52071754
    TIMEFRAME = mt5.TIMEFRAME_M15
    # TIMEFRAME = MetaTrader5.TIMEFRAME_M10


# Trading Parameters Configuration
@dataclass
class TradingConfig:
    # Volume and Position Sizing
    INITIAL_VOLUME: float = 0.1
    MIN_TARGET_VOLUME: float = 0.01  # Minimum target volume for opening position if volume and equity calculations are higher
    MAX_VOLUME_MULTIPLIER: float = 1.5  # Maximum volume increase from initial
    MIN_VOLUME_MULTIPLIER: float = 0.25  # Minimum volume decrease from initial
    VOLUME_STEP_UP: float = 1.15 # Volume increase factor on success
    VOLUME_STEP_DOWN: float = 0.5  # Volume decrease factor on failure
    MARGIN_USAGE_LIMIT: float = 0.3  # Maximum margin usage (30% of available)  # less increases the risk   
    EQUITY_RISK_PER_TRADE: float = 0.8  # Maximum equity risk per trade (80%)

    # Stop Loss and Take Profit   # doubled each
    SL_ATR_MULTIPLIER: float = 3  # Stop Loss distance in ATR
    TP_ATR_MULTIPLIER: float = 4.5 # Take Profit distance in ATR
    TRAILING_STOP_INITIAL_ATR: float = 2.5  # Initial trailing stop distance
    TRAILING_STOP_MID_ATR: float = 3.0   # trailing stop for medium profits
    TRAILING_STOP_TIGHT_ATR: float = 4  #  trailing stop for larger profits
    BREAKEVEN_PLUS_PIPS: int = 20  # Pips above break-even for stop loss

    # Risk Management
    MAX_CONSECUTIVE_LOSSES: int = 3
    MIN_WIN_RATE: float = 0.4  # 40% minimum win rate
    MAX_DRAWDOWN: float = 100  # Maximum account drawdown in base currency
    POSITION_REVERSAL_THRESHOLD: float = -15  # Loss threshold for position reversal
    MIN_PROFIT_THRESHOLD: float = 5.0  # Minimum profit to trigger conservative mode
    PROFIT_LOCK_PERCENTAGE: float = 0.6  # Lock in 60% of max profit
    PROFIT_THRESHOLD_STEP: float = 1.2  # Step size for adjusting profit threshold

    # Time-based Parameters
    MAX_POSITION_AGE_SECONDS: int = 1800  # 30 minutes
    COOLING_PERIOD_SECONDS: int = 900  # 15 minutes after losses
    NEUTRAL_HOLD_DURATION: int = 60  # Neutral state hold period in seconds

    # ML Model Parameters
    NEUTRAL_CONFIDENCE_THRESHOLD: float = 0.4
    HIGH_CONFIDENCE_THRESHOLD: float = 0.6
    MIN_PREDICTED_RETURN: float = 0.001
    MODEL_PREDICTION_LOOKBACK_PERIODS: int = 55
    MODEL_TRAINING_LOOKBACK_PERIOD: int = 1000

    # Trade Direction Memory
    TRADE_DIRECTION_MEMORY_SIZE: int = 5
    MAX_SAME_DIRECTION_TRADES: int = 2

    # Order Execution
    PRICE_DEVIATION_POINTS: int = 10  # Maximum price deviation for order execution
    ORDER_MAGIC_NUMBER: int = 234000

    # Performance Thresholds
    EXCELLENT_PERFORMANCE_WIN_RATE: float = 0.6
    POOR_PERFORMANCE_WIN_RATE: float = 0.4
    SIGNIFICANT_LOSS_THRESHOLD: float = -5

class RiskLevels:
    CONSERVATIVE = {
        "MARGIN_USAGE_LIMIT": 0.3,
        "EQUITY_RISK_PER_TRADE": 0.2,
        "SL_ATR_MULTIPLIER": 2.5,
        "TP_ATR_MULTIPLIER": 2.0,
        "MAX_VOLUME_MULTIPLIER": 1.2,
    }

    MODERATE = {
        "MARGIN_USAGE_LIMIT": 0.5,
        "EQUITY_RISK_PER_TRADE": 0.4,
        "SL_ATR_MULTIPLIER": 2.0,
        "TP_ATR_MULTIPLIER": 3.0,
        "MAX_VOLUME_MULTIPLIER": 1.5,
    }

    AGGRESSIVE = {
        "MARGIN_USAGE_LIMIT": 0.7,
        "EQUITY_RISK_PER_TRADE": 0.6,
        "SL_ATR_MULTIPLIER": 1.5,
        "TP_ATR_MULTIPLIER": 4.0,
        "MAX_VOLUME_MULTIPLIER": 2.0,
    }


@dataclass
class BackTest:
    START_DATE: datetime = datetime(2023, 12, 1, tzinfo=timezone.utc)
    END_DATE: datetime = datetime(2023, 12, 2, tzinfo=timezone.utc)
    TRAINING_LOOKBACK: int = 1000
    PREDICTION_LOOKBACK: int = 100
    TIMEFRAME: int = mt5.TIMEFRAME_M5
    # TIMEFRAME: int = MetaTrader5.TIMEFRAME_H1
    INITIAL_BALANCE: float = 10000
    # Training config
    TRAIN_WINDOW: timedelta = timedelta(days=1)
    RETRAIN_FREQUENCY: timedelta = timedelta(hours=24)
    # Prediction
    HIGH_CONFIDENCE_THRESHOLD: float = 0.6
    # Trading config
    MAX_POSITION_AGE_SECONDS: int = 1800  # 30 minutes
    BREAKEVEN_PLUS_PIPS: int = 20  # Pips above break-even for stop loss
    POSITION_REVERSAL_THRESHOLD: float = -30 
    SL_ATR_MULTIPLIER: float = 2.5  # Stop loss distance in ATR
    TP_ATR_MULTIPLIER: float = 2.0  # Take profit distance in ATR

    @staticmethod
    def get_timezone():
        """Get the timezone offset for MT5"""
        return timezone.utc


class TradingState:
    def __init__(self):
        self.is_conservative_mode: bool = False
        self.global_profit: float = 0.0
        self.symbol_states: Dict = {}


def initialize_mt5():
    """Initialize MT5 with proper path handling"""
    current_os = platform.system()
    possible_paths = MT5_PATHS.get(current_os, [])

    # Add debug logging
    logging.info(f"Current OS: {current_os}")
    logging.info(f"Checking paths: {[str(p) for p in possible_paths]}")

    # Add your specific path explicitly
    custom_path = Path.home() / ".mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"
    if custom_path.exists():
        logging.info(f"Found MT5 at custom path: {custom_path}")
        mt5_path = str(custom_path)
    else:

        mt5_path = next((str(path) for path in possible_paths if path.exists()), None)

    if not mt5_path:
        logging.error(
            f"MetaTrader 5 terminal not found in standard locations for {current_os}!"
        )
        return False

    logging.info(f"Attempting to initialize MT5 with path: {mt5_path}")

    try:
        if not mt5.initialize(
            path=mt5_path,
            login=MT5Config.ACCOUNT_NUMBER,
            password=MT5Config.PASSWORD,
            server=MT5Config.SERVER,
            timeout=60000,
        ):
            error = mt5.last_error()
            logging.error(f"MT5 initialization failed! Error: {error}")
            return False
    except Exception as e:
        logging.error(f"Exception during MT5 initialization: {str(e)}")
        return False

    logging.info(f"Successfully connected to MT5 using path: {mt5_path}")
    account = mt5.account_info()
    print(f"Balance: {account.balance}, Free Margin: {account.margin_free}")
    return True


# Global instances
# mt5 = MetaTrader5()
TRADING_CONFIG = TradingConfig()
trading_state = TradingState()
SHUTDOWN_EVENT = threading.Event()


def update_risk_profile(profile: str):
    """Update trading configuration based on risk profile"""
    risk_settings = getattr(RiskLevels, profile.upper(), RiskLevels.MODERATE)

    for param, value in risk_settings.items():
        if hasattr(TRADING_CONFIG, param):
            setattr(TRADING_CONFIG, param, value)
