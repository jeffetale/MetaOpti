# hedging/config.py

class HedgingConfig:
    BASE_VOLUME = 0.1  # Base volume for trading
    VOLUME_MULTIPLIER = 5.0  # Volume multiplier for increased positions
    MAX_POSITIONS = 15  # Maximum number of hedged (pair) positions to hold
    BASE_TRAILING_ACTIVATION = 0.3  # Activate trailing stop after 0.3 % profit
    BASE_TRAILING_STOP = 0.2  # Set trailing stop at 0.2 % profit
    BASE_PROFIT_TARGET = 10.0  # Profit targets at 20 usd
    POSITION_INCREASE_FACTOR = 1.5  # Increase position size by 50%
    MAX_LOSS_PER_PAIR = -40.0  # Stop losses at -40 usd
    TRAILING_ACTIVATION_PERCENT = 1.0  # Activate trailing at 1% profit
    LOSS_TIGHTENING_PERCENT = 5.0  # Tighten stops at 5% loss
    POSITION_INCREASE_THRESHOLD = 1.0  # Increase position size at 1% profit
    STOP_TIGHTENING_FACTOR = 0.7  # How much to tighten stops (70% closer)
    MAX_WINNING_STREAK_FACTOR = (
        2.5  # Max lot size multiplier limit for winning streak
    )
    VOLATILITY_ADJUSTMENT_FACTOR = 1.2  #
    MIN_CONFIDENCE_THRESHOLD = (
        0.57  # Minimum model confidence for opening positions
    )
    MIN_PROFIT_TICKS = 10  # Minimum profit in ticks before considering exit
    PROFIT_LOCK_THRESHOLD = (
        0.3  # Lock in profits when position reaches 0.3% gain
    )
    PARTIAL_CLOSE_THRESHOLD = 0.5  # Start partial closes at 0.5% gain
    PARTIAL_CLOSE_RATIO = 0.5  # Close 50% of position on first target
    TREND_CONTINUATION_THRESHOLD = (
        0.7  # Confidence needed to maintain position
    )
    MAX_ADVERSE_MOVE = 0.2  # Maximum allowable adverse move before closing
    PRICE_DEVIATION_POINTS = 10  # Price deviation in points to close position
