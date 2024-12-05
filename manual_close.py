# manual_close.py

from models import trading_state
from main import logging, mt5, close_position
from time import datetime

def validate_positions():
    """
    Comprehensive position validation and reconciliation
    Call this after any suspected manual intervention
    """
    positions = mt5.positions_get()
    if positions is None:
        return []
    
    validated_positions = []
    for position in positions:
        try:
            # Detailed position validation
            if position.volume <= 0:
                logging.warning(f"Invalid position detected: {position.ticket}")
                try:
                    # Attempt to close invalid positions
                    close_position(position)
                except Exception as e:
                    logging.error(f"Failed to close invalid position {position.ticket}: {e}")
                continue
            
            # Log position details for debugging
            logging.info(f"Validated Position: {position.symbol} "
                         f"Ticket: {position.ticket}, "
                         f"Volume: {position.volume}, "
                         f"Profit: {position.profit}, "
                         f"Open Price: {position.price_open}")
            
            validated_positions.append(position)
        
        except Exception as e:
            logging.error(f"Error validating position {position.ticket}: {e}")
    
    return validated_positions

def detect_manual_intervention(symbol):
    """
    Detect and handle manual interventions
    """
    # Get current positions
    current_positions = mt5.positions_get(symbol=symbol)
    
    # Retrieve the previous state from trading_state
    state = trading_state.symbol_states[symbol]
    
    # Check for unexpected changes
    if current_positions is None:
        # No positions exist when they should
        if hasattr(state, 'last_known_positions') and state.last_known_positions:
            logging.warning(f"Possible manual intervention detected for {symbol}: All positions closed")
            trading_state.manual_intervention_detected = True
            trading_state.last_manual_intervention_time = datetime.now()
            trading_state.manual_intervention_cooldown = 2  # 2 hours cooldown
            return True
    
    # If positions exist, compare with last known state
    if hasattr(state, 'last_known_positions'):
        # Check for discrepancies in number of positions or their details
        if len(current_positions) != len(state.last_known_positions):
            logging.warning(f"Position count changed for {symbol}: Possible manual intervention")
            trading_state.manual_intervention_detected = True
            trading_state.last_manual_intervention_time = datetime.now()
            trading_state.manual_intervention_cooldown = 2  # 2 hours cooldown
            return True
        
        # Check individual position details
        for curr_pos, last_pos in zip(current_positions, state.last_known_positions):
            if (curr_pos.volume != last_pos.volume or 
                abs(curr_pos.profit - last_pos.profit) > 0.01):  # Allow small floating-point differences
                logging.warning(f"Position details changed for {symbol}: Possible manual intervention")
                trading_state.manual_intervention_detected = True
                trading_state.last_manual_intervention_time = datetime.now()
                trading_state.manual_intervention_cooldown = 2  # 2 hours cooldown
                return True
    
    return False