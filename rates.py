from config import mt5

google = mt5.copy_rates_from_pos("GOOG", mt5.TIMEFRAME_M1, 0, 1000)

eurusd = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 1000)

# print(eurusd)

print(google) 
