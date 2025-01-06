# 🤖 Advanced MT5 Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MT5](https://img.shields.io/badge/MT5-5.0.0+-green.svg)](https://www.metatrader5.com/)

## ⚠️ Risk Warning

```diff
- TRADING INVOLVES SUBSTANTIAL RISK OF LOSS
- THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES
- THE AUTHOR IS NOT RESPONSIBLE FOR ANY FINANCIAL LOSSES
```

This trading bot is for **educational and research purposes only**. Never risk money you cannot afford to lose. Past performance does not guarantee future results.

## 🌟 Features

- 🎯 Multi-symbol trading support
- 🧠 Machine Learning-based signal generation
- ⚡ Real-time position management
- 📊 Advanced risk management
- 🔔 Sound alerts for important events
- 📈 Performance tracking and statistics
- 🛡️ Multiple safety mechanisms
- 🔄 Automatic position reversal on conditions
- 💹 Dynamic volume sizing
- 🎚️ Trailing stop management

## 🛠️ Prerequisites

- Python 3.8 or higher
- MetaTrader 5 terminal installed
- Active MT5 trading account (demo recommended for testing)
- Required Python packages (see Installation)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/jeffetale/MetaOpti.git
cd MetaOpti
```

2. Install required packages:
   For macOS/Linux:
```bash
pip install -r requirements_unix.txt
```
   For Windows:
```bash
pip install -r requirements_windows.txt
```
*Change the Metatrader/mt5 import in config.py depending on your OS*

3. Configure your MT5 credentials in `config.py`:
```python
class MT5Config:
    PASSWORD = os.getenv("PASSWORD")
    SERVER = os.getenv("SERVER")
    ACCOUNT_NUMBER = 1234567 #Insert your account number here
    TIMEFRAME = mt5.TIMEFRAME_M15 # Use your preferred data fetching timeframe
```
4. Create a `.env` file in the projects main directory and input:
   PASSWORD="password" # your account password
   SERVER="server"     # your account server

## ⚙️ Configuration

The bot's behavior can be customized through various configuration files:

### Trading Parameters (`config.py`)
- Trading symbols
- Timeframes
- Risk parameters
- Account credentials

### Symbol Settings (`symbols.py`)
- Define trading and backtest symbols
- Customize per-symbol parameters

## 🚀 Usage

1. Start the bot:
```bash
python main.py
```

2. Monitor the logs for trading activity
3. Use the sound alerts system for real-time notifications
4. Check trading statistics for performance analysis

## 🎛️ Trading Modes

The bot supports multiple trading modes that can be configured in `config.py`:
- AGGRESSIVE
- MODERATE
- CONSERVATIVE

```python
# Uncomment your preferred mode
#update_risk_profile('AGGRESSIVE')
#update_risk_profile('MODERATE')
#update_risk_profile('CONSERVATIVE')
```
*Change these to your preference in the modules under trading directory*

## 🛡️ Safety Features

- ✋ Maximum position age monitoring
- 💰 Dynamic position sizing
- 🎯 Breakeven plus protection
- 📊 Advanced trailing stops
- ⛔ Maximum loss limits
- 🔄 Automatic position reversal
- 📈 Profit protection mechanisms

## 📊 Position Management

The bot includes sophisticated position management features:
- Trailing stops with multiple levels
- Breakeven plus functionality
- Profit lock-in mechanisms
- Position aging controls
- Reversal strategies

## 🔔 Alerts System

The bot includes an advanced alerts system (`trade_alerts.py`) that provides:
- Sound notifications for important events
- Critical loss warnings
- Profit milestones
- Position status changes

## 🔄 Backtesting System

> ⚠️ **Note: Backtesting is currently only available on Windows systems** due to MT5 terminal dependencies.

### 📊 Backtesting Features

- 📈 Multi-symbol backtesting
- 🕒 Custom date range testing
- 💰 Configurable initial balance
- ⚡ Multiple timeframe support
- 📊 Detailed performance metrics
- 📉 Drawdown analysis
- 📈 Equity curve visualization
- 🔍 Trade-by-trade analysis

### ⚙️ Configuration

Modify backtesting parameters in `config.py`:

```python
class BacktestConfig:
    INITIAL_BALANCE = 10000  # Starting balance
    START_DATE = "2023-01-01"  # Backtest start date
    END_DATE = "2024-01-01"    # Backtest end date
    TIMEFRAME = mt5.TIMEFRAME_M15  # Timeframe for testing
    RISK_PER_TRADE = 0.02     # 2% risk per trade
    SPREAD_POINTS = 20        # Spread simulation
```
*You can set the backtest symbols in symbols.py*

### 🚀 Running Backtests

1. Configure your parameters in `config.py`
2. Run the backtest script:
```bash
python3 backtest/backtest.py
```

### 📈 Backtest Results

The system generates comprehensive reports including:
- Overall performance metrics
- Trade statistics
- Monthly returns
- Maximum drawdown
- Win rate and profit factor
- Risk-adjusted returns
- Equity curve chart
- Trade distribution analysis

Example output:
```
Backtest Results Summary
------------------------
Total Trades: 124
Win Rate: 62.9%
Profit Factor: 1.85
Max Drawdown: 4.2%
Sharpe Ratio: 1.92
Initial Balance: $10,000
Final Balance: $14,320
Net Profit: $4,320 (43.2%)
```

## 🗂️ Project Structure

```
MetaOpti/
├── main.py              # Main bot entry point
├── config.py            # Configuration settings
├── symbols.py           # Symbol definitions
├── trade_alerts.py      # Alerts system
├── trading/
│   ├── order_manager.py     # Order handling
│   ├── position_manager.py  # Position management
│   └── risk_manager.py      # Risk controls
├── models/
│   └── trading_state.py     # Trading state management
├── ml/
│   ├── predictor.py         # ML model predictions
│   ├── trainer.py           # Model training pipeline
│   ├── model_optimization.py          # For gpu training
│   └── background_train.py         # Background model training while live tradind
└── backtest/
    ├── backtest.py         # Backtesting engine
    ├── backtest_model_trainer.py         # Model training for backtest
    ├── backtest_data_fetcher.py       # Backtest Data Fetching
    └── backtest_data_preparation.py       # Backtest Data Preparation
```

## 🐛 Common Issues & Solutions

1. **MT5 Initialization Failed**
   - Verify MT5 installation path
   - Check account credentials
   - Ensure MT5 terminal is running

2. **Symbol Not Found**
   - Verify symbol availability in your MT5 terminal
   - Check symbol spelling and format
   - Ensure market is open for symbol

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Final Warning

```diff
! This is a complex trading system that can result in financial losses
! Never use it with money you cannot afford to lose
! Always test thoroughly on a demo account first
! Past performance does not indicate future results
```

## 📞 Support

For issues and feature requests, please use the GitHub issues system.

---

**Remember:** This is experimental software. Use at your own risk. The author accepts no responsibility for any financial losses incurred through the use of this software.
