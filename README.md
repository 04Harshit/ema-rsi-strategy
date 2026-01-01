# Intraday EMA-RSI Trading Strategy


![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Trading](https://img.shields.io/badge/Algorithmic-Trading-orange.svg)
![NSE](https://img.shields.io/badge/Data-NSE%20India-yellow.svg)


A comprehensive algorithmic trading system implementing an intraday mean reversion strategy using EMA crossovers and RSI indicators for the Indian stock market (NSE).


## 📋 Table of Contents
- [Strategy Overview](#-strategy-overview)
- [Key Features](#-key-features)
- [Strategy Logic](#-strategy-logic)
- [Important Assumptions](#-important-assumptions)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Files](#-output-files)
- [Performance Metrics](#-performance-metrics)
- [Risk Management](#️-risk-management)
- [File Structure](#-file-structure)
- [Limitations & Considerations](#-limitations--considerations)
- [Disclaimer](#-disclaimer)
- [License](#-license)


## 🎯 Strategy Overview


This algorithm implements an automated intraday trading strategy that:
- Selects top 10 stocks by morning turnover (9:15-9:25 AM)
- Uses multi-timeframe technical analysis (10-min and hourly)
- Executes trades based on EMA crossovers and RSI signals
- Implements comprehensive risk management
- Logs all trades and calculates key performance metrics


## ✨ Key Features


- **Intelligent Stock Selection**: Dynamic selection based on morning liquidity
- **Multi-Timeframe Analysis**: Combines 10-minute and hourly indicators
- **Complete Risk Management**: Position sizing, stop losses, and trailing stops
- **Automated Execution**: Fully automated trade entry and exit logic
- **Performance Tracking**: Detailed logging and essential metrics calculation
- **Configurable Parameters**: Easy adjustment of strategy constants


## 🧠 Strategy Logic


### Entry Signals
- **Long Entry**: 
  - EMA(3) > EMA(10) on 10-minute chart
  - RSI(14) > 60 (overbought bounce)
  - Price > EMA(50) on hourly chart


- **Short Entry**:
  - EMA(3) < EMA(10) on 10-minute chart
  - RSI(14) < 30 (oversold bounce)
  - Price < EMA(50) on hourly chart


### Exit Conditions
- **Take Profit**: 2% target
- **Stop Loss**: 0.5% initial stop
- **Trailing Stop**: 0.75% trailing stop (activates after 0.5% profit)
- **End of Day**: All positions squared off at market close


## ⚠️ Important Assumptions


This implementation makes several important assumptions due to incomplete information in the original document:


### 1. **Data Structure & Format**
- Assumes CSV files named `dataNSE_YYYYMMDD.csv`
- Assumes columns: `ticker`, `time`, `open`, `high`, `low`, `close`, `volume`
- Assumes `time` column is in a format parseable by pandas
- Assumes data is in minute intervals (1-minute candles)


### 2. **Trading Execution**
- **Entry Prices**: 
  - Long positions: Assumes entry at the high of the signal candle
  - Short positions: Assumes entry at the low of the last 5 candles (more conservative)
- **Slippage**: Not accounted for (ideal execution)
- **Market Orders**: Assumes instant execution at specified prices
- **Transaction Costs**: Brokerage, taxes, and fees are NOT included in P&L calculation


### 3. **Timing & Scheduling**
- Indicator updates assumed at:
  - 9:25 AM (stock selection time)
  - Every 10 minutes thereafter (at XX:05, XX:15, etc.)
  - Every hour at :15 for EMA(50) updates
- Assumes continuous data without gaps or missing intervals


### 4. **Market Conditions**
- Assumes liquidity for all top 10 selected stocks
- Assumes no circuit filters or trading halts
- Assumes continuous trading during market hours (9:15 AM - 3:30 PM)
- No consideration for news events, earnings, or corporate actions


### 5. **Position Management**
- **Maximum Positions**: Can theoretically hold up to 10 positions simultaneously (one per selected stock)
- **No Pyramid/Add-on**: Does not add to existing positions
- **Square-off**: All positions forcibly closed at 3:30 PM
- **No overnight positions allowed**


### 6. **Technical Indicators**
- EMA calculations use `adjust=False` (standard trading convention)
- RSI calculation uses standard formula with 14-period lookback
- Assumes sufficient historical data (8 days) for reliable indicator calculation
- Hourly data resampled from minute data (may differ from actual hourly candles)


### 7. **Risk Management Simplifications**
- Position sizing based on static risk percentage (0.5%)
- Stop losses are static percentages, not volatility-based
- No correlation adjustment between positions
- No portfolio-level risk management
- Capital allocation fixed at 10% per stock regardless of volatility


## 🚀 Installation


### Prerequisites
- Python 3.7 or higher
- pandas, numpy libraries


### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/intraday-ema-rsi.git
cd intraday-ema-rsi


# Install required packages
pip install pandas numpy
```


## 📊 Usage


### Data Format
Place your NSE data files in a folder with the naming convention:
```
data_folder/
├── dataNSE_YYYYMMDD.csv
├── dataNSE_YYYYMMDD.csv
└── ...
```


Each CSV should contain the following columns: `ticker`, `time`, `open`, `high`, `low`, `close`, `volume`


### Running the Strategy
```bash
python intraday_ema_rsi.py ./path/to/your/data/folder/
```


### Example
```bash
python intraday_ema_rsi.py ./nse_data/
```


## 📁 Output Files


The strategy generates two essential output files:


### 1. `trades_log.csv`
Contains detailed records of all executed trades:
- Ticker symbol
- Entry and exit times
- Entry and exit prices
- Position side (LONG/SHORT)
- Quantity
- P&L per trade
- Exit reason (Stop Loss, Target Hit, Trailing Stop, End of Day)


### 2. `performance_metrics.csv`
Contains four key performance indicators:
- Total Return %
- Max Drawdown %
- Win Rate %
- Sharpe Ratio


## 📈 Performance Metrics


The strategy calculates and reports these essential metrics:


| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Total Return %** | Overall profitability | > 0% |
| **Max Drawdown %** | Maximum peak-to-trough decline | < 10% |
| **Win Rate %** | Percentage of profitable trades | > 50% |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 |


## 🛡️ Risk Management


### Capital Allocation
- **Base Capital**: ₹10,00,000 (configurable)
- **Per Stock Allocation**: 10% of base capital
- **Risk Per Trade**: 0.5% of allocated capital


### Position Sizing
Position size is calculated dynamically based on:
```
Position Size = (Capital per Stock × Risk per Trade) ÷ (Entry Price - Stop Loss Price)
```


### Risk Controls
- Maximum 10 positions concurrently (top 10 stocks)
- Individual position risk limited to 0.5%
- Automatic stop loss on every position
- Trailing stops to protect profits


## 📂 File Structure


```
intraday-ema-rsi/
├── intraday_ema_rsi.py    # Main strategy code
├── trades_log.csv         # Generated: Trade records
├── performance_metrics.csv# Generated: Performance metrics
├── README.md              # This file
├── LICENSE                # MIT License
├── .gitignore            # Git ignore file
└── data/                  # Your data folder (not included)
    └── dataNSE_*.csv     # NSE minute data files
```


## ⚙️ Configuration


Key parameters in the code (easily modifiable):


```python
# Capital and Risk
BASE_CAPITAL = 1000000
CAPITAL_ALLOCATION_PER_STOCK = 0.10
RISK_PER_TRADE = 0.005


# Trading Parameters
PROFIT_TARGET_MULTIPLIER = 1.02
STOP_LOSS_MULTIPLIER = 0.995


# Indicator Parameters
EMA_FAST_PERIOD = 3
EMA_SLOW_PERIOD = 10
EMA_HOURLY_PERIOD = 50
RSI_PERIOD = 14


# Time Parameters
STOCK_SELECTION_TIME = "09:25:00"
MARKET_CLOSE_TIME = "15:30:00"
```


## 🔧 Limitations & Considerations


### Known Limitations
1. **No Transaction Costs**: Brokerage, STT, GST, SEBI charges not included
2. **Ideal Execution**: No slippage, instant fills assumed
3. **Data Quality**: Assumes clean, continuous data without errors
4. **Market Impact**: Large orders don't affect prices
5. **No Blackout Periods**: Trades during news/events without restrictions


### For Live Trading Consider
1. Add brokerage and tax calculations (approx. 0.05-0.1% per trade)
2. Implement circuit filter checks
3. Add volume filters for entry/exit
4. Consider bid-ask spreads
5. Add maximum position limits per stock
6. Implement actual order placement logic
