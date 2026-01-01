import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STRATEGY CONSTANTS
# ============================================================================

# Capital and Risk Management
BASE_CAPITAL = 1000000  # 10 lac
CAPITAL_ALLOCATION_PER_STOCK = 0.10  # 10% of base capital per stock
RISK_PER_TRADE = 0.005  # 0.5% of allocated capital

# Trading Parameters
PROFIT_TARGET_MULTIPLIER = 1.02  # 2% target
STOP_LOSS_MULTIPLIER = 0.995  # 0.5% stop loss
TRAILING_START_MULTIPLIER = 1.005  # Start trailing after 0.5% profit
TRAILING_STOP_MULTIPLIER = 0.9925  # 0.75% trailing stop

# Indicator Parameters
EMA_FAST_PERIOD = 3
EMA_SLOW_PERIOD = 10
EMA_HOURLY_PERIOD = 50
RSI_PERIOD = 14

# Time Parameters
MARKET_OPEN_TIME = "09:15:00"
STOCK_SELECTION_TIME = "09:25:00"
MARKET_CLOSE_TIME = "15:30:00"
TRADING_HOURS_PER_DAY = 6.25

# Data Requirements
MIN_HISTORICAL_DAYS = 8
MIN_HOURLY_CANDLES = 50

# Signal Thresholds
RSI_LONG_THRESHOLD = 60
RSI_SHORT_THRESHOLD = 30

# Timeframes
TIMEFRAME_10MIN = '10T'
TIMEFRAME_1HOUR = '1H'

# Exit Reasons
EXIT_SL = "Stop Loss"
EXIT_TP = "Target Hit"
EXIT_TS = "Trailing Stop"
EXIT_EOD = "End of Day"

# ============================================================================
# STRATEGY CLASS
# ============================================================================

class IntradayEMA_RSI_Strategy:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.trade_log = []
        self.selected_stocks = {}
        self.all_dates = []
        self.symbol_cache = {}  # Cache for symbol data
        
    def discover_dates(self):
        """Discover all available dates from the data folder"""
        files = glob.glob(os.path.join(self.data_folder, "dataNSE_*.csv"))
        dates = []
        
        for file in files:
            try:
                date_str = file.split('_')[-1].split('.')[0]
                date = datetime.strptime(date_str, "%Y%m%d").date()
                dates.append(date)
            except Exception:
                continue
        
        return sorted(dates)
    
    def load_data_for_turnover_calculation(self, date):
        """Load data for turnover calculation (first 10 minutes)"""
        date_str = date.strftime("%Y%m%d")
        file_path = os.path.join(self.data_folder, f"dataNSE_{date_str}.csv")
        
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path)
        df['date'] = date
        df['datetime'] = pd.to_datetime(df['time'])
        
        # Filter for first 10 minutes
        selection_time = pd.Timestamp(f"{date} {STOCK_SELECTION_TIME}")
        df = df[df['datetime'] < selection_time]
        
        return df
    
    def calculate_turnover(self, date):
        """Calculate turnover for first 10 minutes (9:15-9:25)"""
        df = self.load_data_for_turnover_calculation(date)
        
        if df is None or df.empty:
            return []
        
        # Calculate turnover: sum(volume * close) for each ticker
        turnover_dict = {}
        for ticker, group in df.groupby('ticker'):
            # Ensure we have data up to 9:24 (at least 10 candles)
            if len(group) >= 10:
                turnover = (group['volume'] * group['close']).sum()
                turnover_dict[ticker] = turnover
        
        # Sort and get top 10
        sorted_tickers = sorted(turnover_dict.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:10]
        
        return [ticker for ticker, _ in sorted_tickers]
    
    def load_data_for_symbol(self, ticker, current_date):
        """Load data specifically for a symbol including historical data"""
        # Check cache first
        cache_key = f"{ticker}_{current_date}"
        if cache_key in self.symbol_cache:
            return self.symbol_cache[cache_key].copy()
        
        # Load current day's data
        date_str = current_date.strftime("%Y%m%d")
        current_file = os.path.join(self.data_folder, f"dataNSE_{date_str}.csv")
        
        if not os.path.exists(current_file):
            return None
        
        # Read and filter for specific symbol
        current_df = pd.read_csv(current_file)
        current_df = current_df[current_df['ticker'] == ticker].copy()
        
        if current_df.empty:
            return None
        
        current_df['date'] = current_date
        current_df['datetime'] = pd.to_datetime(current_df['time'])
        current_df.sort_values('datetime', inplace=True)
        
        # Check if we need historical data for indicators
        unique_dates = sorted(self.all_dates)
        try:
            current_idx = list(unique_dates).index(current_date)
        except ValueError:
            # If current date not found, return only current day data
            self.symbol_cache[cache_key] = current_df
            return current_df.copy()
        
        # Load historical data (previous MIN_HISTORICAL_DAYS)
        historical_data = []
        start_idx = max(0, current_idx - MIN_HISTORICAL_DAYS)
        
        for i in range(start_idx, current_idx):
            hist_date = unique_dates[i]
            hist_date_str = hist_date.strftime("%Y%m%d")
            hist_file = os.path.join(self.data_folder, f"dataNSE_{hist_date_str}.csv")
            
            if os.path.exists(hist_file):
                try:
                    hist_df = pd.read_csv(hist_file)
                    hist_df = hist_df[hist_df['ticker'] == ticker].copy()
                    
                    if not hist_df.empty:
                        hist_df['date'] = hist_date
                        hist_df['datetime'] = pd.to_datetime(hist_df['time'])
                        hist_df.sort_values('datetime', inplace=True)
                        historical_data.append(hist_df)
                except Exception as e:
                    print(f"Warning: Could not load historical data for {ticker} on {hist_date}: {e}")
                    continue
        
        # Combine historical and current data
        if historical_data:
            historical_df = pd.concat(historical_data, ignore_index=True)
            combined_df = pd.concat([historical_df, current_df], ignore_index=True)
            combined_df.sort_values(['date', 'datetime'], inplace=True)
        else:
            combined_df = current_df.copy()
        
        # Cache the result
        self.symbol_cache[cache_key] = combined_df.copy()
        
        return combined_df.copy()
    
    def has_sufficient_historical_data(self, ticker, date):
        """Check if we have sufficient historical data for indicators"""
        # Get data for this symbol
        symbol_data = self.load_data_for_symbol(ticker, date)
        
        if symbol_data is None or symbol_data.empty:
            return False
        
        # Count unique dates in the data
        unique_dates = symbol_data['date'].unique()
        
        if len(unique_dates) < MIN_HISTORICAL_DAYS:
            return False
        
        # Check if we have at least MIN_HOURLY_CANDLES across these days
        # Exclude current day for this check
        historical_data = symbol_data[symbol_data['date'] < date]
        
        if historical_data.empty:
            return False
        
        # Resample to hourly and count candles
        historical_data.set_index('datetime', inplace=True)
        hourly_data = historical_data.resample(TIMEFRAME_1HOUR).agg({
            'close': 'last'
        }).dropna()
        
        return len(hourly_data) >= MIN_HOURLY_CANDLES
    
    def calculate_indicators_for_stock(self, ticker, current_date, current_time=None):
        """Calculate all indicators for a stock at a given time"""
        # Load data for this symbol
        symbol_data = self.load_data_for_symbol(ticker, current_date)
        
        if symbol_data is None or symbol_data.empty:
            return {}
        
        # Filter data up to current_time
        if current_time:
            filtered_data = symbol_data[symbol_data['datetime'] <= current_time].copy()
        else:
            filtered_data = symbol_data.copy()
        
        if filtered_data.empty:
            return {}
        
        indicators = {}
        
        # Separate current day data for minute-by-minute analysis
        current_day_data = filtered_data[filtered_data['date'] == current_date]
        historical_data = filtered_data[filtered_data['date'] < current_date]
        
        # Combine for indicator calculations
        combined_for_indicators = pd.concat([historical_data, current_day_data], ignore_index=True)
        combined_for_indicators.sort_values('datetime', inplace=True)
        
        # Calculate EMA(50) on hourly timeframe
        combined_for_indicators.set_index('datetime', inplace=True)
        
        try:
            hourly_resampled = combined_for_indicators.resample(TIMEFRAME_1HOUR).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(hourly_resampled) >= EMA_HOURLY_PERIOD:
                indicators['ema_50'] = hourly_resampled['close'].ewm(
                    span=EMA_HOURLY_PERIOD, adjust=False).mean().iloc[-1]
        except Exception as e:
            print(f"Warning: Error calculating hourly EMA for {ticker}: {e}")
            indicators['ema_50'] = None
        
        # Calculate EMA(3), EMA(10), RSI(14) on 10-minute timeframe
        try:
            ten_min_resampled = combined_for_indicators.resample(TIMEFRAME_10MIN).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(ten_min_resampled) >= max(EMA_SLOW_PERIOD, RSI_PERIOD):
                # Calculate EMAs
                indicators['ema_3'] = ten_min_resampled['close'].ewm(
                    span=EMA_FAST_PERIOD, adjust=False).mean().iloc[-1]
                indicators['ema_10'] = ten_min_resampled['close'].ewm(
                    span=EMA_SLOW_PERIOD, adjust=False).mean().iloc[-1]
                
                # Calculate RSI
                delta = ten_min_resampled['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
                rs = gain / loss
                indicators['rsi_14'] = 100 - (100 / (1 + rs)).iloc[-1]
        except Exception as e:
            print(f"Warning: Error calculating 10-min indicators for {ticker}: {e}")
            indicators['ema_3'] = None
            indicators['ema_10'] = None
            indicators['rsi_14'] = None
        
        return indicators
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk management rules"""
        capital_per_stock = BASE_CAPITAL * CAPITAL_ALLOCATION_PER_STOCK
        risk_amount = capital_per_stock * RISK_PER_TRADE
        
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share > 0:
            position_size = int(risk_amount / risk_per_share)
            return max(1, position_size)  # At least 1 share
        return 0
    
    def check_entry_signals(self, indicators, current_price):
        """Check for long and short entry signals"""
        signals = {'long': False, 'short': False}
        
        # Check if all indicators are available and valid
        required_indicators = ['ema_3', 'ema_10', 'ema_50', 'rsi_14']
        if not all(ind in indicators for ind in required_indicators):
            return signals
        
        # Check for None values
        if any(indicators[ind] is None for ind in required_indicators):
            return signals
        
        ema_3 = indicators['ema_3']
        ema_10 = indicators['ema_10']
        ema_50 = indicators['ema_50']
        rsi_14 = indicators['rsi_14']
        
        # Long signal
        if (ema_3 > ema_10 and 
            rsi_14 > RSI_LONG_THRESHOLD and 
            current_price > ema_50):
            signals['long'] = True
        
        # Short signal
        elif (ema_3 < ema_10 and 
              rsi_14 < RSI_SHORT_THRESHOLD and 
              current_price < ema_50):
            signals['short'] = True
        
        return signals
    
    def process_stock_for_day(self, ticker, date):
        """Process trading for a single stock on a single day"""
        # Load data for this symbol
        symbol_data = self.load_data_for_symbol(ticker, date)
        
        if symbol_data is None:
            print(f"  Warning: No data found for {ticker} on {date}")
            return
        
        # Get only current day's data for minute-by-minute processing
        day_data = symbol_data[symbol_data['date'] == date].copy()
        
        if day_data.empty:
            print(f"  Warning: No current day data for {ticker} on {date}")
            return
        
        day_data.sort_values('datetime', inplace=True)
        
        # Check if we have enough data points for the day
        if len(day_data) < 50:  # Minimum 50 minutes of data
            print(f"  Warning: Insufficient data points for {ticker} on {date}")
            return
        
        # Initialize position tracking
        current_position = None
        last_indicators_update = None
        
        for idx, row in day_data.iterrows():
            current_time = row['datetime']
            current_price = row['close']
            
            # Update indicators at appropriate times
            should_update_indicators = False
            
            # Update at 9:25 and then every 10 minutes
            if (current_time.minute % 10 == 5 and current_time.hour >= 9) or (
                current_time.hour == 9 and current_time.minute == 25):
                should_update_indicators = True
            
            # Update EMA(50) every hour at :15
            if current_time.minute == 15 and current_time.hour >= 10:
                should_update_indicators = True
            
            if should_update_indicators:
                try:
                    indicators = self.calculate_indicators_for_stock(
                        ticker, date, current_time)
                    last_indicators_update = indicators
                except Exception as e:
                    print(f"    Error calculating indicators for {ticker} at {current_time}: {e}")
                    continue
            
            # Check for entry signals if we have updated indicators
            if (last_indicators_update and 
                current_position is None and 
                current_time.time() >= pd.Timestamp(STOCK_SELECTION_TIME).time()):
                
                try:
                    signals = self.check_entry_signals(last_indicators_update, current_price)
                except Exception as e:
                    print(f"    Error checking signals for {ticker} at {current_time}: {e}")
                    continue
                
                if signals['long']:
                    # Long entry
                    entry_price = row['high']  # Buy at high of entry candle
                    stop_loss = entry_price * STOP_LOSS_MULTIPLIER
                    target_price = entry_price * PROFIT_TARGET_MULTIPLIER
                    
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    
                    if position_size > 0:
                        current_position = {
                            'ticker': ticker,
                            'side': 'LONG',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'position_size': position_size,
                            'stop_loss': stop_loss,
                            'target_price': target_price,
                            'trailing_start': entry_price * TRAILING_START_MULTIPLIER,
                            'trailing_active': False,
                            'trailing_stop': None
                        }
                        print(f"LONG entry: {ticker} at {entry_price:.2f}, "
                              f"Size: {position_size}, Time: {current_time}")
                
                elif signals['short']:
                    # Short entry - get low of last 5 candles
                    last_5_idx = max(0, idx - 4)
                    last_5_candles = day_data.iloc[last_5_idx:idx+1]
                    entry_price = last_5_candles['low'].min()
                    
                    stop_loss = entry_price * (2 - STOP_LOSS_MULTIPLIER)  # For short
                    target_price = entry_price * (2 - PROFIT_TARGET_MULTIPLIER)
                    
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    
                    if position_size > 0:
                        current_position = {
                            'ticker': ticker,
                            'side': 'SHORT',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'position_size': position_size,
                            'stop_loss': stop_loss,
                            'target_price': target_price,
                            'trailing_start': entry_price * (2 - TRAILING_START_MULTIPLIER),
                            'trailing_active': False,
                            'trailing_stop': None
                        }
                        print(f"SHORT entry: {ticker} at {entry_price:.2f}, "
                              f"Size: {position_size}, Time: {current_time}")
            
            # Check exit conditions if in position
            if current_position:
                try:
                    exit_signal = self.check_exit_conditions(current_position, row)
                    if exit_signal:
                        self.close_position(current_position, exit_signal, row)
                        current_position = None
                except Exception as e:
                    print(f"Error checking exit conditions for {ticker} at {current_time}: {e}")
                    continue
        
        # Square off any remaining position at market close
        if current_position:
            last_row = day_data.iloc[-1]
            self.close_position(current_position, EXIT_EOD, last_row)
    
    def check_exit_conditions(self, position, current_row):
        """Check if exit conditions are met for a position"""
        current_price = current_row['close']
        current_time = current_row['datetime']
        
        if position['side'] == 'LONG':
            # Check stop loss
            if current_price <= position['stop_loss']:
                return EXIT_SL
            
            # Check target
            if current_price >= position['target_price']:
                return EXIT_TP
            
            # Check trailing stop
            if current_price >= position['trailing_start']:
                if not position['trailing_active']:
                    position['trailing_active'] = True
                    position['trailing_stop'] = current_price * TRAILING_STOP_MULTIPLIER
                else:
                    new_trailing_stop = current_price * TRAILING_STOP_MULTIPLIER
                    position['trailing_stop'] = max(position['trailing_stop'], new_trailing_stop)
                
                if current_price <= position['trailing_stop']:
                    return EXIT_TS
        
        else:  # SHORT position
            # Check stop loss (inverse for short)
            if current_price >= position['stop_loss']:
                return EXIT_SL
            
            # Check target (inverse for short)
            if current_price <= position['target_price']:
                return EXIT_TP
            
            # Check trailing stop (inverse for short)
            if current_price <= position['trailing_start']:
                if not position['trailing_active']:
                    position['trailing_active'] = True
                    position['trailing_stop'] = current_price * (2 - TRAILING_STOP_MULTIPLIER)
                else:
                    new_trailing_stop = current_price * (2 - TRAILING_STOP_MULTIPLIER)
                    position['trailing_stop'] = min(position['trailing_stop'], new_trailing_stop)
                
                if current_price >= position['trailing_stop']:
                    return EXIT_TS
        
        return None
    
    def close_position(self, position, exit_reason, exit_row):
        """Close a position and record the trade"""
        exit_price = exit_row['close']
        exit_time = exit_row['datetime']
        
        # Calculate P&L
        if position['side'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Record trade
        trade_record = {
            'Ticker': position['ticker'],
            'Entry Time': position['entry_time'],
            'Exit Time': exit_time,
            'Entry Price': round(position['entry_price'], 2),
            'Exit Price': round(exit_price, 2),
            'Side': position['side'],
            'Quantity': position['position_size'],
            'PNL': round(pnl, 2),
            'Exit Reason': exit_reason
        }
        
        self.trade_log.append(trade_record)
        
        # Print trade summary
        pnl_color = "\033[92m" if pnl > 0 else "\033[91m"  # Green for profit, red for loss
        print(f"{pnl_color}Trade Closed: {position['ticker']} {position['side']} | "
              f"Entry: {position['entry_price']:.2f} | Exit: {exit_price:.2f} | "
              f"PNL: ₹{pnl:.2f} | Reason: {exit_reason}\033[0m")
    
    def process_trading_day(self, date):
        """Process a complete trading day"""
        print(f"\nProcessing Date: {date}")
        
        # Select top 10 stocks by turnover
        selected_stocks = self.calculate_turnover(date)
        
        if not selected_stocks:
            print(f"No stocks selected for trading")
            return
        
        print(f"Selected Stocks: {', '.join(selected_stocks)}")
        self.selected_stocks[date] = selected_stocks
        
        # Filter stocks with sufficient historical data
        valid_stocks = []
        for ticker in selected_stocks:
            if self.has_sufficient_historical_data(ticker, date):
                valid_stocks.append(ticker)
            else:
                print(f"Skipping {ticker}: Insufficient historical data")
        
        if not valid_stocks:
            print(f"No stocks with sufficient historical data")
            return
        
        # Process each selected stock
        for ticker in valid_stocks:
            print(f"\nProcessing {ticker}")
            self.process_stock_for_day(ticker, date)
    
    def calculate_performance_metrics(self):
        """Calculate only essential performance metrics: return, drawdown, win rate, sharpe ratio"""
        if not self.trade_log:
            print("\nNo trades were executed during the period")
            return None, None
        
        trades_df = pd.DataFrame(self.trade_log)
        
        # Calculate only the required metrics
        
        # Total Return %
        total_pnl = trades_df['PNL'].sum()
        total_return_pct = (total_pnl / BASE_CAPITAL) * 100
        
        # Win Rate %
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PNL'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Drawdown calculation
        trades_df['Cumulative PNL'] = trades_df['PNL'].cumsum()
        trades_df['Running Max'] = trades_df['Cumulative PNL'].expanding().max()
        trades_df['Drawdown'] = trades_df['Cumulative PNL'] - trades_df['Running Max']
        max_drawdown = abs(trades_df['Drawdown'].min()) if trades_df['Drawdown'].min() < 0 else 0
        max_drawdown_pct = (max_drawdown / BASE_CAPITAL) * 100
        
        # Sharpe Ratio (assuming risk-free rate = 0 for intraday)
        returns = trades_df['PNL'] / BASE_CAPITAL
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Create simplified metrics dictionary with only required metrics
        metrics = {
            'Total Return %': round(total_return_pct, 2),
            'Max Drawdown %': round(max_drawdown_pct, 2),
            'Win Rate %': round(win_rate, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2)
        }
        
        return metrics, trades_df
    
    def run_strategy(self):
        # Discover available dates
        print("Discovering available trading days...")
        self.all_dates = self.discover_dates()
        
        if not self.all_dates:
            print("Error: No valid data files found in the specified folder")
            return
        
        print(f"Found data for {len(self.all_dates)} trading days")
        print(f"Date range: {self.all_dates[0]} to {self.all_dates[-1]}")
        
        # Process each trading day
        print("\nStarting strategy execution...")
        for date in self.all_dates:
            self.process_trading_day(date)
        
        metrics, trades_df = self.calculate_performance_metrics()
        
        if metrics:
            # Save trade log with essential columns only
            essential_cols = ['Ticker', 'Entry Time', 'Exit Time', 'Side', 
                            'Entry Price', 'Exit Price', 'PNL', 'Exit Reason']
            trades_df[essential_cols].to_csv('trades_log.csv', index=False)

            print(f"\nTrade log saved to: trades_log.csv ({len(trades_df)} trades)")
            
            # Save simplified performance metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv('performance_metrics.csv', index=False)
            print(f"Performance metrics saved to: performance_metrics.csv")
            
            print(f"\nEssential Metrics Summary:")
            print(f"Total Return: {metrics['Total Return %']:.2f}%")
            print(f"Max Drawdown: {metrics['Max Drawdown %']:.2f}%")
            print(f"Win Rate: {metrics['Win Rate %']:.2f}%")
            print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
            
            print("\nStrategy execution completed successfully!")
        else:
            print("No trades were executed. Check data and parameters.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python intraday_ema_rsi.py <data_folder_path>")
        print("Example: python intraday_ema_rsi.py ./nse_data/")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' does not exist")
        sys.exit(1)
    
    try:
        strategy = IntradayEMA_RSI_Strategy(data_folder)
        strategy.run_strategy()
    except Exception as e:
        print(f"Error running strategy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()