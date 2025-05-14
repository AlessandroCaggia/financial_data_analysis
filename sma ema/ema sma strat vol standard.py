import pandas as pd
import numpy as np
import yfinance as yf
import itertools
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# Define symbols list
symbols = ['GC=F']  # Add more symbols as needed 'NDX', '^SPX', 'CC=F','GOOG', 'AAPL'

# Panel
start_date = '2023-04-21'
end_date = datetime.today().strftime('%Y-%m-%d')
interval = '1d'  # Change this to '5m', '1d', or '1wk'

short_windows = range(2, 20, 2)
long_windows = range(2, 100, 10)

scenarios = [
    (True, True),  # EMA-EMA
    (False, False),  # SMA-SMA
    (True, False),  # EMA-SMA
    (False, True)  # SMA-EMA
]

# Function to adjust the start date based on the interval and moving average windows
def adjust_start_date(start_date: str, short_window: int, long_window: int, interval: str) -> str:
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if interval == '1wk':
        max_window = max(short_window, long_window)
        adjusted_date = start_date - timedelta(weeks=max_window)
    elif interval == '1d':
        max_window = max(short_window, long_window)
        adjusted_date = start_date - timedelta(days=max_window)
    elif interval == '5m':
        max_window = max(short_window, long_window) * 5  # Convert the window to minutes
        adjusted_date = start_date - timedelta(minutes=max_window)
    else:
        raise ValueError("Unsupported interval: " + interval)
    
    return adjusted_date.strftime('%Y-%m-%d')

# Function to download stock data
def download_stock_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval, threads=True, progress=False, actions=False)
    return data

# Function to apply the strategy
def apply_strategy(data: pd.DataFrame, short_window: int, long_window: int, short_ema: bool = True, long_ema: bool = True) -> pd.DataFrame:
    data = data.copy()
    if short_ema:
        data['Short MA'] = data['Adj Close'].ewm(span=short_window, adjust=False).mean()
    else:
        data['Short MA'] = data['Adj Close'].rolling(window=short_window, min_periods=1).mean()
        
    if long_ema:
        data['Long MA'] = data['Adj Close'].ewm(span=long_window, adjust=False).mean()
    else:
        data['Long MA'] = data['Adj Close'].rolling(window=long_window, min_periods=1).mean()

    data = data[data.index >= start_date]

    data['Signal'] = 0
    data['Buy'] = np.where((data['Adj Close'].shift(1) < data['Long MA'].shift(1)) & (data['Adj Close'] > data['Long MA']), data['Adj Close'], np.nan)
    data['Sell'] = np.where((data['Adj Close'].shift(1) > data['Long MA'].shift(1)) & (data['Adj Close'] < data['Long MA']), data['Adj Close'], np.nan)
    data['Position'] = np.where(data.index > data.index[1], np.where(data['Buy'].notna(), 1, np.where(data['Sell'].notna(), -1, np.nan)), np.nan)
    data['Position'].ffill(inplace=True)
    data['Position'].fillna(0, inplace=True)
    data['Strategy Return'] = data['Adj Close'].pct_change() * data['Position'].shift()
    data['Cumulative Return'] = (1 + data['Strategy Return']).cumprod().fillna(1)
    
    return data

# Function to calculate yearly returns
def calculate_yearly_returns(data: pd.DataFrame, short_window: int, long_window: int, short_ema: bool = True, long_ema: bool = True) -> pd.Series:
    data = apply_strategy(data, short_window, long_window, short_ema, long_ema)
    data['Year'] = data.index.year
    start_of_year_cumulative_returns = data.resample('Y').first()['Cumulative Return']
    end_of_year_cumulative_returns = data.resample('Y').last()['Cumulative Return']
    yearly_returns = (end_of_year_cumulative_returns / start_of_year_cumulative_returns - 1).fillna(0)
    return yearly_returns

# Function to optimize the strategy for each symbol
def optimize_strategy(data: pd.DataFrame, short_windows: range, long_windows: range, scenarios: list) -> tuple:
    best_return = -np.inf
    best_params = (0, 0, True, True)
    
    for short_window, long_window, (short_ema, long_ema) in itertools.product(short_windows, long_windows, scenarios):
        if short_window >= long_window:
            continue
        return_total = calculate_yearly_returns(data, short_window, long_window, short_ema, long_ema).mean()
        if return_total > best_return:
            best_return = return_total
            best_params = (short_window, long_window, short_ema, long_ema)
    
    return best_params, best_return

# Download data once for each symbol
symbol_data = {symbol: download_stock_data(symbol, start_date, end_date, interval) for symbol in symbols}

# Optimize strategy for each symbol
best_params_list = []
results = []

for symbol, data in symbol_data.items():
    best_params, best_return = optimize_strategy(data, short_windows, long_windows, scenarios)
    best_params_list.append((symbol, best_params, best_return))
    results.append((symbol, data))

    # Print average return for each symbol
    print(f'Symbol: {symbol}, Best Parameters: {best_params}, Average Yearly Return: {best_return:.4f}')

# Plotting all the results
fig, axs = plt.subplots(3, 2, figsize=(22, 18))

for i, (symbol, best_params, best_return) in enumerate(best_params_list):
    short_window, long_window, short_ema, long_ema = best_params
    data = results[i][1]
    strategy_data = apply_strategy(data, short_window, long_window, short_ema, long_ema)
    ax = axs[i // 2, i % 2]
    ax.plot(strategy_data.index, strategy_data['Adj Close'], label='Stock Price', color='black')
    ax.plot(strategy_data.index, strategy_data['Short MA'], label=f'{"EMA" if short_ema else "SMA"} (short window = {short_window})', color='blue', alpha=0.7)
    ax.plot(strategy_data.index, strategy_data['Long MA'], label=f'{"EMA" if long_ema else "SMA"} (long window = {long_window})', color='red', alpha=0.7)
    ax.plot(strategy_data.index, strategy_data['Buy'], '^', label='Buy Signal', color='green', alpha=1)
    ax.plot(strategy_data.index, strategy_data['Sell'], 'v', label='Sell Signal', color='red', alpha=1)
    ax.set_title(f'{symbol}: Best Parameters: Short Window = {short_window}, Long Window = {long_window}\nAverage Yearly Return: {best_return:.4f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Saving the results to an Excel file
with pd.ExcelWriter('symbol_strat_results.xlsx') as writer:
    for symbol, data in results:
        strategy_data = apply_strategy(data, *best_params_list[symbols.index(symbol)][1])
        strategy_data.to_excel(writer, sheet_name=f'{symbol}_Strategy Data')
