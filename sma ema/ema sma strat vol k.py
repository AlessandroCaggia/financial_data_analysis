import pandas as pd
import numpy as np
import yfinance as yf
import itertools
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to adjust the start date based on the interval and moving average windows
def adjust_start_date(start_date: str, k_short: int, k_long: int, interval: str) -> str:
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if interval == '1wk':
        max_window = max(k_short, k_long)
        adjusted_date = start_date - timedelta(weeks=max_window)
    elif interval == '1d':
        max_window = max(k_short, k_long)
        adjusted_date = start_date - timedelta(days=max_window)
    elif interval == '5m':
        max_window = max(k_short, k_long) * 5  # Convert the window to minutes
        adjusted_date = start_date - timedelta(minutes=max_window)
    else:
        raise ValueError("Unsupported interval: " + interval)
    
    return adjusted_date.strftime('%Y-%m-%d')

# Panel
symbol = '^SPX'
start_date = '2022-05-21'
end_date = '2024-05-23'
interval = '1wk'  # Change this to '5m', '1d', or '1wk'

# Define coefficient range and scenarios
coefficients = np.linspace(10, 1000, 20)  # Coefficients from 10 to 1000 in steps of 20

best_sharpe = -np.inf
best_params = (0, 0, True, True)  # Initialize with default EMA-EMA scenario

scenarios = [
    (True, True),  # EMA-EMA
    (False, False),  # SMA-SMA
    (True, False),  # EMA-SMA
    (False, True)  # SMA-EMA
]

# Function to download stock data
def download_stock_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval, threads=True, progress=False, actions=False)
    return data

# Calculate volatility as the standard deviation of log returns
def calculate_volatility(data: pd.DataFrame) -> float:
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    volatility = log_returns.std() * np.sqrt(len(log_returns.dropna()))
    return volatility

# Apply strategy with given coefficients and EMAs/SMAs
def apply_strategy(data: pd.DataFrame, k_short: float, k_long: float, short_ema: bool = True, long_ema: bool = True) -> pd.DataFrame:
    data = data.copy()
    volatility = calculate_volatility(data)
    short_window = int(k_short * volatility)
    long_window = int(k_long * volatility)
    
    adjusted_start_date = adjust_start_date(start_date, short_window, long_window, interval)
    
    if short_window < 2 or long_window < 2:
        return pd.DataFrame()  # Invalid window size

    if short_ema:
        data['Short MA'] = data['Adj Close'].ewm(span=short_window, adjust=False).mean()
    else:
        data['Short MA'] = data['Adj Close'].rolling(window=short_window, min_periods=1).mean()

    if long_ema:
        data['Long MA'] = data['Adj Close'].ewm(span=long_window, adjust=False).mean()
    else:
        data['Long MA'] = data['Adj Close'].rolling(window=long_window, min_periods=1).mean()

    data = data[data.index >= adjusted_start_date]

    data['Signal'] = 0
    data['Buy'] = np.where((data['Short MA'].shift(1) < data['Long MA'].shift(1)) & (data['Short MA'] > data['Long MA']), data['Adj Close'], np.nan)
    data['Sell'] = np.where((data['Short MA'].shift(1) > data['Long MA'].shift(1)) & (data['Short MA'] < data['Long MA']), data['Adj Close'], np.nan)
    data['Position'] = np.where(data.index > data.index[1], np.where(data['Buy'].notna(), 1, np.where(data['Sell'].notna(), -1, np.nan)), np.nan)
    data['Position'].ffill(inplace=True)
    data['Position'].fillna(0, inplace=True)
    data['Strategy Return'] = data['Adj Close'].pct_change() * data['Position'].shift()
    data['Cumulative Return'] = (1 + data['Strategy Return']).cumprod().fillna(1)

    return data

# Calculate Sharpe ratio
def calculate_sharpe_ratio(data: pd.DataFrame) -> float:
    if data.empty:
        return np.nan
    daily_return = data['Strategy Return'].mean()
    daily_volatility = data['Strategy Return'].std()
    sharpe_ratio = np.sqrt(252) * daily_return / daily_volatility  # Annualized Sharpe Ratio assuming 252 trading days
    return sharpe_ratio

# Calculate yearly returns
def calculate_yearly_returns(data: pd.DataFrame) -> pd.Series:
    data.index = pd.to_datetime(data.index)  # Ensure index is datetime
    data['Year'] = data.index.year
    start_of_year_cumulative_returns = data.resample('Y').first()['Cumulative Return']
    end_of_year_cumulative_returns = data.resample('Y').last()['Cumulative Return']
    yearly_returns = (end_of_year_cumulative_returns / start_of_year_cumulative_returns - 1).fillna(0)
    return yearly_returns

# Download initial data to calculate volatility
initial_data = download_stock_data(symbol, start_date, end_date, interval)
volatility = calculate_volatility(initial_data)

# Optimize coefficients
for k_short, k_long, (short_ema, long_ema) in itertools.product(coefficients, coefficients, scenarios):
    if k_short >= k_long:
        continue
    
    short_window = int(k_short * volatility)
    long_window = int(k_long * volatility)
    
    adjusted_start_date = adjust_start_date(start_date, short_window, long_window, interval)
    data = download_stock_data(symbol, adjusted_start_date, end_date, interval)
    
    strategy_data = apply_strategy(data, k_short, k_long, short_ema, long_ema)

    if strategy_data is None or strategy_data.empty:
        continue
    
    sharpe_ratio = calculate_sharpe_ratio(strategy_data)
    if sharpe_ratio > best_sharpe:
        best_sharpe = sharpe_ratio
        best_params = (k_short, k_long, short_ema, long_ema)

print('Best Parameters:', best_params)
print('Best Sharpe Ratio:', best_sharpe)

# Calculate strategy with best parameters
k_short, k_long, short_ema, long_ema = best_params
short_window = int(k_short * volatility)
long_window = int(k_long * volatility)
adjusted_start_date = adjust_start_date(start_date, short_window, long_window, interval)
data = download_stock_data(symbol, adjusted_start_date, end_date, interval)
strategy_data = apply_strategy(data, k_short, k_long, short_ema, long_ema)
yearly_returns = calculate_yearly_returns(strategy_data)
print(yearly_returns)

# Calculate and print the real period values used
print(f'Period Values Used: Short Window = {short_window}, Long Window = {long_window}')

# Handle gaps in 5-minute interval data for plotting and remove weekends
if interval == '5m':
    strategy_data = strategy_data.between_time('09:25', '16:00')
    strategy_data = strategy_data[strategy_data.index.dayofweek < 5]
    strategy_data = strategy_data[strategy_data.index.strftime('%H:%M:%S') != '00:00:00']
    strategy_data = strategy_data[strategy_data.index.time != pd.Timestamp("00:00:00").time()]
    strategy_data.index = strategy_data.index.tz_localize(None)


# Plotting the stock price, short MA, long MA, and crossing points
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(strategy_data.index, strategy_data['Adj Close'], label='Stock Price', color='black')
ax.plot(strategy_data.index, strategy_data['Short MA'], label=f'{"EMA" if short_ema else "SMA"} (short window = {short_window})', color='blue', alpha=0.7)
ax.plot(strategy_data.index, strategy_data['Long MA'], label=f'{"EMA" if long_ema else "SMA"} (long window = {long_window})', color='red', alpha=0.7)
ax.plot(strategy_data.index, strategy_data['Buy'], '^', label='Buy Signal', color='green', alpha=1)
ax.plot(strategy_data.index, strategy_data['Sell'], 'v', label='Sell Signal', color='red', alpha=1)

ax.set_title(f'Stock Price, Short MA, Long MA, and Crossing Points\nBest Parameters: k_short = {k_short:.2f}, k_long = {k_long:.2f}')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)

plt.show()

# Ensure the plot can display negative values
plt.figure(figsize=(12, 6))
yearly_returns.plot(kind='bar', color='skyblue')
plt.title(f'Yearly Returns for {"EMA" if short_ema else "SMA"}-{"EMA" if long_ema else "SMA"} Strategy\nBest Parameters: k_short = {k_short:.2f}, k_long = {k_long:.2f}')
plt.xlabel('Year')
plt.ylabel('Return')
plt.ylim(yearly_returns.min() - 0.1, yearly_returns.max() + 0.1)
plt.axhline(0, color='red', linewidth=1)
plt.show()

# Save the strategy data to an Excel file
output_file = 'strat_sma_ema_results.xlsx'
with pd.ExcelWriter(output_file) as writer:
    strategy_data.to_excel(writer, sheet_name='Strategy Data')
