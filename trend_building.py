import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
import random

warnings.filterwarnings("ignore")

def download_symbols(all_symbols_list, start_date_year1, end_date_year2, interval, failed_symbols_list, data_dict, valid_symbols_list):
    if all_symbols_list:
        for symbol in all_symbols_list:
            data = yf.download(symbol, start=start_date_year1, end=end_date_year2, interval=interval, threads=True, progress=False, actions=False)
            data = data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
            data['daily_returns'] = data['Adj Close'].pct_change()
            data = data.dropna(subset=['daily_returns'], how='all')

            if interval == '1wk':
                if data.empty:
                    failed_symbols_list.add(symbol)
                else:
                    data_dict[symbol] = data.assign(daily_returns=np.log(data['Adj Close'] / data['Adj Close'].shift(1))).dropna()
                    valid_symbols_list.append(symbol)
            else:
                data_dict[symbol] = data.assign(daily_returns=np.log(data['Adj Close'] / data['Adj Close'].shift(1))).dropna()
                valid_symbols_list.append(symbol)

        if failed_symbols_list:
            pd.DataFrame(list(failed_symbols_list), columns=['Failed Symbols']).to_csv('failed_symbols.csv', index=False)

def calc_trend_t_stat(prices):
    n = len(prices)
    X = np.arange(n)
    X = add_constant(X)
    y = prices.values
    model = OLS(y, X).fit()
    t_stat = model.tvalues[1]  # t-statistic for the slope coefficient
    beta = model.params[1]  # slope coefficient (trend)
    return t_stat, beta

def get_trend_signal(t_stat, beta):
 
    if t_stat > 1:
            return 1
    elif t_stat < -1:
            return -1
    else:
        return 0

# Define the parameters
symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA',     # Technology
    'JPM', 'BAC',                        # Financial
    'JNJ', 'PFE',                        # Healthcare
    'XOM', 'CVX',                        # Energy
    'PG', 'KO',                          # Consumer Staples
    'DIS', 'AMZN',                       # Consumer Discretionary
    'BA', 'LMT',                         # Industrial
    'GLD',                               # Gold (SPDR Gold Shares ETF)
    'USO',                               # Oil (United States Oil Fund)
    'VNQ',                               # Real Estate (Vanguard Real Estate ETF)
    'TLT',                               # Bonds (iShares 20+ Year Treasury Bond ETF)
    'BTC-USD',                           # Bitcoin
]
start_date = '2000-01-01'
end_date = '2024-01-01'
interval = '1wk'

# Initialize lists and dictionaries
failed_symbols = set()
data_dict = {}
valid_symbols = []

# Download the data
download_symbols(symbols, start_date, end_date, interval, failed_symbols, data_dict, valid_symbols)

# Prepare a DataFrame to hold strategy returns and signals
strategy_returns = pd.DataFrame(index=data_dict[symbols[0]].index)
signals = pd.DataFrame(index=data_dict[symbols[0]].index)

# Initialize lists to store t-statistics, betas, and their associated returns for long and short positions
long_t_stats_with_returns = []
long_betas_with_returns = []
short_t_stats_with_returns = []
short_betas_with_returns = []

# Process each valid symbol and generate strategy returns
for symbol in valid_symbols:
    data = data_dict[symbol]
    
    # Calculate the 12-month return
    data['Return_12m'] = data['Adj Close'].pct_change(periods=252)
    data['Volatility'] = data['daily_returns'].rolling(window=252).std()
    data['Adjusted_Return'] = data['daily_returns'] / data['Volatility']
    
    # Calculate SIGN
    data['SIGN'] = np.where(data['Return_12m'] >= 0, 1, -1)
    
    # Calculate TREND and BETA separately
    data['TREND'] = data['Adj Close'].rolling(window=252).apply(lambda x: calc_trend_t_stat(x)[0], raw=False)
    data['BETA'] = data['Adj Close'].rolling(window=252).apply(lambda x: calc_trend_t_stat(x)[1], raw=False)
    
    # Generate trading signals
    data['TREND_Signal'] = data.apply(lambda row: get_trend_signal(row['TREND'], row['BETA']), axis=1)
    
    # Combine SIGN and TREND signals
    data['Signal'] = data['SIGN'] * data['TREND_Signal']
    
    # Calculate strategy returns
    data['Strategy_Return'] = data['Signal'].shift(1) * data['daily_returns']
    
    # Store the signals and returns for later use
    signals[symbol] = data['Signal']
    strategy_returns[symbol] = data['Strategy_Return']
    
    # Store t-statistics, betas, and their associated returns for long and short positions separately
    for i in range(len(data)):
        if not np.isnan(data['TREND'].iloc[i]) and not np.isnan(data['Strategy_Return'].iloc[i]):
            if data['Signal'].iloc[i] > 0:
                long_t_stats_with_returns.append((data['TREND'].iloc[i], data['Strategy_Return'].iloc[i]))
                long_betas_with_returns.append((data['BETA'].iloc[i], data['Strategy_Return'].iloc[i]))
            elif data['Signal'].iloc[i] < 0:
                short_t_stats_with_returns.append((data['TREND'].iloc[i], data['Strategy_Return'].iloc[i]))
                short_betas_with_returns.append((data['BETA'].iloc[i], data['Strategy_Return'].iloc[i]))

# Set the seed for reproducibility
seed = 42
random.seed(seed)

# Adjust portfolio returns to include only top 5 stocks by signal strength
strategy_returns['Portfolio_Return'] = 0

for date in strategy_returns.index:
    # Get symbols with a non-zero signal
    eligible_symbols = signals.loc[date][signals.loc[date] != 0].index
    """if len(eligible_symbols) >= 5:
        selected_symbols = signals.loc[date].nlargest(5).index"""
    # Calculate the portfolio return as the average return of the selected symbols
    strategy_returns.at[date, 'Portfolio_Return'] = strategy_returns.loc[date, eligible_symbols].mean()

# Download S&P 500 data
sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval=interval)
sp500['SP500_Return'] = sp500['Adj Close'].pct_change()
sp500 = sp500.dropna(subset=['SP500_Return'])

# Align strategy returns with SP500 returns and signals
combined_returns = strategy_returns[['Portfolio_Return']].merge(sp500[['SP500_Return']], left_index=True, right_index=True, how='inner')
combined_signals = signals.merge(sp500[['SP500_Return']], left_index=True, right_index=True, how='inner')

# Calculate monthly aggregate returns
monthly_returns = combined_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

# Calculate annual returns
annual_returns = combined_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)

# Calculate 10-year average returns
last_10_years = combined_returns.loc[combined_returns.index >= '2014-01-01']
avg_10yr_strategy_return = (1 + last_10_years['Portfolio_Return']).prod() ** (1 / 10) - 1
avg_10yr_sp500_return = (1 + last_10_years['SP500_Return']).prod() ** (1 / 10) - 1

# Print 10-year average returns
print(f"10-Year Average Return (Strategy): {avg_10yr_strategy_return:.2%}")
print(f"10-Year Average Return (S&P 500): {avg_10yr_sp500_return:.2%}")

# Plot the annual returns in a bar chart
annual_returns.plot(kind='bar', figsize=(14, 8))
plt.title('Annual Returns Comparison: Trend-Following Strategy vs S&P 500')
plt.xlabel('Year')
plt.ylabel('Annual Return')
plt.legend(['Trend-Following Strategy', 'S&P 500'])
plt.show()

# Convert t-stats and betas with returns to DataFrames for plotting
long_t_stats_df = pd.DataFrame(long_t_stats_with_returns, columns=['T-Statistic', 'Return'])
long_betas_df = pd.DataFrame(long_betas_with_returns, columns=['Beta', 'Return'])
short_t_stats_df = pd.DataFrame(short_t_stats_with_returns, columns=['T-Statistic', 'Return'])
short_betas_df = pd.DataFrame(short_betas_with_returns, columns=['Beta', 'Return'])

# Define bins for grouping
t_stat_bins = np.linspace(-5, 5, 21)  # 20 bins for t-stat from -5 to 5
beta_bins = np.linspace(-0.2, 0.2, 21)  # 20 bins for beta from -0.2 to 0.2

# Bin the data
long_t_stats_df['T-Stat Bin'] = pd.cut(long_t_stats_df['T-Statistic'], bins=t_stat_bins)
long_betas_df['Beta Bin'] = pd.cut(long_betas_df['Beta'], bins=beta_bins)
short_t_stats_df['T-Stat Bin'] = pd.cut(short_t_stats_df['T-Statistic'], bins=t_stat_bins)
short_betas_df['Beta Bin'] = pd.cut(short_betas_df['Beta'], bins=beta_bins)

# Calculate the average return for each bin
avg_return_by_long_t_stat_bin = long_t_stats_df.groupby('T-Stat Bin')['Return'].mean()
avg_return_by_long_beta_bin = long_betas_df.groupby('Beta Bin')['Return'].mean()
avg_return_by_short_t_stat_bin = short_t_stats_df.groupby('T-Stat Bin')['Return'].mean()
avg_return_by_short_beta_bin = short_betas_df.groupby('Beta Bin')['Return'].mean()

# Plot the average returns by T-Statistic bins for long positions
plt.figure(figsize=(12, 6))
avg_return_by_long_t_stat_bin.plot(kind='bar')
plt.title('Average Return by T-Statistic Bin (Long Positions)')
plt.xlabel('T-Statistic Bin')
plt.ylabel('Average Return')
plt.show()

# Plot the average returns by Beta bins for long positions
plt.figure(figsize=(12, 6))
avg_return_by_long_beta_bin.plot(kind='bar')
plt.title('Average Return by Beta Bin (Long Positions)')
plt.xlabel('Beta Bin')
plt.ylabel('Average Return')
plt.show()

# Plot the average returns by T-Statistic bins for short positions
plt.figure(figsize=(12, 6))
avg_return_by_short_t_stat_bin.plot(kind='bar')
plt.title('Average Return by T-Statistic Bin (Short Positions)')
plt.xlabel('T-Statistic Bin')
plt.ylabel('Average Return')
plt.show()

# Plot the average returns by Beta bins for short positions
plt.figure(figsize=(12, 6))
avg_return_by_short_beta_bin.plot(kind='bar')
plt.title('Average Return by Beta Bin (Short Positions)')
plt.xlabel('Beta Bin')
plt.ylabel('Average Return')
plt.show()

# Output the summary statistics
long_t_stat_summary = long_t_stats_df.groupby('T-Stat Bin')['Return'].describe()
long_beta_summary = long_betas_df.groupby('Beta Bin')['Return'].describe()
short_t_stat_summary = short_t_stats_df.groupby('T-Stat Bin')['Return'].describe()
short_beta_summary = short_betas_df.groupby('Beta Bin')['Return'].describe()

