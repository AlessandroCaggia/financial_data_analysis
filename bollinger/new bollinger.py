import yfinance as yf
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from optuna.pruners import MedianPruner
from optuna.pruners import HyperbandPruner
from datetime import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as ticker
from ta.trend import PSARIndicator

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


interval = '1d' #1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
today = pd.Timestamp(datetime.today().date())

start = pd.to_datetime('2023-11-5')
end_date = pd.to_datetime('2023-11-24')
days = 2

trading = 'n'
n_trials = 70

tickers = ['BAC']

# Function to apply log transformation and first-order differencing
def transform_data(data):
    data['Adj Close'] = np.log(data['Adj Close'])
    data['Adj Close'] = data['Adj Close'].diff()
    return data.dropna()

# Function to revert log transformation and first-order differencing
def inverse_transform_data(data, original_data):
    data['Adj Close'] = data['Adj Close'].cumsum()
    data['Adj Close'] += np.log(original_data['Adj Close'].iloc[0])
    data['Adj Close'] = np.exp(data['Adj Close'])
    return data

def calculate_sharpe_ratio(returns, risk_free_rate=3):
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    if sharpe_ratio < 0 or np.isnan(sharpe_ratio):
        return 0
    return sharpe_ratio

def download_data(tickers):
    data = yf.download(tickers, start=start, interval = interval)
    data.dropna(inplace=True)
    columns_to_drop = ['Open', 'Close', 'Volume']
    data = data.drop(columns=columns_to_drop, axis=1)
    return transform_data(data)

def calculate_bollinger_bands(data, window, upper_std_multiplier, lower_std_multiplier):
    ema = data['Adj Close'].ewm(span=window, adjust=False).mean()
    rstd = data['Adj Close'].rolling(window=window).std()  # Standard deviation based on rolling window
    upper_band = ema + rstd * upper_std_multiplier
    lower_band = ema - rstd * lower_std_multiplier
    return upper_band, lower_band

def calculate_cumulative_returns(strategy_returns):
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    return cumulative_returns.iloc[-1]

def calculate_max_drawdown(returns):
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    # Find the minimum value of the cumulative returns
    min_cumulative_return = cumulative_returns.min()
    max_drawdown = min_cumulative_return - 1
    return max_drawdown

def calculate_sortino_ratio(returns, target_return=0):
    # Calculate mean return and downside deviation
    mean_return = returns.mean()
    negative_returns = [min(0, r - target_return) for r in returns]
    downside_deviation = np.std(negative_returns, ddof=1)
    # Check if downside deviation is zero (to avoid division by zero)
    if downside_deviation == 0:
        return 0
    # Calculate Sortino Ratio
    sortino_ratio = (mean_return - target_return) / downside_deviation
    if sortino_ratio < 0 or np.isnan(sortino_ratio) or np.isinf(sortino_ratio):
        return 0
    return sortino_ratio


def objective(trial, data, window_size, step_size):
    total_sortino_ratio = 0
    n_splits = (len(data) - window_size) // step_size
    step = trial.suggest_float('step', 0.01, 0.1)
    max_step = trial.suggest_float('max_step', 0.1, 0.5)
    data['Market_Returns'] = data['Adj Close'].pct_change()
    
    for i in range(n_splits):
        train_data = data.iloc[i * step_size : i * step_size + window_size]
        test_data = data.iloc[i * step_size + window_size : i * step_size + window_size + step_size]

        window = trial.suggest_int('window', 3, min(window_size - 1, 110))
        upper_std_multiplier = trial.suggest_float('upper_std_multiplier', 0.5, 3.0)
        lower_std_multiplier = trial.suggest_float('lower_std_multiplier', 0.5, 3.0)

        # Apply Bollinger Bands and Parabolic SAR
        upper_band, lower_band = calculate_bollinger_bands(train_data, window, upper_std_multiplier, lower_std_multiplier)
        psar = PSARIndicator(high=np.exp(train_data['High']), low=np.exp(train_data['Low']), close=np.exp(train_data['Adj Close']), step=step, max_step=max_step)
        train_data['SAR'] = np.log(psar.psar())

        # Apply to test data
        test_data['Upper_Band'] = upper_band.reindex(test_data.index, method='ffill')
        test_data['Lower_Band'] = lower_band.reindex(test_data.index, method='ffill')
        psar_test = PSARIndicator(high=np.exp(test_data['High']), low=np.exp(test_data['Low']), close=np.exp(test_data['Adj Close']), step=step, max_step=max_step)
        test_data['SAR'] = np.log(psar_test.psar())

        # Signal generation
        test_data['SAR_diff'] = (test_data['SAR'].diff())
        test_data['Buy_Signal'] = (test_data['Adj Close'] < test_data['Lower_Band']) & (test_data['SAR_diff'] > 0)
        test_data['Sell_Signal'] = (test_data['Adj Close'] > test_data['Upper_Band']) & (test_data['SAR_diff'] < 0)

        # Positions
        test_data['Position'] = 0
        test_data['Position'] = np.where(test_data['Buy_Signal'] & (test_data['Position'].shift(1) == 0), 1, np.nan)
        test_data['Position'] = np.where(test_data['Sell_Signal'], 0, test_data['Position'])
        test_data['Position'].ffill(inplace=True)
        test_data['Position'].fillna(0, inplace=True)
        test_data['Strategy_Returns'] = test_data['Market_Returns'] * test_data['Position'].shift(1)
        sortino_ratio = calculate_sortino_ratio(test_data['Strategy_Returns'])
        test_data.dropna(inplace=True)

        if not np.isnan(sortino_ratio):
            total_sortino_ratio += sortino_ratio

    average_sortino_ratio = total_sortino_ratio / n_splits if n_splits > 0 else -np.inf
    return average_sortino_ratio

    
def plot_bollinger_bands(data, window, upper_std_multiplier, lower_std_multiplier):
    # Filter the data for the last year
    if interval == '1d':
        last_year = pd.Timestamp(datetime.now() - pd.Timedelta(days=365))
        recent_data = data[data.index >= last_year]

        # Calculate Bollinger Bands for the filtered data
        upper_band, lower_band = calculate_bollinger_bands(recent_data, window, upper_std_multiplier, lower_std_multiplier)

        # Plotting
        plt.figure(figsize=(14, 8))
        plt.plot(recent_data['Adj Close'], label='Close Price', color='blue')
        plt.plot(upper_band, label='Upper Bollinger Band', color='red')
        plt.plot(lower_band, label='Lower Bollinger Band', color='green')
        plt.title('Bollinger Bands - Last Year')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        

    if interval == '1m' or interval == '5m':
        last_two_weeks = pd.Timestamp(datetime.now() - pd.Timedelta(days=7))
        recent_data = data[data.index >= last_two_weeks]

        # Calculate Bollinger Bands for the filtered data
        upper_band, lower_band = calculate_bollinger_bands(recent_data, window, upper_std_multiplier, lower_std_multiplier)

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(recent_data))  # Create an evenly spaced x-axis

        ax.plot(x, recent_data['Adj Close'], label='Close Price', color='blue')
        ax.plot(x, upper_band.values, label='Upper Bollinger Band', color='red')
        ax.plot(x, lower_band.values, label='Lower Bollinger Band', color='green')

        # Define the date format
        def format_date(index, pos):
            if index < 0 or index >= len(recent_data.index):
                return ''
            return recent_data.index[int(index)].strftime('%Y-%m-%d %H:%M')

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_date))

        # Set title and labels with adjusted interval
        ax.set_title(f'Bollinger Bands - Last Two Weeks 1 Minute Interval')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Price')
        ax.legend()

        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

def make_decision(last_day_data):
    if last_day_data['Buy_Signal'].iloc[0]:
        return "Buy"
    elif last_day_data['Sell_Signal'].iloc[0]:
        return "Sell"
    else:
        return "-"

def apply_strategy(data, window, upper_std_multiplier, lower_std_multiplier, step, max_step, original_data):
    data = inverse_transform_data(data, original_data)

    # Calculate Bollinger Bands and Parabolic SAR
    upper_band, lower_band = calculate_bollinger_bands(data, window, upper_std_multiplier, lower_std_multiplier)
    psar_indicator = PSARIndicator(data['High'], data['Low'], data['Adj Close'], step=step, max_step=max_step)
    
    data['Upper_Band'] = upper_band
    data['Lower_Band'] = lower_band
    data['SAR'] = psar_indicator.psar()
    
    # Calculate the rate of change in SAR
    data['SAR_diff'] = data['SAR'].diff()

    # Generate signals based on Bollinger Bands and SAR changes
    data['Buy_Signal'] = (data['Adj Close'] < data['Lower_Band']) & (data['SAR_diff'] < 0)
    data['Sell_Signal'] = (data['Adj Close'] > data['Upper_Band']) & (data['SAR_diff'] > 0)

    # Define the position
    data['Position'] = 0
    data['Position'] = np.where(data['Buy_Signal'] & (data['Position'].shift(1) == 0), 1, np.nan)
    data['Position'] = np.where(data['Sell_Signal'], 0, data['Position'])
    data['Position'].ffill(inplace=True)
    data['Position'].fillna(0, inplace=True)

    # Calculate strategy returns
    data['Market_Returns'] = data['Adj Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
    data['Cum_Strategy_Returns'] = data['Strategy_Returns'].cumsum()
    data.dropna(inplace=True)

    return data



def main():
    cutoff = end_date - pd.Timedelta(days=days)
    
    # Download data
    original_data = yf.download(tickers, start=start, interval=interval)
    original_data.dropna(inplace=True)
    
    # Apply transformations to data
    df = download_data(tickers)
    data = df[df.index <= cutoff]
    
    # Set up the optimization study
    study = optuna.create_study(direction='maximize')
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, data, window_size=252, step_size=20), n_trials=n_trials)
    
    # Get best parameters from the optimization study
    best_params = study.best_params
    
    # Apply the strategy with the best parameters found
    best_window = best_params['window']
    best_upper_std_multiplier = best_params['upper_std_multiplier']
    best_lower_std_multiplier = best_params['lower_std_multiplier']
    best_step = best_params['step']
    best_max_step = best_params['max_step']
    
    full_data_with_strategy = apply_strategy(data, best_window, best_upper_std_multiplier, best_lower_std_multiplier, best_step, best_max_step, original_data)
    
    # Apply the strategy on the recent data segment
    recent_data = df[(df.index > cutoff)]
    applied_data = apply_strategy(recent_data, best_window, best_upper_std_multiplier, best_lower_std_multiplier, best_step, best_max_step, original_data)    
    # Calculate and display performance metrics
    applied_metrics = {
        'Sharpe Ratio': calculate_sharpe_ratio(applied_data['Strategy_Returns']),
        'Cumulative Return': calculate_cumulative_returns(applied_data['Strategy_Returns']),
        'Max Drawdown': calculate_max_drawdown(applied_data['Strategy_Returns'])
    }
    print(applied_data)
    print("Applied Data Metrics:", applied_metrics)
    
    # Plot Bollinger Bands with SAR
    plot_bollinger_bands(original_data, best_window, best_upper_std_multiplier, best_lower_std_multiplier)
    

if __name__ == "__main__":
    main()


if trading == 'y':
        df = download_data(tickers, frequency = frequency)
        new_study = optuna.create_study(direction='maximize', pruner=pruner)
        new_study.optimize(lambda trial: objective(trial, df, window_size, step_size), n_trials= n_trials)
        best_params = new_study.best_params
        best_trial = new_study.best_trial
        print(f"Best Parameters: {best_params}")
        # Re-run the strategy on the entire dataset with the best parameters
        best_window = best_params['window']
        best_upper_std_multiplier = best_params['upper_std_multiplier']
        best_lower_std_multiplier = best_params['lower_std_multiplier']
        full_data_with_strategy = apply_strategy(df, best_window, best_upper_std_multiplier, best_lower_std_multiplier, original_data)   
        # Extract the last day's data
        last_day_data = full_data_with_strategy.tail(1)
        # Make the decision
        decision = make_decision(last_day_data)
        print(f"Today's Decision: {decision}")
