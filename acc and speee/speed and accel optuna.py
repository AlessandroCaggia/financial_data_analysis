import numpy as np
import pandas as pd
import optuna
import yfinance as yf
import matplotlib.pyplot as plt

# Options and Settings
pd.options.mode.chained_assignment = None  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Function Definitions
def calculate_speed_and_acceleration(data, speed_window):
    df = data.copy()
    df['Speed'] = df['Adj Close'].diff()
    df['Avg_Speed'] = df['Speed'].rolling(window=speed_window).mean()
    df['Acceleration'] = df['Speed'].diff()
    return df

def calculate_true_range_and_atr(data, atr_window):
    df = data.copy()
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift())
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift())
    df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=atr_window).mean()
    return df

def calculate_mean_params(params_list):
    
    # Initialize dictionary to store mean of parameters
    mean_params = {}
    for param in params_list[0].keys():
        mean_params[param] = np.mean([params[param] for params in params_list])
    return mean_params

def trading_strategy(data, params, calculate_return=True):
    data = calculate_true_range_and_atr(data, params['atr_window'])
    data = calculate_speed_and_acceleration(data, params['speed_window'])
    
    # Adjust buy and sell thresholds based on ATR
    adjusted_avg_speed_buy = params['avg_speed_buy'] * data['ATR']
    adjusted_avg_speed_sell = params['avg_speed_sell'] * data['ATR']
    
    buy_condition = (data['Avg_Speed'] > adjusted_avg_speed_buy) & \
                    (data['Speed'] > params['speed_buy']) & \
                    (data['Acceleration'] > params['acceleration_buy'])
    sell_condition = (data['Avg_Speed'] < adjusted_avg_speed_sell) & \
                     (data['Speed'] < params['speed_sell']) & \
                     (data['Acceleration'] < params['acceleration_sell'])

    data['Position'] = 0  
    data.loc[buy_condition, 'Position'] = 1
    data.loc[sell_condition, 'Position'] = -1
    data['Daily_Return'] = data['Adj Close'].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_Return']
    return data['Strategy_Return'].cumsum().iloc[-1] if calculate_return else data


def objective(trial, train_data, study, callback = None):
    params = {
        'avg_speed_buy': trial.suggest_float("avg_speed_buy", -100, 0),
        'avg_speed_sell': trial.suggest_float("avg_speed_sell", -0, 100),
        'speed_buy': trial.suggest_float("speed_buy", -100, 0),
        'speed_sell': trial.suggest_float("speed_sell", -0, 100),
        'acceleration_buy': trial.suggest_float("acceleration_buy", -100, 0),
        'acceleration_sell': trial.suggest_float("acceleration_sell", 0, 100),
        'atr_window': trial.suggest_int("atr_window", 3, 300),
        'speed_window': trial.suggest_int("speed_window", 3, 300)
    }
    result = trading_strategy(train_data, params)

    if result is None:
        raise optuna.exceptions.TrialPruned()  # Skip this trial
    return result

def record_best_trial(trial):
    best_iterations.append(trial.number)

def plot_results(results_df):
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(results_df))
    plt.bar(index, results_df['Strategy_Cumulative_Return'], bar_width, label='Strategy Cumulative Return')
    plt.bar(index + bar_width, results_df['Stock_Cumulative_Return'], bar_width, label='Stock (BAC) Cumulative Return')

    plt.xlabel('Test Period')
    plt.ylabel('Cumulative Return')
    plt.title('Strategy vs Stock Cumulative Return Over Test Periods')
    plt.xticks(index + bar_width / 2, results_df['Test_Period'], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_parameter_evolution(param_history):
    num_params = len(param_history[0])
    num_rows = int(np.ceil(num_params / 5))  # Calculate the number of rows needed
    fig, axs = plt.subplots(nrows=num_rows, ncols=5, figsize=(25, 5 * num_rows))
    fig.suptitle('Best Parameters Over Iterations')

    param_keys = list(param_history[0].keys())
    for idx, param in enumerate(param_keys):
        row = idx // 5
        col = idx % 5
        param_values = [iteration[param] for iteration in param_history]
        axs[row, col].plot(param_values, label=param)
        axs[row, col].set_title(f"Parameter: {param}")
        axs[row, col].set_xlabel("Iteration")
        axs[row, col].set_ylabel("Value")
        axs[row, col].legend()

    # Adjust layout for better display
    plt.tight_layout(pad=3.0)
    plt.show()

    
def main():
    global best_iterations
    best_iterations = []

    train_window = 4000
    test_window = 250
    analysis_start = pd.to_datetime('2010-06-03')
    start = analysis_start - pd.Timedelta(days=train_window)
    interval = '1d'
    data = yf.download('BAC', start=start, interval=interval)

    # Results DataFrame
    results_df = pd.DataFrame(columns=['Test_Period', 'Strategy_Cumulative_Return', 'Stock_Cumulative_Return', 'Best_Params'])

    param_history = []

    # Rolling Training and Testing
    for start in range(0, len(data) - train_window - test_window, test_window):
        train_data = data.iloc[start:start + train_window]
        test_data = data.iloc[start + train_window:start + train_window + test_window]

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_data,study), n_trials=900)

        param_history.append(study.best_params)
        
        # Check if there are any completed trials before accessing best_trial
        if study.trials_dataframe().loc[study.trials_dataframe()['state'] == 'COMPLETE'].empty:
            print("No completed trials yet.")
        else:
            best_trial = study.best_trial
            record_best_trial(best_trial)
            
        best_params = study.best_params
        

        strategy_result = trading_strategy(test_data, best_params, calculate_return=False)
        strategy_cumulative_return = strategy_result['Strategy_Return'].cumsum().iloc[-1]
        stock_cumulative_return = test_data['Adj Close'].pct_change().cumsum().iloc[-1]

        # DataFrame Update - Using pandas.concat instead of append
        test_period = f"Days {start + train_window} to {start + train_window + test_window}"
        new_row = pd.DataFrame([{
            'Test_Period': test_period,
            'Strategy_Cumulative_Return': strategy_cumulative_return,
            'Stock_Cumulative_Return': stock_cumulative_return,
            'Best_Params': best_params
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)


    # Display Results
    print(results_df)

    if best_iterations:
        print(f"best interaction: {best_iterations}")

    # Extracting and Calculating Mean for All Parameters
    mean_params = calculate_mean_params(results_df['Best_Params'].tolist())
    for param, value in mean_params.items():
        print(f"Mean {param}: {value}")

    # Additional Calculations
    mean_Strategy_Cumulative_Return = results_df['Strategy_Cumulative_Return'].mean()
    mean_Cumulative_Return = results_df['Stock_Cumulative_Return'].mean()
    net_performance = mean_Strategy_Cumulative_Return - mean_Cumulative_Return
    print(f"Mean Strategy Cumulative Return: {mean_Strategy_Cumulative_Return}")
    print(f"Net Performance: {net_performance}")
    # Plotting
    plot_results(results_df)
    plot_parameter_evolution(param_history)

if __name__ == "__main__":
    main()
