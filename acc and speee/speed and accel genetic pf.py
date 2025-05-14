import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import numpy as np

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

#use parameters dumbass

def trading_strategy(data, params, calculate_return=True):
    data = calculate_true_range_and_atr(data, params['atr_window'])
    data = calculate_speed_and_acceleration(data, params['speed_window'])
    
    # Adjust buy and sell thresholds based on ATR
    adjusted_avg_speed_buy = params['avg_speed_buy'] * data['ATR']
    adjusted_avg_speed_sell = params['avg_speed_sell'] * data['ATR']

    
    buy_condition = (data['Avg_Speed'] > params['avg_speed_buy']) & \
                    (data['Avg_Speed'] > adjusted_avg_speed_buy) & \
                    (data['Speed'] > params['speed_buy']) & \
                    (data['Acceleration'] > params['acceleration_buy'])
    sell_condition = (data['Avg_Speed'] < adjusted_avg_speed_sell) & \
                     (data['Avg_Speed'] < params['avg_speed_sell']) & \
                     (data['Speed'] < params['speed_sell']) & \
                     (data['Acceleration'] < params['acceleration_sell'])

    data['Buy_Signal'] = buy_condition
    data['Sell_Signal'] = sell_condition
    data['Position'] = 0  
    data.loc[buy_condition, 'Position'] = 1
    data.loc[sell_condition, 'Position'] = -1
    data['Daily_Return'] = data['Adj Close'].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_Return']
    return data['Strategy_Return'].cumsum().iloc[-1] if calculate_return else data

def evaluate_portfolio(individual, stock_data_dict):
    total_return = 0
    for stock, data in stock_data_dict.items():
        params = {
            'avg_speed_buy': individual[0],
            'avg_speed_sell': individual[1],
            'speed_buy': individual[2],
            'speed_sell': individual[3],
            'acceleration_buy': individual[4],
            'acceleration_sell': individual[5],
            'atr_window': int(individual[6]),
            'speed_window': int(individual[7])
        }
        total_return += trading_strategy(data, params, calculate_return=True)
    avg_return = total_return / len(stock_data_dict)
    return avg_return,


def setup_ga(train_data):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -100, 100)
    toolbox.register("attr_int", np.random.randint, 3, 50)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float,) * 6 + (toolbox.attr_int,) * 2, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_portfolio, train_data=train_data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


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
    
    train_window = 4000
    test_window = 250
    analysis_start = pd.to_datetime('2014-03-03')
    start = analysis_start - pd.Timedelta(days=train_window)
    interval = '1d'
    stock_symbols = ['AAPL', 'MSFT']
    results_df = pd.DataFrame(columns=['Test_Period', 'Strategy_Cumulative_Return', 'Stock_Cumulative_Return', 'Best_Params'])
    stock_data_dict = {symbol: yf.download(symbol, start=start, interval=interval) for symbol in stock_symbols}

    for start in range(0, len(list(stock_data_dict.values())[0]) - train_window - test_window, test_window):
        train_data_dict = {symbol: data.iloc[start:start + train_window] for symbol, data in stock_data_dict.items()}
        test_data_dict = {symbol: data.iloc[start + train_window :start + train_window + test_window] for symbol, data in stock_data_dict.items()}

        toolbox = setup_ga(list(train_data_dict.values())[0])
        toolbox.register("evaluate", evaluate_portfolio, stock_data_dict=train_data_dict)
        
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.1, ngen=7, halloffame=hof, verbose=False)
        best_params = {
            'avg_speed_buy': hof[0][0],
            'avg_speed_sell': hof[0][1],
            'speed_buy': hof[0][2],
            'speed_sell': hof[0][3],
            'acceleration_buy': hof[0][4],
            'acceleration_sell': hof[0][5],
            'atr_window': int(hof[0][6]),
            'speed_window': int(hof[0][7])
        }
        
        print(best_params)
        for symbol, test_data in test_data_dict.items():
            strategy_result = trading_strategy(test_data, best_params, calculate_return=False)

            # Check if Strategy_Return column has at least one entry
            if not strategy_result['Strategy_Return'].empty:
                strategy_cumulative_return = strategy_result['Strategy_Return'].cumsum().iloc[-1]
            else:
                strategy_cumulative_return = 0  # Or any other default value you deem appropriate

            stock_cumulative_return = test_data['Adj Close'].pct_change().cumsum().iloc[-1] if not test_data['Adj Close'].empty else 0

            test_period = f"Days {start + train_window} to {start + train_window + test_window}"
            new_row = {
                'Stock': symbol,
                'Test_Period': test_period,
                'Strategy_Cumulative_Return': strategy_cumulative_return,
                'Stock_Cumulative_Return': stock_cumulative_return,
                'Best_Params': best_params
            }
            new_row_df = pd.DataFrame([new_row])
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

            train_strategy_result = trading_strategy(train_data_dict[symbol], best_params, calculate_return=False)
            test_strategy_result = trading_strategy(test_data_dict[symbol], best_params, calculate_return=False)

            # Define columns to exclude
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'High-Low', 'High-PrevClose', 'Low-PrevClose', 'TrueRange']

            # Filtering out the unwanted columns
            filtered_train_result = train_strategy_result.drop(columns=exclude_columns)
            filtered_test_result = test_strategy_result.drop(columns=exclude_columns)

            # Printing the last few rows of train and test DataFrames
            print(f"\n{symbol} - Last entries in Train Data with Buy/Sell Signals:")
            print(filtered_train_result)
            print(f"\n{symbol} - Last entries in Test Data with Buy/Sell Signals:")
            print(filtered_test_result)





    print(results_df)


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

if __name__ == "__main__":
    main()
