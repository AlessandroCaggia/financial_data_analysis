import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from sklearn.ensemble import RandomForestRegressor

# Options and Settings
pd.options.mode.chained_assignment = None  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def calculate_metrics(data, speed_window, atr_window):
    df = data.copy()
    df['Speed'] = df['Adj Close'].diff()
    df['Avg_Speed'] = df['Speed'].rolling(window=speed_window).mean()
    df['Acceleration'] = df['Speed'].diff()

    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift())
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift())
    df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=atr_window).mean()

    return df

def calculate_mean_params(params_list):
    return {param: np.mean([p[param] for p in params_list]) for param in params_list[0]}


#hyperparam tuning
hyperparameter_space = {
    'cxpb': (0.05, 0.99),  # Range for crossover probability
    'mutpb': (0.05, 0.8),  # Range for mutation probability
    'ngen': (3, 100)  # Range for number of generations
}

def random_hyperparameters(hyperparameter_space):
    return {
        'cxpb': np.random.uniform(*hyperparameter_space['cxpb']),
        'mutpb': np.random.uniform(*hyperparameter_space['mutpb']),
        'ngen': np.random.randint(*hyperparameter_space['ngen'])
    }

def random_search_hyperparameter_tuning(train_data, hyperparameter_space, n_iterations=3):
    best_score = float('-inf')
    best_hyperparams = None

    for _ in range(n_iterations):
        hyperparams = random_hyperparameters(hyperparameter_space)
        score = run_ga_with_hyperparams(train_data, **hyperparams)
        
        if score > best_score:
            best_score = score
            best_hyperparams = hyperparams

    return best_hyperparams

def run_ga_with_hyperparams(train_data, cxpb, mutpb, ngen):
    toolbox = setup_ga(train_data)
    population = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)
    # Assuming the first individual in the hall of fame is the best one
    best_individual = hof[0]
    score = evaluate_individual(best_individual, train_data)

    # Ensure the score is a single float value
    return score[0] if isinstance(score, tuple) else score


def trading_strategy(data, params, calculate_return=True):
    data = calculate_metrics(data, params['speed_window'], params['atr_window'])
    
    # Adjust buy and sell thresholds based on ATR
    #adjusted_avg_speed_buy = params['avg_speed_buy'] * data['ATR']
    #adjusted_avg_speed_sell = params['avg_speed_sell'] * data['ATR']

    buy_condition = (data['Avg_Speed'] > params['avg_speed_buy']) & \
                    (data['Speed'] > params['speed_buy']) & \
                    (data['Acceleration'] > params['acceleration_buy'])
                    
    sell_condition = (data['Avg_Speed'] < params['avg_speed_sell']) & \
                     (data['Speed'] < params['speed_sell']) & \
                     (data['Acceleration'] < params['acceleration_sell'])

    data['Position'] = 0  
    data.loc[buy_condition, 'Position'] = 1
    data.loc[sell_condition, 'Position'] = -1
    data['Daily_Return'] = data['Adj Close'].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_Return']
    # Add Buy and Sell Signal Columns
    data['Buy_Signal'] = buy_condition
    data['Sell_Signal'] = sell_condition

    return data['Strategy_Return'].cumsum().iloc[-1] if calculate_return else data

def evaluate_individual(individual, train_data):
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
    if individual[6] <= 0 or individual[7] <= 0:
        return (-999999999,)  # Assign a very low fitness score

    score = trading_strategy(train_data, params, calculate_return=True)
    return (score,)  # Wrap the score in a tuple

def setup_ga(train_data):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal, loc=0, scale=30)  # Example: mean=0, std-dev=10
    toolbox.register("attr_int", np.random.randint, 3, 300)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float,) * 6 + (toolbox.attr_int,) * 2, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, train_data=train_data)
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

    
def main():
    global best_iterations
    train_window = 4000
    test_window = 400
    analysis_start = pd.to_datetime('2013-06-03')
    start = analysis_start - pd.Timedelta(days=train_window)
    interval = '1d'
    data = yf.download('AMZN', start=start, interval=interval)

    results_df = pd.DataFrame(columns=['Test_Period', 'Strategy_Cumulative_Return', 'Stock_Cumulative_Return', 'Best_Params'])
    last_train_data = None
    last_test_data = None
    iteration = 0

    for start in range(0, len(data) - train_window - test_window, test_window):
        iteration += 1
        train_data = data.iloc[start:start + train_window]
        test_data = data.iloc[start + train_window:start + train_window + test_window]
        
        best_hyperparams = random_search_hyperparameter_tuning(train_data, hyperparameter_space)
        print("Best Hyperparameters:", best_hyperparams)

        toolbox = setup_ga(train_data)
        population = toolbox.population(n=20)
        hof = tools.HallOfFame(1)

        # Use the best hyperparameters
        algorithms.eaSimple(population, toolbox, cxpb=best_hyperparams['cxpb'], mutpb=best_hyperparams['mutpb'], ngen=best_hyperparams['ngen'], halloffame=hof, verbose=False)
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

        strategy_result = trading_strategy(test_data, best_params, calculate_return=False)
        strategy_cumulative_return = strategy_result['Strategy_Return'].cumsum().iloc[-1]
        stock_cumulative_return = test_data['Adj Close'].pct_change().cumsum().iloc[-1]

        test_period = f"Days {start + train_window} to {start + train_window + test_window}"
        new_row = {
            'Test_Period': test_period,
            'Strategy_Cumulative_Return': strategy_cumulative_return,
            'Stock_Cumulative_Return': stock_cumulative_return,
            'Best_Params': best_params
        }
        # New way using concat
        new_row_df = pd.DataFrame([new_row])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        # Apply trading_strategy to both train and test data
        processed_train_data = trading_strategy(train_data, best_params, calculate_return=False)
        processed_test_data = trading_strategy(test_data, best_params, calculate_return=False)

        # Update last train and test data with processed data
        last_train_data = processed_train_data
        last_test_data = processed_test_data
        

    # Before printing, remove specified columns
    columns_to_remove = ['Open', 'High', 'Low', 'Close', 'Volume', 
                         'High-Low', 'High-PrevClose', 'Low-PrevClose', 
                         'TrueRange', 'ATR']

    if last_train_data is not None and last_test_data is not None:
        # Dropping specified columns
        last_train_data = last_train_data.drop(columns=columns_to_remove, errors='ignore')
        last_test_data = last_test_data.drop(columns=columns_to_remove, errors='ignore')

        print("Last Train Data:")
        print(last_train_data)
        print("\nLast Test Data:")
        print(last_test_data)


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

    # Prepare the data
    X = ga_results.drop('score', axis=1)
    y = ga_results['score']

    # Train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature importances
    print("Feature importances:")
    for f in range(X.shape[1]):
        print(f"{X.columns[indices[f]]}: {importances[indices[f]]}")

if __name__ == "__main__":
    main()

    
