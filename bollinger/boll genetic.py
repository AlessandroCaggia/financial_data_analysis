
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
import yfinance as yf
import random

pd.options.mode.chained_assignment = None  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Download stock data
start_date = "2010-01-01"
today = "2019-01-01"

train_size = 180
test_size = 30
max_window_size = train_size    #must be smaller ,than train
ngen = 5
min_window = 3
min_k = 0.2
max_k = 3
cxpb = 0.7
mutpb = 0.2

def bollinger_bands(data, mean_window, std_window, k1, k2, previous_data):
    # If previous data is provided, prepend it to the current data

    if previous_data is not None:
        extended_data = pd.concat([previous_data, data])
    else:
        extended_data = data.copy()

    # Calculate rolling mean and standard deviation
    rolling_mean = extended_data['Adj Close'].rolling(window=mean_window).mean()
    rolling_std = extended_data['Adj Close'].rolling(window=std_window).std()

    # Calculate upper and lower bands
    upper_band = rolling_mean + (rolling_std * k1)
    lower_band = rolling_mean - (rolling_std * k2)

    # Adjust bands to align with original data
    upper_band = upper_band.iloc[len(previous_data):]
    lower_band = lower_band.iloc[len(previous_data):]

    return upper_band, lower_band

def rolling_windows(data, train_size, test_size, max_window_size):
    step = train_size + test_size
    start_index = max_window_size
    for i in range(start_index, len(data) - step, step):

        train_set = data.iloc[i:i+train_size]
        test_set = data.iloc[i+train_size:i+step]

        yield train_set, test_set


def fitness(individual, data, extension_size):
    buy_executed = False
    mean_window, std_window, k1, k2 = individual
    mean_window = max(1, int(round(mean_window)))
    std_window = max(1, int(round(std_window)))
    upper_band, lower_band = bollinger_bands(data, mean_window, std_window, k1, k2, previous_data)

    buy_signals = (data['Adj Close'] < lower_band) 
    sell_signals = (data['Adj Close'] > upper_band) 
        
    shares_held = 0
    cumulative_returns = 0
    returns = 0
    shares_held = 0
    nominal_value = 0
       
    for i in range(len(data)):
        next_open_price = data['Adj Close'].iloc[i + 1] if i + 1 < len(data) else data['Adj Close'].iloc[i]

        # Buy logic
        if buy_signals.iloc[i] and not buy_executed:
            shares_held += 1
            nominal_value += next_open_price
            buy_executed = True  # Set flag to true after buy

        # Update market value 
        market_value = shares_held * next_open_price
        
        if nominal_value != 0:
            # Sell signal detected, sell all shares at next day's opening price
            returns = (market_value - nominal_value) / nominal_value

            
        # Sell logic
        if sell_signals.iloc[i] and shares_held > 0:
            # After selling, reset 
            shares_held = 0
            nominal_value = 0
            market_value = 0
            cumulative_returns += returns
            buy_executed = False  # Reset flag after sell

        if i == len(data) - 1:
            cumulative_returns += returns
        
    fitness_value = cumulative_returns
    return (fitness_value,)



# Genetic Algorithm Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, min_window, max_window_size)  # For mean_window and std_window
toolbox.register("attr_float", random.uniform, min_k, max_k)  # For k1 and k2

# Define a function to create an individual as a combination of mean_window, std_window, k1, and k2
def create_individual():
    return creator.Individual(
        [toolbox.attr_int(),  # mean_window
         toolbox.attr_int(),  # std_window
         toolbox.attr_float(),  # k1
         toolbox.attr_float()]  # k2
        )

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def calculate_strategy(data, individual, previous_data=None):
    buy_executed = False
    mean_window, std_window, k1, k2 = individual
    mean_window = max(1, int(round(mean_window)))
    std_window = max(1, int(round(std_window)))
    
    upper_band, lower_band = bollinger_bands(data, mean_window, std_window, k1, k2, previous_data)
    

    buy_signals = (data['Adj Close'] < lower_band) 
    sell_signals = (data['Adj Close'] > upper_band)
    
    shares_held = 0
    nominal_value = 0
    market_value = 0
    cumulative_returns = 0
    
    shares_held_history = []
    market_value_history = []
    nominal_value_history = []
    returns_history = []
    cumulative_returns_history = [cumulative_returns]

    for i in range(len(data)):
        # Assume no returns at the start of the day
        returns = 0
        next_open_price = data['Adj Close'].iloc[i + 1] if i + 1 < len(data) else data['Adj Close'].iloc[i]

        # Buy logic
        if buy_signals.iloc[i] and not buy_executed:
            shares_held += 1
            nominal_value += next_open_price
            buy_executed = True  # Set flag to true after buy
            
        # Update market value 
        market_value = shares_held * next_open_price

        if nominal_value != 0:
            # Sell signal detected, sell all shares at next day's opening price
            returns = (market_value - nominal_value) / nominal_value
            
        if sell_signals.iloc[i] and shares_held > 0:
            # After selling, reset 
            shares_held = 0
            nominal_value = 0
            market_value = 0
            cumulative_returns += returns
            buy_executed = False  # Reset flag after sell
            

        if i == len(data) - 1:
            cumulative_returns += returns

        
        # Record values in history lists
        shares_held_history.append(shares_held)
        market_value_history.append(market_value)
        nominal_value_history.append(nominal_value)
        returns_history.append(returns)
        cumulative_returns_history.append(cumulative_returns)

    # Assign values to DataFrame
    data['Upper Band'] = upper_band
    data['Lower Band'] = lower_band
    data['Shares Held'] = shares_held_history
    data['Market Value'] = market_value_history
    data['Nominal Value'] = nominal_value_history
    data['Returns'] = returns_history
    data['Cumulative Returns'] = cumulative_returns_history[1:]
    return data

symbols = ["AAPL", "MSFT", "BAC"]

# Store the train and test sets for each stock
stock_datasets = {symbol: {'train': [], 'test': []} for symbol in symbols}

for symbol in symbols:
    print('Hi there,', symbol, 'here being processed')
    data = yf.download(symbol, start=start_date, end=today, threads=True, progress=False)


    train_cumulative_returns = []
    test_cumulative_returns = []

    previous_data = {}
    for i, (train_set, test_set) in enumerate(rolling_windows(data, train_size, test_size, max_window_size)):
        print(f"Processing Window {i+1}: {train_set.index[0].date()} to {test_set.index[-1].date()}")

        # If first iteration, use unused data window for previous_data
        if i == 0:
            previous_data = data.iloc[:max_window_size]
        
        # Define the wrapper function for the current window
        def evaluate_individual(individual):
            # Determine the extension size for the current individual
            extension_size = max(individual[0], individual[1])
            return fitness(individual, train_set, extension_size)
        
        # Register the wrapper function as the evaluate function
        toolbox.register("evaluate", evaluate_individual)
        
        # Run the genetic algorithm to find the best individual parameters
        population = toolbox.population(n=50)

        algorithms.eaSimple(population, toolbox, cxpb= cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
        

        best_individual = tools.selBest(population, k=1)[0]
        #print('best_individual:', best_individual)

        updated_train_set = calculate_strategy(train_set, best_individual, previous_data)
        updated_test_set = calculate_strategy(test_set, best_individual, train_set)
        
        #print(updated_train_set[['Adj Close', 'Lower Band', 'Upper Band', 'Shares Held', 'Nominal Value', 'Market Value', 'Returns', 'Cumulative Returns']])
        #print(updated_test_set[['Adj Close', 'Lower Band', 'Upper Band', 'Shares Held', 'Nominal Value', 'Market Value', 'Returns', 'Cumulative Returns']])

        #print('cumulative returns for train set:', updated_train_set['Cumulative Returns'].iloc[-1])
        #print('cumulative returns for test set:', updated_test_set['Cumulative Returns'].iloc[-1])
        
        # Calculate and store the cumulative returns
        train_cumulative_return = updated_train_set['Cumulative Returns'].iloc[-1]
        test_cumulative_return = updated_test_set['Cumulative Returns'].iloc[-1]
        train_cumulative_returns.append(train_cumulative_return)
        test_cumulative_returns.append(test_cumulative_return)
        previous_data = pd.concat([train_set, test_set])

        # Store the updated train and test sets
        stock_datasets[symbol]['train'].append(updated_train_set)
        stock_datasets[symbol]['test'].append(updated_test_set)

    train_avg_return = np.mean(train_cumulative_returns)
    train_median_return = np.median(train_cumulative_returns)
    test_avg_return = np.mean(test_cumulative_returns)
    test_median_return = np.median(test_cumulative_returns)

    train_summary_measure = (train_avg_return + train_median_return) / 2
    test_summary_measure = (test_avg_return + test_median_return) / 2

    print("Avg return train set:", train_summary_measure)
    print("Avg return test set:", test_summary_measure)

    print('')
    print('----------------------------------------------------------------------------------------------------------')


