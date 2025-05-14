import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import time
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from pytz import timezone
from stocksymbol import StockSymbol
import os
import pickle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import copy
from sklearn.linear_model import LinearRegression



#VARIABLES----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#FUNCTIONALITIES
#analysis
analysis = 'y'
prediction = 'y'
#trading
trading = 'n'
update_params = 'y'

#basics
yrs = 1
div = 1
interval = '1d'
days_backtest = 365

if prediction == 'y':
    prediction_period = 1
    end_date = dt.datetime.today() - timedelta(days= 365 * prediction_period)
else:
    end_date = dt.datetime.today()

#bollinger
start_window = 3
start_k1 = 30
start_k2 = 30
end_window = 200         
end_k1 = 200
end_k2 = 200
ceal = 10 #max_div

#otimization params
THRESHOLD = 0.47
pop_size = 100
max_gen = 9   #this is used as a maximum
inpb = 0.05
tournsize = 3
cxpb = 0.8     #refinement   
mutpb = 0.4     #exploration  
n_iterations = 3
step_size_ranges = [(2, 10), (2, 10), (2, 10)]  # mutation steps size



#VARIABLES SELECTION---------------------------------------------------------------------------------------------------------------------------------------

target = 'index'
market = 'US'
index = 'DJI'

def is_market_open():
    eastern = timezone('US/Eastern')
    eastern_time = datetime.now(eastern)
    hour = eastern_time.hour
    minute = eastern_time.minute
    weekday = eastern_time.weekday()  # Monday is 0 and Sunday is 6
    return weekday < 5 and ((hour > 9 or (hour == 9 and minute >= 30)) and hour < 16)

stock_mkt_open = is_market_open()

#get lists for indexes
api_key = "458ba089-8ff9-41fe-9a23-9d2756368b9f"
ss = StockSymbol(api_key)
market_list = ss.market_list    # show a list of available market
index_list = ss.index_list  # show a list of available index

if target == 'index':
    # get symbol list based on market (if you want an index: index="SPX", for markets: market="US")
    all_symbols_list = ss.get_symbol_list(index=index, symbols_only=True)
    data_filename = f'{index}_downloaded_data.pickle'  #stai attento al data filename

if target == 'market':
    # get symbol list based on market (if you want an index: index="SPX", for markets: market="US")
    all_symbols_list = ss.get_symbol_list(market=market, symbols_only=True)
    data_filename = f'{market}_downloaded_data.pickle'  #stai attento al data filename


# Get symbol lists----------------------------------------------------------------------
start_date = end_date -  pd.Timedelta(days = days_backtest  + (end_window*365)/252)
most_recent_date = start_date
today = datetime.today()

if os.path.exists(data_filename):
    # If the file exists, load the data from it
    with open(data_filename, 'rb') as handle:
        try:
            pickle_dict = pickle.load(handle)
            # Find the most recent date in the data
            data_dict = pickle_dict['data_dict']
            valid_symbols_list = pickle_dict['valid_symbols_list']
            if data_dict:
                most_recent_date = max([data.index[-1] if not data.empty else start_date for data in data_dict.values()])
            else:
                most_recent_date = start_date  # start_date instead of placeholder date
            # If the most recent date is before today, download the missing data
            if stock_mkt_open and most_recent_date.date() < today.date() or (not stock_mkt_open and (today.date() - most_recent_date.date()).days >= 2):
                for symbol in all_symbols_list:
                    new_data = yf.download(symbol, start = most_recent_date + pd.Timedelta(days=1), end=today, threads=True, progress=False)    
                    new_data.dropna(inplace=True)

                    # Concatenate the new data to the old data
                    if not new_data.empty:  # Add this check to ensure data was downloaded
                        data_dict[symbol] = pd.concat([data_dict.get(symbol, pd.DataFrame()), new_data])
            # Save the updated data to a file for future use + create a dictionary to store both data and valid symbols list
            pickle_dict = {'data_dict': data_dict, 'valid_symbols_list': valid_symbols_list}
            with open(data_filename, 'wb') as handle:
                pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except KeyError:
            # If data_dict not found in pickle_dict, initialize as empty dict
            data_dict = {}
            valid_symbols_list = []
else:
    # If the file doesn't exist, download the data
    data_dict = {}
    valid_symbols_list = []
    for symbol in all_symbols_list:
        data = yf.download(symbol, start = start_date, end=today, threads=True, progress=False)
        if not data.empty:  # Add this check to ensure data was downloaded
            data['daily_returns'] = np.log(data['Open'] / data['Open'].shift(1))
            data.dropna(inplace=True)
            data_dict[symbol] = data
            valid_symbols_list.append(symbol)  # Add the symbol to the new list

    # Save the downloaded data to a file for future use
    with open(data_filename, 'wb') as handle:
        pickle.dump({'data_dict': data_dict, 'valid_symbols_list': valid_symbols_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Replace all_symbols_list with the new list of valid symbols
symbols = valid_symbols_list

# SOME BASIC STUFF---------------------------------------------------------------------------------------------------------------------------------------------------------------
#some basic visualization options
pd.options.mode.chained_assignment = None  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None) #fundamental to display the df not in pieces
cols_to_exclude = ['upper_band','lower_band', 'moving_avg', 'moving_std', 'buying_period']

# Define the number of colors you want
n_colors = 20
cmap = plt.get_cmap('tab20', n_colors) 

# Set up DEAP toolbox
if "FitnessMax" not in creator.__dict__:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
                
toolbox = base.Toolbox()
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=tournsize)


#1ST CLASSES---------------------------------------------------------------------------------------------------------------------------------------------------
# Define a class to handle financial data and Bollinger bands
class FinancialData:
    def __init__(self, symbol, end, data):
        self.symbol = symbol
        self.end = end
        self.data = data.copy()
        self.data['daily_returns'] = np.log(self.data['Open'] / self.data['Open'].shift(1))
        self.data.dropna(inplace=True)

    def retrieve_data(self, symbol, start, end):
        return yf.download(symbol, start=start, end = end_date, period= '1d', interval = interval, threads=True, progress=False)

class BollingerStrat(FinancialData):
    def __init__(self, symbol, end, days):
        super().__init__(symbol, end, days)

    def backtest_bollinger(self, window, upper_coeff, lower_coeff, div = 1, start = None):  
        # print('using window', int(window), 'k1', int(upper_coeff), 'k2', int(lower_coeff), 'div', div)
        # slice the downloaded df according to 'div'
        start_date = self.end - pd.Timedelta(days = div * days_backtest  + (end_window*365)/252)
        data = pd.DataFrame()
        data = self.data[self.data.index >= start_date]
        # create variable_df
        global variable_df
        variable_df = pd.DataFrame()
        variable_df['price'] = data['Open']
        variable_df['daily_returns'] = data['daily_returns']
        # compute bollinger's essentials
        variable_df[['moving_avg', 'moving_std']] = variable_df['price'].rolling(window = window).agg(['mean', 'std'])
        #variable_df['moving_avg'] = variable_df['price'].ewm(span=window, adjust=False).mean()
        #variable_df['moving_std'] = variable_df['price'].ewm(span=window, adjust=False).std()
        #go on
        variable_df = variable_df.iloc[end_window:]
        variable_df['upper_band'] = variable_df['moving_avg'] + upper_coeff/100 * variable_df['moving_std']
        variable_df['lower_band'] = variable_df['moving_avg'] - lower_coeff/100 * variable_df['moving_std']
        # define buy and sell signals
        variable_df['buy_signal'] = np.where(variable_df['price'] < variable_df['lower_band'], 1, 0)
        variable_df['sell_signal'] = np.where(variable_df['price'] > variable_df['upper_band'], 1, 0)
        # pf units
        variable_df.loc[variable_df['buy_signal'] == 1, 'units'] = 1
        variable_df['buying_period'] = 0
        variable_df['buying_period'] = (variable_df['sell_signal'] == 1).shift().cumsum()  # Now, calculate the cumulative sum of units within each buying period
        variable_df.fillna(0, inplace = True)
        variable_df['units'] = variable_df.groupby('buying_period')['units'].cumsum()
        variable_df.loc[(variable_df['sell_signal'] == 1) & (variable_df['buy_signal'].shift(1) == 0) & (variable_df['sell_signal'].shift(1) == 1), 'units'] = 0
        variable_df['units'].fillna(method='ffill', inplace=True)
        #cost of cumultative position
        variable_df['cost'] = variable_df['price'].where(variable_df['buy_signal'] == 1, 0)
        variable_df['cumulative_cost'] =  variable_df.groupby('buying_period')['cost'].cumsum()
        #  portfolio value
        variable_df['mkt_value'] =(variable_df['units'] * variable_df['price'])
        #startegy returns and bnh returns
        variable_df['strategy_returns_per_op'] =  (variable_df['units'] * variable_df['price']) / (variable_df['cumulative_cost'])  - 1
        # Create a new column 'deductions' that adds 0.05 each time there's a 'buy_signal'
        variable_df['deductions'] = (variable_df['buy_signal'] * 0.0030)
        variable_df['deductions'] = variable_df.groupby('buying_period')['deductions'].cumsum()
        # Then subtract 'deductions' from 'strategy_returns'
        variable_df['strategy_returns_per_op'] -= variable_df['deductions']
        variable_df['strategy_returns_f'] = 0
        variable_df.loc[variable_df['sell_signal'] == 1, 'strategy_returns_f'] = variable_df['strategy_returns_per_op']
        variable_df['strategy_returns_f'] = variable_df['strategy_returns_f'].cumsum()
        variable_df['strategy_returns_f'].fillna(method='ffill', inplace=True)
        variable_df['strategy_returns'] = variable_df['strategy_returns_per_op'].add(variable_df['strategy_returns_f'], fill_value=0)
        variable_df.loc[variable_df['sell_signal'] == 1, 'strategy_returns'] = variable_df['strategy_returns'] - variable_df['strategy_returns_per_op']
        variable_df['strategy_returns'] = variable_df['strategy_returns'].replace(0, np.nan)
        variable_df['strategy_returns'].fillna(method='ffill', inplace=True)
        variable_df = variable_df.fillna(0)
        variable_df['bnh_returns'] = variable_df['daily_returns'].cumsum()
        performance = variable_df[['bnh_returns', 'strategy_returns']].iloc[-1]
        # print strat i returns
        #print('here', performance['strategy_returns'])
        #cols_to_include = [col for col in variable_df.columns if col not in cols_to_exclude]       
        #print(variable_df[cols_to_include])
        return  -performance
                
    def optimize_bollinger_band_parameters(self, params):
        performance = self.backtest_bollinger(int(params[0]), params[1], params[2]) # Params[0] corresponds to window, params[1] to upper_coeff and params[2] to lower_coeff
        return -performance['strategy_returns'],

    def buy_and_sell_signals_bollinger(self):
        global buy_list
        global sell_list
        close_list = pd.to_numeric(variable_df['price'], downcast='float')
        upper_list = pd.to_numeric(variable_df['upper_band'], downcast='float')
        lower_list = pd.to_numeric(variable_df['lower_band'], downcast='float')
        buy_list  = pd.to_numeric(variable_df['buy_signal'], downcast='float')
        sell_list = pd.to_numeric(variable_df['sell_signal'], downcast='float')
        df_plot = variable_df.astype(float)
        df_plot[['price', 'moving_avg','upper_band','lower_band']].plot()
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('price',fontsize=18)
        x_axis = df_plot.index
        plt.fill_between(x_axis, df_plot['lower_band'], df_plot['upper_band'], color='grey',alpha=0.30)
        plt.scatter(df_plot[buy_list == 1].index, df_plot[buy_list == 1]['price'], color='green',label='Buy',  marker='^', alpha = 1)
        plt.scatter(df_plot[sell_list == 1].index, df_plot[sell_list == 1]['price'], color='red',label='Sell',  marker='v', alpha = 1)
        plt.draw()
        plt.show()        
                    
def evalOneMax(individual):
        return bollinger.optimize_bollinger_band_parameters(individual),
                
def custom_mutate(individual, mutpb):
    for i in range(len(individual)):
        if random.random() < mutpb:  # check if we are to mutate this attribute
            step_size = random.uniform(*step_size_ranges[i])  # select the step size based on the variable's range
            individual[i] += (random.uniform(-step_size, step_size))  # add or subtract the step size
    # Ensure that attributes are within their required bounds
    individual[0] = max(min(individual[0], end_k1), start_k1)  
    individual[1] = max(min(individual[1], end_k2), start_k2)  
    individual[2] = max(min(individual[2], end_window), start_window) 
    return individual,

def calculate_trend(series):
    series = series[-50:]
    y = series.values.reshape(-1,1)
    X = np.array(range(len(series))).reshape(-1,1)
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0][0]

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    halloffame.update(population)  # Initial HoF update
    best_ind_each_gen.append(halloffame[0])
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    for gen in range(1, max_gen + 1):   # Start evolution
        #print('we at gen', gen)
        offspring = toolbox.select(population, len(population)) # Select the next generation individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)  # Vary the pool of individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid] # Evaluate the individuals with an invalid fitness
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        halloffame.update(offspring) 
        best_ind_each_gen.append(halloffame[0])
        population[:] = offspring # Replace the current population by the offspring
        if stats is not None:
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        else:
            logbook.record(gen=gen, nevals=len(invalid_ind))
        if verbose:
            print(logbook.stream)
        if max([ind.fitness.values[0] for ind in population]) >= THRESHOLD:   # Check if any individual has reached the threshold value
            #print('I hit the treshold')
            break
    return population, logbook, best_ind_each_gen

# Initialize the list before the for loop
best_individuals_per_iteration = []
cumulative_returns_all_symbols = {}
cumulative_returns_all_symbols_backtest = {}
best_params_per_symbol = {}
cumulative_returns_all_symbols_NOW = {}
best_params_df = pd.DataFrame()
max_data_per_symbol = {}
playing_dict = {}
optimal_params_dict = {}
        
if analysis == 'y':                         
    for _sym in symbols: #correct 3
                max_data =data_dict[_sym]
                print()
                max_data_per_symbol[_sym] = max_data
                num_years = (max_data.index.max() - max_data.index.min()).days / 365.25
                maximum_div = min(int(num_years), ceal)
            
            #while end_k1 > 60:
            #while min_div < maximum_div:
                best_ind_each_gen = []
                toolbox.register("attr_window", random.randint, start_window, end_window)
                toolbox.register("attr_upper_coeff", random.randint, start_k1, end_k1)
                toolbox.register("attr_lower_coeff", random.randint, start_k1, end_k2)
                toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_window, toolbox.attr_upper_coeff, toolbox.attr_lower_coeff), n=1)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                toolbox.register("mutate", custom_mutate, mutpb = mutpb)
                print('')
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                print("We playing with", _sym)
                #OPTIMIZATION
                for i in range(1, n_iterations):
                    bollinger = BollingerStrat(_sym, end_date, max_data)
                    toolbox.register("evaluate", bollinger.optimize_bollinger_band_parameters)                       
                    population = toolbox.population(n=pop_size)
                    halloffame = tools.HallOfFame(maxsize=1) # Create a Hall of Fame object that will contain the top individuals     
                    eaSimpleWithElitism(population, toolbox, cxpb=cxpb, mutpb= mutpb, ngen=max_gen, halloffame=halloffame, verbose=False)
                    best_ind = tools.selBest(population, 1)[0] # Now, the population should contain the N best individuals due to elitism:
                    bollinger.backtest_bollinger(int(best_ind[0]), best_ind[1], best_ind[2])# Run the strategy with the best parameters
                    annual_return = variable_df['strategy_returns'].iloc[-1] 

                cols_to_include = [col for col in variable_df.columns if col not in cols_to_exclude]       
                #QUI C'è TUTTO ILL CODICE PER VEDERE TUTTI I TOP TROVATI
                #for individual in best_ind_each_gen:
                #    print(f"Individual: {individual}, Fitness: {individual.fitness.values[0]}")
                best_ind_each_gen.sort(key=lambda x: x.fitness.values[0], reverse=True) # Sort the individuals in descending order based on their fitness
                optimal_params = best_ind_each_gen[0]
                print(optimal_params) # Extract the fitness value of the optimal individual
                optimal_fitness = optimal_params.fitness.values[0]
                print(f"Optimal fitness: {optimal_fitness}")
                optimal_window, optimal_k1, optimal_k2 = optimal_params
                print('')
                print('BACKTEST')
                backtest_div = 1
                bollinger.backtest_bollinger(int(optimal_window), int(optimal_k1), optimal_k2)
                min_value = variable_df['strategy_returns_per_op'].min() #Drawdown
                print('Strat_returns', variable_df['strategy_returns'].iloc[-1] )
                print("Max training pf drawdown:", min_value)
                invested_days = variable_df[variable_df['units'] > 0].shape[0]
                print("Total number of days invested:", invested_days)
                print("best training pf")
                print(variable_df[cols_to_include])
                if variable_df['buy_signal'].iloc[-1] > 0:
                        print ("Buy under:", variable_df['price'].iloc[-1])
                elif variable_df['sell_signal'].iloc[-1] > 0:
                        print ("Sell over:", variable_df['price'].iloc[-1])
                else:
                        print ("")
                # Store the optimal parameters for this symbol
                optimal_params_dict[_sym] = (optimal_window, optimal_k1, optimal_k2)

                    
                if prediction == 'y':
                        # Loop over each parameter combination in the hall of fame
                        print('')
                        monthly_returns = {}        
                        # Run the strategy with the optimal parameters and calculate returns for each month
                        for month in (1, 3):
                            end_date_month = end_date + pd.Timedelta(days = 30*month)
                            div_pred = month * 0.082   # this oughts to be changed as days backtest change
                            max_data_new = yf.download(_sym, start='2021-07-07', end= end_date_month, period= '1d', interval = '1d', threads=True, progress=False)
                            bollinger = BollingerStrat(_sym,end_date_month, max_data_new)
                            performance = bollinger.backtest_bollinger(int(optimal_window), optimal_k1, optimal_k2, div = div_pred)  
                            # Store the returns in the dictionary
                            monthly_returns[f"After {month} month(s)"] =  variable_df['strategy_returns'].iloc[-1]
                            #print(variable_df)                  
                            if month == 3:
                                    bollinger.backtest_bollinger(int(optimal_window), optimal_k1, optimal_k2, div = div_pred)  
                                    min_value = variable_df['strategy_returns_per_op'].min()
                                    print("Max testing pf drawdown:", min_value)
                                    # Now, create the new Series for each symbol and dv, using only the last 252 rows of data
                                    cumulative_returns_all_symbols[_sym] = variable_df['strategy_returns'].copy()
                                    print('n_buy:', (variable_df['buy_signal'] == 1).sum()/ div )
                                    print(variable_df)  
                                    #playing_dict[end_k1] = monthly_returns[f"After 3 month(s)"]        #here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # Display the results
                        for month, returns in monthly_returns.items():
                            print(f"{month}: {returns}")
                        print('')

    # After all symbols have been processed, print the optimal parameters for each
    for symbol, params in optimal_params_dict.items():
        print(f"Optimal parameters for {symbol}: {params}")
    # Define the file path
    file_path = 'optimal_params.xlsx'
    # Check if the file already exists
    if os.path.exists(file_path):
        # If the file exists, read it
        df = pd.read_excel(file_path, index_col=0)
    else:
        # If the file does not exist, initialize an empty DataFrame
        df = pd.DataFrame(columns=['Optimal_Window', 'Optimal_K1', 'Optimal_K2'])

    # Update the DataFrame with the new parameters
    for symbol, params in optimal_params_dict.items():
        if update_params == 'y':
            df.loc[symbol] = params
        else:
            if symbol not in df.index:
                df.loc[symbol] = params  # Only add the new symbol if it does not exist in the DataFrame
                
    # Save DataFrame to Excel file
    df.to_excel(file_path)
                    
#####################--------------------

retrieved_data_per_symbol = {}

if trading == 'y':
    # Load the DataFrame from the Excel file
    df = pd.read_excel('optimal_params.xlsx', index_col=0)  # index_col=0 because the first column (0-indexed) is the index
    for _sym in symbols:
            # Retrieve optimal parameters from dictionary
            retrieved_data =  yf.download(_sym, start='2022-07-01', end = end_date,interval = interval, threads=True, progress=False)
            retrieved_data_per_symbol[_sym] = retrieved_data
            optimal_params = df.loc[_sym]  # Use .loc to get the row for this symbol
            optimal_window = optimal_params['Optimal_Window']
            optimal_k1 = optimal_params['Optimal_K1']
            optimal_k2 = optimal_params['Optimal_K2']
            bollinger = BollingerStrat(_sym, end_date, retrieved_data)
            bollinger.backtest_bollinger(int(optimal_window), int(optimal_k1), int(optimal_k2))
            total_return = variable_df['strategy_returns'].iloc[-1]
            cumulative_returns_all_symbols_NOW[_sym] = variable_df['strategy_returns'].copy()
            
    trends = {sym: calculate_trend(series) for sym, series in cumulative_returns_all_symbols_NOW.items()}
    top_series = sorted(trends, key=trends.get, reverse=True)[:8] # Sort the series by their trend and select the top n
    print(f"The series with the strongest trends are: {top_series}")

if trading == 'y':
    cumulative_returns_all_symbols_NOW = {}
    for _sym in top_series:
    # Retrieve optimal parameters from dictionary
            top_data = retrieved_data_per_symbol[_sym]
            optimal_params = df.loc[_sym]  # Use .loc to get the row for this symbol
            optimal_window = int(optimal_params['Optimal_Window'])
            optimal_k1 = int(optimal_params['Optimal_K1'])
            optimal_k2 = int(optimal_params['Optimal_K2'])
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            print(f"We playing with {_sym}, using optimal_window {optimal_window}, optimal_k1 {optimal_k1}, and optimal_k2 {optimal_k2}")
            bollinger = BollingerStrat(_sym, end_date, top_data)
            div = 0.35
            bollinger.backtest_bollinger(int(optimal_window), int(optimal_k1), int(optimal_k2), div = div)
            min_value = variable_df['strategy_returns_per_op'].min() #drawdown
            total_return = variable_df['strategy_returns'].iloc[-1]
            cols_to_include = [col for col in variable_df.columns if col not in cols_to_exclude]       
            print(variable_df[cols_to_include])
            print('total return:', total_return)
            print("Max training pf drawdown:", min_value)
            # Calculate the return of the last n months (approx. 84 trading days)
            three_month_return = variable_df['strategy_returns'].iloc[-1] - variable_df['strategy_returns'].iloc[-63]
            print('Three month return:', three_month_return)
            one_month_return = variable_df['strategy_returns'].iloc[-1] - variable_df['strategy_returns'].iloc[-21]
            print('One month return:', one_month_return)
            invested_days = variable_df[variable_df['units'] > 0].shape[0]
            print("Total number of days invested:", invested_days)
            print("best training pf")
            if variable_df['buy_signal'].iloc[-1] > 0:
                print ("BUYYYYYYYYYYYYYYYYYYYY:", variable_df['price'].iloc[-1])
                print('')
            elif variable_df['sell_signal'].iloc[-1] > 0:
                print ("SELLLLLLLLLLLLLLLLLLLL:", variable_df['price'].iloc[-1])
                print('')
            else:
                print ("")
            cumulative_returns_all_symbols_NOW[_sym] = variable_df['strategy_returns'].copy()
            
    # Check if 'cumulative_returns_all_symbols' dictionary is empty
    if not cumulative_returns_all_symbols_NOW:
        print('No data to plot')
    else:
        # Plotting cumulative returns
        plt.figure(figsize=(10,6))
        for symbol, returns in cumulative_returns_all_symbols_NOW.items():
            if returns.isnull().any():
                print(f'{symbol} has NaN values')
            if not returns.empty:
                plt.plot(returns.index, returns, label=symbol)
            else:
                print(f'{symbol} has no data')

        plt.title('Cumulative Returns over Time for All Symbols')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.show()

