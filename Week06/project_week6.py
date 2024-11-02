#problem 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

current_stock_price = 165  
current_date = datetime(2023, 3, 3)  
expiration_date = datetime(2023, 3, 17)  
risk_free_rate = 5.25 / 100  
dividend_yield = 0.53 / 100 

cost_of_carry = risk_free_rate - dividend_yield

days_to_maturity = (expiration_date - current_date).days
time_to_maturity = days_to_maturity / 365

def gbsm(call, S, K, T, rf, b, sigma):
    d1 = (np.log(S / K) + (b + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * np.exp((b - rf) * T) * norm.cdf(d1) - K * np.exp(-rf * T) * norm.cdf(d2)
    else:
        return K * np.exp(-rf * T) * norm.cdf(-d2) - S * np.exp((b - rf) * T) * norm.cdf(-d1)

strike_price = current_stock_price

implied_volatilities = np.linspace(0.1, 0.8, 100)

call_values = [gbsm(True, current_stock_price, strike_price, time_to_maturity, risk_free_rate, cost_of_carry, sigma) for sigma in implied_volatilities]
put_values = [gbsm(False, current_stock_price, strike_price, time_to_maturity, risk_free_rate, cost_of_carry, sigma) for sigma in implied_volatilities]

plt.figure(figsize=(10, 6))
plt.plot(implied_volatilities, call_values, label="Call Option Value", color="blue")
plt.plot(implied_volatilities, put_values, label="Put Option Value", color="red")
plt.xlabel("Implied Volatility")
plt.ylabel("Option Value")
plt.title("Option Value vs. Implied Volatility (Call and Put) with Continuous Dividend Yield")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

#problem 2
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from datetime import datetime

file_path = 'AAPL_Options.csv'  
aapl_options = pd.read_csv(file_path)

current_price = 170.15  
current_date = datetime(2023, 10, 30)  
risk_free_rate = 5.25 / 100  
dividend_rate = 0.57 / 100  

def calculate_time_to_maturity(expiration_date):
    expiration = datetime.strptime(expiration_date, "%m/%d/%Y")
    days_to_maturity = (expiration - current_date).days
    return days_to_maturity / 365

def gbsm_price(call, S, K, T, rf, b, sigma):
    d1 = (np.log(S / K) + (b + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * np.exp((b - rf) * T) * norm.cdf(d1) - K * np.exp(-rf * T) * norm.cdf(d2)
    else:
        return K * np.exp(-rf * T) * norm.cdf(-d2) - S * np.exp((b - rf) * T) * norm.cdf(-d1)

def calculate_implied_volatility(row):
    strike = row['Strike']
    option_price = row['Last Price']
    option_type = row['Type']
    T = calculate_time_to_maturity(row['Expiration'])
    is_call = option_type == 'Call'
    cost_of_carry = risk_free_rate - dividend_rate 

    def implied_volatility_function(sigma):
        return gbsm_price(is_call, current_price, strike, T, risk_free_rate, cost_of_carry, sigma) - option_price

    try:
        result = root_scalar(implied_volatility_function, bracket=[0.01, 3.0], method='brentq')
        return result.root if result.converged else np.nan
    except:
        return np.nan

aapl_options['Implied Volatility'] = aapl_options.apply(lambda row: calculate_implied_volatility(row), axis=1)

calls = aapl_options[aapl_options['Type'] == 'Call']
puts = aapl_options[aapl_options['Type'] == 'Put']

call_strikes = calls['Strike'].values
call_iv = calls['Implied Volatility'].values
put_strikes = puts['Strike'].values
put_iv = puts['Implied Volatility'].values

plt.figure(figsize=(12, 6))
plt.plot(call_strikes, call_iv, label='Call Options', color='blue', marker='o', linestyle='-')
plt.plot(put_strikes, put_iv, label='Put Options', color='red', marker='o', linestyle='-')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price for AAPL Call and Put Options')
plt.legend()
plt.grid(True)
plt.show()

#problem 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import root_scalar
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg

portfolios_path = 'problem3.csv'
portfolios = pd.read_csv(portfolios_path)

current_price = 170.15 
current_date = datetime(2023, 10, 30)  
risk_free_rate = 5.25 / 100  
dividend_rate = 0.57 / 100 

def calculate_time_to_maturity(expiration_date):
    expiration = datetime.strptime(expiration_date, "%m/%d/%Y")
    days_to_maturity = (expiration - current_date).days
    return days_to_maturity / 365

def bs_price(call, S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def calculate_implied_volatility(row, S, T, r, q):
    market_price = row['CurrentPrice']
    strike = row['Strike']
    option_type = row['OptionType']
    is_call = option_type == 'Call'
    
    def bs_price_diff(sigma):
        return bs_price(is_call, S, strike, T, r, q, sigma) - market_price

    try:
        result = root_scalar(bs_price_diff, bracket=[0.01, 3.0], method='brentq')
        return result.root if result.converged else np.nan
    except:
        return np.nan

T = calculate_time_to_maturity("12/15/2023")  
portfolios['Implied Volatility'] = portfolios.apply(calculate_implied_volatility, args=(current_price, T, risk_free_rate, dividend_rate), axis=1)

underlying_values = np.linspace(100, 240, 100)

portfolio_values = {}

for portfolio in portfolios['Portfolio'].unique():
    portfolio_data = portfolios[portfolios['Portfolio'] == portfolio]
    portfolio_values[portfolio] = []

    for S in underlying_values:
        value = 0
        for _, row in portfolio_data.iterrows():
            holding = row['Holding']
            strike = row['Strike']
            option_type = row['OptionType']
            is_call = option_type == 'Call'
            sigma = row['Implied Volatility']  

            option_price = bs_price(is_call, S, strike, T, risk_free_rate, dividend_rate, sigma)
            value += holding * option_price
        
        portfolio_values[portfolio].append(value)

plt.figure(figsize=(12, 6))
for portfolio, values in portfolio_values.items():
    plt.plot(underlying_values, values, label=f'Portfolio: {portfolio}')

plt.xlabel('AAPL Underlying Price')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Values vs AAPL Underlying Price')
plt.legend()
plt.grid(True)
plt.show()


daily_prices_path = 'DailyPrices.csv'
daily_prices = pd.read_csv(daily_prices_path)

daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
daily_prices.set_index('Date', inplace=True)

aapl_prices = daily_prices['AAPL']
aapl_log_returns = np.log(aapl_prices / aapl_prices.shift(1)).dropna()

demeaned_log_returns = aapl_log_returns - aapl_log_returns.mean()

ar_model = AutoReg(demeaned_log_returns, lags=1).fit()

np.random.seed(42)  
last_return = demeaned_log_returns.iloc[-1]  
simulated_returns = []

phi = ar_model.params['AAPL.L1']
constant = ar_model.params['const']
residual_std = ar_model.sigma2 ** 0.5 

for _ in range(10):
    new_return = constant + phi * last_return + np.random.normal(0, residual_std)
    simulated_returns.append(new_return)
    last_return = new_return

current_price = 170.15 
simulated_prices = [current_price * np.exp(np.sum(simulated_returns[:i])) for i in range(1, 11)]

pnl = np.array(simulated_prices) - current_price

mean_pnl = np.mean(pnl)
var_95 = np.percentile(pnl, 5)  
es_95 = np.mean(pnl[pnl <= var_95])  

print("Mean P&L:", mean_pnl)
print("VaR (95%):", var_95)
print("Expected Shortfall (95%):", es_95)
