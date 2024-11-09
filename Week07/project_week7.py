#Problem 1
import numpy as np
from scipy.stats import norm
from datetime import datetime

def time_to_maturity(current_date, expiration_date):
    date_format = "%m/%d/%Y"
    current = datetime.strptime(current_date, date_format)
    expiration = datetime.strptime(expiration_date, date_format)
    ttm = (expiration - current).days / 365
    return ttm

def gbsm_price_and_greeks(call, S, K, ttm, rf, b, ivol):
    d1 = (np.log(S / K) + (b + 0.5 * ivol**2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)

    if call:
        price = S * np.exp((b - rf) * ttm) * norm.cdf(d1) - K * np.exp(-rf * ttm) * norm.cdf(d2)
        delta = np.exp((b - rf) * ttm) * norm.cdf(d1)
        rho = ttm * K * np.exp(-rf * ttm) * norm.cdf(d2)
    else:
        price = K * np.exp(-rf * ttm) * norm.cdf(-d2) - S * np.exp((b - rf) * ttm) * norm.cdf(-d1)
        delta = -np.exp((b - rf) * ttm) * norm.cdf(-d1)
        rho = -ttm * K * np.exp(-rf * ttm) * norm.cdf(-d2)

    gamma = np.exp((b - rf) * ttm) * norm.pdf(d1) / (S * ivol * np.sqrt(ttm))
    vega = S * np.exp((b - rf) * ttm) * norm.pdf(d1) * np.sqrt(ttm)
    theta = -(S * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm))) \
            - (b - rf) * S * np.exp((b - rf) * ttm) * norm.cdf(d1 if call else -d1) \
            - rf * K * np.exp(-rf * ttm) * norm.cdf(d2 if call else -d2)

    return price, delta, gamma, vega, theta, rho

def finite_difference_greeks(call, S, K, ttm, rf, b, ivol, epsilon=1e-5):
    def price_func(S, K, ttm, rf, b, ivol):
        return gbsm_price_and_greeks(call, S, K, ttm, rf, b, ivol)[0]

    delta = (price_func(S + epsilon, K, ttm, rf, b, ivol) - price_func(S - epsilon, K, ttm, rf, b, ivol)) / (2 * epsilon)

    gamma = (price_func(S + epsilon, K, ttm, rf, b, ivol) - 2 * price_func(S, K, ttm, rf, b, ivol) + price_func(S - epsilon, K, ttm, rf, b, ivol)) / (epsilon**2)

    vega = (price_func(S, K, ttm, rf, b, ivol + epsilon) - price_func(S, K, ttm, rf, b, ivol - epsilon)) / (2 * epsilon)

    theta = (price_func(S, K, ttm - epsilon, rf, b, ivol) - price_func(S, K, ttm + epsilon, rf, b, ivol)) / (2 * epsilon)

    rho = (price_func(S, K, ttm, rf + epsilon, b, ivol) - price_func(S, K, ttm, rf - epsilon, b, ivol)) / (2 * epsilon)

    return delta, gamma, vega, theta, rho


S = 151.03                        
K = 165                      
current_date = "03/13/2022"       
expiration_date = "04/15/2022"    
rf = 0.0425                  
b = 0.0053                        
ivol = 0.2                     

ttm = time_to_maturity(current_date, expiration_date)

price_call, delta_call, gamma_call, vega_call, theta_call, rho_call = gbsm_price_and_greeks(True, S, K, ttm, rf, b, ivol)
price_put, delta_put, gamma_put, vega_put, theta_put, rho_put = gbsm_price_and_greeks(False, S, K, ttm, rf, b, ivol)

delta_fd_call, gamma_fd_call, vega_fd_call, theta_fd_call, rho_fd_call = finite_difference_greeks(True, S, K, ttm, rf, b, ivol)
delta_fd_put, gamma_fd_put, vega_fd_put, theta_fd_put, rho_fd_put = finite_difference_greeks(False, S, K, ttm, rf, b, ivol)

print("Call Option:")
print(f"Price: {price_call}")
print("Closed-Form Greeks:")
print(f"Delta: {delta_call}, Gamma: {gamma_call}, Vega: {vega_call}, Theta: {theta_call}, Rho: {rho_call}")
print("Finite Difference Greeks:")
print(f"Delta: {delta_fd_call}, Gamma: {gamma_fd_call}, Vega: {vega_fd_call}, Theta: {theta_fd_call}, Rho: {rho_fd_call}\n")

print("Put Option:")
print(f"Price: {price_put}")
print("Closed-Form Greeks:")
print(f"Delta: {delta_put}, Gamma: {gamma_put}, Vega: {vega_put}, Theta: {theta_put}, Rho: {rho_put}")
print("Finite Difference Greeks:")
print(f"Delta: {delta_fd_put}, Gamma: {gamma_fd_put}, Vega: {vega_fd_put}, Theta: {theta_fd_put}, Rho: {rho_fd_put}")

import numpy as np

def binomial_tree_american(call, S, K, ttm, rf, b, ivol, N, div_amt=None, div_date=None, current_date="03/13/2022"):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    if div_amt and div_date:
        from datetime import datetime
        date_format = "%m/%d/%Y"
        current = datetime.strptime(current_date, date_format)
        dividend = datetime.strptime(div_date, date_format)
        div_time_step = int((dividend - current).days / (ttm * 365) * N)
    else:
        div_time_step = None

    option_values = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        price = S * (u ** i) * (d ** (N - i))
        option_values[i, N] = max(0, z * (price - K))

    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            if div_time_step is not None and j == div_time_step:
                price = S * (u ** i) * (d ** (j - i)) - div_amt
            else:
                price = S * (u ** i) * (d ** (j - i))

            early_exercise_value = max(0, z * (price - K))
            continuation_value = df * (pu * option_values[i + 1, j + 1] + pd * option_values[i, j + 1])
            option_values[i, j] = max(early_exercise_value, continuation_value)

    return option_values[0, 0]

S = 151.03
K = 165
ttm = 33 / 365  
rf = 0.0425
b = 0.0053
ivol = 0.2
N = 100  
div_amt = 0.88
div_date = "04/11/2022"

call_price_no_div = binomial_tree_american(True, S, K, ttm, rf, b, ivol, N)
put_price_no_div = binomial_tree_american(False, S, K, ttm, rf, b, ivol, N)
call_price_with_div = binomial_tree_american(True, S, K, ttm, rf, b, ivol, N, div_amt, div_date)
put_price_with_div = binomial_tree_american(False, S, K, ttm, rf, b, ivol, N, div_amt, div_date)

print(f"Call Option Price without Dividend: {call_price_no_div}")
print(f"Put Option Price without Dividend: {put_price_no_div}")
print(f"Call Option Price with Dividend: {call_price_with_div}")
print(f"Put Option Price with Dividend: {put_price_with_div}")

def calculate_greeks(call, S, K, ttm, rf, b, ivol, N, div_amt=None, div_date=None):
    epsilon = 1e-4
    price_up = binomial_tree_american(call, S + epsilon, K, ttm, rf, b, ivol, N, div_amt, div_date)
    price_down = binomial_tree_american(call, S - epsilon, K, ttm, rf, b, ivol, N, div_amt, div_date)
    delta = (price_up - price_down) / (2 * epsilon)

    gamma = (price_up - 2 * binomial_tree_american(call, S, K, ttm, rf, b, ivol, N, div_amt, div_date) + price_down) / (epsilon ** 2)

    ivol_epsilon = 1e-4
    price_ivol_up = binomial_tree_american(call, S, K, ttm, rf, b, ivol + ivol_epsilon, N, div_amt, div_date)
    price_ivol_down = binomial_tree_american(call, S, K, ttm, rf, b, ivol - ivol_epsilon, N, div_amt, div_date)
    vega = (price_ivol_up - price_ivol_down) / (2 * ivol_epsilon)

    ttm_epsilon = 1e-4
    price_ttm_down = binomial_tree_american(call, S, K, ttm - ttm_epsilon, rf, b, ivol, N, div_amt, div_date)
    theta = (price_ttm_down - binomial_tree_american(call, S, K, ttm, rf, b, ivol, N, div_amt, div_date)) / ttm_epsilon

    rf_epsilon = 1e-4
    price_rf_up = binomial_tree_american(call, S, K, ttm, rf + rf_epsilon, b, ivol, N, div_amt, div_date)
    price_rf_down = binomial_tree_american(call, S, K, ttm, rf - rf_epsilon, b, ivol, N, div_amt, div_date)
    rho = (price_rf_up - price_rf_down) / (2 * rf_epsilon)

    return delta, gamma, vega, theta, rho

call_greeks = calculate_greeks(True, S, K, ttm, rf, b, ivol, N, div_amt, div_date)
put_greeks = calculate_greeks(False, S, K, ttm, rf, b, ivol, N, div_amt, div_date)

print("\nCall Option Greeks with Dividend:")
print(f"Delta: {call_greeks[0]}, Gamma: {call_greeks[1]}, Vega: {call_greeks[2]}, Theta: {call_greeks[3]}, Rho: {call_greeks[4]}")

print("\nPut Option Greeks with Dividend:")
print(f"Delta: {put_greeks[0]}, Gamma: {put_greeks[1]}, Vega: {put_greeks[2]}, Theta: {put_greeks[3]}, Rho: {put_greeks[4]}")

def calculate_dividend_sensitivity(call, S, K, ttm, rf, b, ivol, N, div_amt, div_date, epsilon=1e-4):
    price_higher_div = binomial_tree_american(call, S, K, ttm, rf, b, ivol, N, div_amt + epsilon, div_date)
    price_lower_div = binomial_tree_american(call, S, K, ttm, rf, b, ivol, N, div_amt - epsilon, div_date)

    dividend_sensitivity = (price_higher_div - price_lower_div) / (2 * epsilon)
    return dividend_sensitivity

call_dividend_sensitivity = calculate_dividend_sensitivity(True, S, K, ttm, rf, b, ivol, N, div_amt, div_date)
put_dividend_sensitivity = calculate_dividend_sensitivity(False, S, K, ttm, rf, b, ivol, N, div_amt, div_date)

print("\nSensitivity of Call Option to Change in Dividend Amount:")
print(f"Dividend Sensitivity: {call_dividend_sensitivity}")

print("\nSensitivity of Put Option to Change in Dividend Amount:")
print(f"Dividend Sensitivity: {put_dividend_sensitivity}")

#Problem 2
import pandas as pd
import numpy as np
from scipy.stats import norm

np.random.seed(42)  

daily_prices_df = pd.read_csv('DailyPrices.csv')
options_portfolio_df = pd.read_csv('problem2.csv')

current_aapl_price = 165
risk_free_rate = 0.0425
dividend_payment = 1.0
dividend_date = pd.to_datetime("2023-03-15")
current_date = pd.to_datetime("2023-03-03")
days_to_dividend = (dividend_date - current_date).days

daily_prices_df['Return'] = daily_prices_df['AAPL'].pct_change().dropna()
sigma = daily_prices_df['Return'].std()

n_simulations = 10000
simulated_returns = np.random.normal(0, sigma, (n_simulations, 10))
simulated_prices = current_aapl_price * np.exp(np.cumsum(simulated_returns, axis=1))

final_prices = simulated_prices[:, -1]
losses = current_aapl_price - final_prices

mean_loss = np.mean(losses)
var_95 = np.percentile(losses, 95)
es_95 = np.mean(losses[losses >= var_95])

delta = sigma * current_aapl_price
var_delta_normal = norm.ppf(0.95) * delta
es_delta_normal = delta * norm.pdf(norm.ppf(0.95)) / (1 - 0.95)

print(mean_loss, var_95, es_95, var_delta_normal, es_delta_normal)

#problem 3
import pandas as pd
import numpy as np
import scipy.optimize as sco
import statsmodels.api as sm

file_path_factors = 'F-F_Research_Data_Factors_daily.CSV'
file_path_momentum = 'F-F_Momentum_Factor_daily.CSV'
factors_df = pd.read_csv(file_path_factors)
momentum_df = pd.read_csv(file_path_momentum)

factors_df['Date'] = pd.to_datetime(factors_df['Date'], format='%Y%m%d')
momentum_df['Date'] = pd.to_datetime(momentum_df['Date'], format='%Y%m%d', errors='coerce')
momentum_df = momentum_df.dropna(subset=['Date'])

momentum_df.rename(columns=lambda x: x.strip(), inplace=True)
merged_df = pd.merge(factors_df, momentum_df, on='Date', how='inner')
merged_df.rename(columns=lambda x: x.strip(), inplace=True)

merged_df[['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']] = merged_df[['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']] / 100

stock_symbols = ['AAPL', 'META', 'UNH', 'MA', 'MSFT', 'NVDA', 'HD', 'PFE',
                 'AMZN', 'BRK-B', 'PG', 'XOM', 'TSLA', 'JPM', 'V', 'DIS',
                 'GOOGL', 'JNJ', 'BAC', 'CSCO']

dates = pd.date_range(start='1926-11-03', periods=10 * 252, freq='B')  
np.random.seed(0)
stock_returns = pd.DataFrame(data=np.random.normal(0, 0.01, (len(dates), len(stock_symbols))),
                             index=dates, columns=stock_symbols)

aligned_dates = stock_returns.index.intersection(merged_df['Date'])
stock_returns = stock_returns.loc[aligned_dates]
factor_returns = merged_df[merged_df['Date'].isin(aligned_dates)].set_index('Date')

def fit_four_factor_model(stock_return_series, factor_data):
    X = factor_data[['Mkt-RF', 'SMB', 'HML', 'Mom']]
    X = sm.add_constant(X) 
    excess_returns = stock_return_series - factor_data['RF']
    model = sm.OLS(excess_returns, X).fit()
    expected_excess_return = np.dot(model.params[1:], X.mean().values[1:])
    expected_annual_return = (expected_excess_return + factor_data['RF'].mean()) * 252 
    return expected_annual_return

expected_returns = {stock: fit_four_factor_model(stock_returns[stock], factor_returns) for stock in stock_symbols}
expected_returns_df = pd.DataFrame(list(expected_returns.items()), columns=['Stock', 'Expected Annual Return'])

annual_cov_matrix = stock_returns.cov() * 252  

expected_returns_vector = expected_returns_df['Expected Annual Return'].values
cov_matrix = annual_cov_matrix.values
risk_free_rate = 0.05  

def negative_sharpe_ratio(weights):
    portfolio_return = np.dot(weights, expected_returns_vector)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1) for _ in range(len(stock_symbols))]  

initial_weights = np.array([1 / len(stock_symbols)] * len(stock_symbols))

result = sco.minimize(negative_sharpe_ratio, initial_weights, bounds=bounds, constraints=constraints)

optimal_weights = result.x

optimal_return = np.dot(optimal_weights, expected_returns_vector)
optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

super_efficient_portfolio = pd.DataFrame({
    'Stock': stock_symbols,
    'Weight': optimal_weights
})

print("Super-Efficient Portfolio Weights:")
print(super_efficient_portfolio)
print(f"\nExpected Annual Return: {optimal_return}")
print(f"Portfolio Volatility: {optimal_volatility}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio}")

output_file_path = 'problem3_results.xlsx'
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    super_efficient_portfolio.to_excel(writer, sheet_name='Portfolio Weights', index=False)
    expected_returns_df.to_excel(writer, sheet_name='Expected Returns', index=False)
    pd.DataFrame(annual_cov_matrix, index=stock_symbols, columns=stock_symbols).to_excel(writer, sheet_name='Covariance Matrix')

print(f"\nResults saved to {output_file_path}")
