#problem 1
import numpy as np

mu = 0
sigma = np.sqrt(0.01) 
P_t_minus_1 = 100  # Assume an initial price at t-1
num_simulations = 10000  

returns = np.random.normal(mu, sigma, num_simulations)

classical_brownian = P_t_minus_1 + returns
arithmetic_returns = P_t_minus_1 * (1 + returns)
log_returns = P_t_minus_1 * np.exp(returns)

classical_mean = np.mean(classical_brownian)
classical_std = np.std(classical_brownian)

arithmetic_mean = np.mean(arithmetic_returns)
arithmetic_std = np.std(arithmetic_returns)

log_mean = np.mean(log_returns)
log_std = np.std(log_returns)

print("Classical Brownian Motion:")
print("Expected Price (Mean):", classical_mean)
print("Standard Deviation of Price:", classical_std)

print("\nArithmetic Return System:")
print("Expected Price (Mean):", arithmetic_mean)
print("Standard Deviation of Price:", arithmetic_std)

print("\nLog Return or Geometric Brownian Motion:")
print("Expected Price (Mean):", log_mean)
print("Standard Deviation of Price:", log_std)

#problem 2
import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.ar_model import AutoReg

file_path = 'DailyPrices.csv'
prices_df = pd.read_csv(file_path)
prices_df['Date'] = pd.to_datetime(prices_df['Date'])

def return_calculate(prices, method='DISCRETE', date_column='Date'):
    if date_column not in prices.columns:
        raise ValueError(f"{date_column} not found in DataFrame")
    
    data = prices.drop(columns=[date_column])
    if method.upper() == 'DISCRETE':
        returns = data.div(data.shift(1)) - 1 
    elif method.upper() == 'LOG':
        returns = np.log(data.div(data.shift(1)))
    else:
        raise ValueError("Method must be 'DISCRETE' or 'LOG'")

    returns = pd.concat([prices[date_column].iloc[1:], returns], axis=1).dropna()
    return returns

returns_df = return_calculate(prices_df, method='DISCRETE')
output_path = 'Return_Results.xlsx'
returns_df.to_excel(output_path)

returns_df['META'] -= returns_df['META'].mean()

def calculate_var(returns, confidence_level=0.05, lambda_=0.94):
    mean_normal = np.mean(returns)
    std_normal = np.std(returns)
    var_normal = norm.ppf(confidence_level, mean_normal, std_normal)

    weights = np.array([(1-lambda_)*lambda_**i for i in range(len(returns))][::-1])
    weights /= weights.sum()
    mean_ew = np.average(returns, weights=weights)
    std_ew = np.sqrt(np.average((returns-mean_ew)**2, weights=weights))
    var_ew = norm.ppf(confidence_level, mean_ew, std_ew)

    params = t.fit(returns)
    var_t = t.ppf(confidence_level, *params)

    model = AutoReg(returns, lags=1)
    model_fit = model.fit()
    forecast = model_fit.predict(start=len(returns), end=len(returns), dynamic=False)
    var_ar = norm.ppf(confidence_level, forecast, std_normal)

    var_hs = np.percentile(returns, confidence_level*100)

    return {
        "Normal": var_normal,
        "EW Normal": var_ew,
        "T distribution": var_t,
        "AR(1)": var_ar,
        "Historical": var_hs
    }

meta_returns = returns_df['META'].dropna()
vars_meta = calculate_var(meta_returns)

print("Value at Risk (VaR) Estimates:")
for key, value in vars_meta.items():
    print(f"{key}: {value}")

#problem 3
import pandas as pd
import numpy as np

portfolio_df = pd.read_csv('portfolio.csv')
daily_prices_df = pd.read_csv('DailyPrices.csv', parse_dates=['Date'], index_col='Date')

available_stocks = set(daily_prices_df.columns).intersection(set(portfolio_df['Stock']))
portfolio_df = portfolio_df[portfolio_df['Stock'].isin(available_stocks)]

available_stocks_list = list(available_stocks)

daily_returns = daily_prices_df[available_stocks_list].pct_change().dropna()

lambda_value = 0.97

def calc_exponential_covariance(returns, lambda_value):
    mean_returns = returns.mean()
    demeaned_returns = returns - mean_returns
    weights = np.array([(1 - lambda_value) * (lambda_value ** i) for i in range(len(demeaned_returns))][::-1])
    covariance = np.cov(demeaned_returns.T, aweights=weights, bias=True)
    return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)

cov_matrix = calc_exponential_covariance(daily_returns, lambda_value)

def calculate_portfolio_var(portfolio_df, cov_matrix, confidence_level=0.95):
    z_score = np.abs(np.percentile(np.random.standard_normal(10000), (1 - confidence_level) * 100))
    portfolio_var = {}
    total_var = 0

    for portfolio_name in portfolio_df['Portfolio'].unique():
        portfolio = portfolio_df[portfolio_df['Portfolio'] == portfolio_name]
        weights = portfolio.set_index('Stock')['Holding'] / portfolio['Holding'].sum()
        port_variance = np.dot(weights.T, np.dot(cov_matrix.loc[weights.index, weights.index], weights))
        portfolio_var[portfolio_name] = z_score * np.sqrt(port_variance)
        total_var += portfolio_var[portfolio_name]

    return portfolio_var, total_var

portfolio_var, total_var = calculate_portfolio_var(portfolio_df, cov_matrix)

for portfolio, var in portfolio_var.items():
    print(f"VaR for {portfolio}: ${var:.4f}")
print(f"Total VaR of all portfolios: ${total_var:.4f}")

cov_matrix_simple = daily_returns.cov()
portfolio_var_simple, total_var_simple = calculate_portfolio_var(portfolio_df, cov_matrix_simple)

print("\nUsing Simple Covariance Model:")
for portfolio, var in portfolio_var_simple.items():
    print(f"VaR for {portfolio}: ${var:.4f}")
print(f"Total VaR of all portfolios: ${total_var_simple:.4f}")
