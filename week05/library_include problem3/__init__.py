
#problem 3

import pandas as pd
import numpy as np
from scipy.stats import t, norm, multivariate_normal
from var import VaRCalculator
from es import ESCalculator
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF

portfolio_df = pd.read_csv('portfolio.csv')
daily_prices_df = pd.read_csv('DailyPrices.csv')

daily_prices_df['Date'] = pd.to_datetime(daily_prices_df['Date']).dt.date
daily_prices_df.set_index('Date', inplace=True)
returns_df = daily_prices_df.pct_change().dropna()

portfolio_a = portfolio_df[portfolio_df['Portfolio'] == 'A']
portfolio_b = portfolio_df[portfolio_df['Portfolio'] == 'B']
portfolio_c = portfolio_df[portfolio_df['Portfolio'] == 'C']

returns_a = returns_df[portfolio_a['Stock']]
returns_b = returns_df[portfolio_b['Stock']]
returns_c = returns_df[portfolio_c['Stock']]

joint_returns = pd.concat([returns_a, returns_b, returns_c], axis=1)

marginals = []
for col in joint_returns.columns:
    if col in returns_a.columns or col in returns_b.columns:
        params = t.fit(joint_returns[col])
        marginals.append((t, params))
    else:
        mean = np.mean(joint_returns[col])
        std = np.std(joint_returns[col])
        marginals.append((norm, (mean, std)))
        
uniform_returns = np.zeros(joint_returns.shape)
for i, col in enumerate(joint_returns.columns):
    ecdf = ECDF(joint_returns[col])
    uniform_returns[:, i] = ecdf(joint_returns[col])

corr_matrix = np.corrcoef(uniform_returns.T)

num_simulations = 10000
scaler = StandardScaler()
simulated_normals = multivariate_normal.rvs(cov=corr_matrix, size=num_simulations)
simulated_uniforms = norm.cdf(simulated_normals)

simulated_joint_returns = np.zeros_like(simulated_uniforms)
for i, (dist, params) in enumerate(marginals):
    simulated_joint_returns[:, i] = dist.ppf(simulated_uniforms[:, i], *params)

weights_a = portfolio_a['Holding'].values / portfolio_a['Holding'].sum()
weights_b = portfolio_b['Holding'].values / portfolio_b['Holding'].sum()
weights_c = portfolio_c['Holding'].values / portfolio_c['Holding'].sum()

portfolio_a_returns = simulated_joint_returns[:, :len(weights_a)] @ weights_a
portfolio_b_returns = simulated_joint_returns[:, len(weights_a):len(weights_a) + len(weights_b)] @ weights_b
portfolio_c_returns = simulated_joint_returns[:, len(weights_a) + len(weights_b):] @ weights_c

confidence_level = 0.95

var_a = VaRCalculator.historical_var(portfolio_a_returns, confidence_level=confidence_level)
es_a = ESCalculator.historical_es(portfolio_a_returns, confidence_level=confidence_level)

var_b = VaRCalculator.historical_var(portfolio_b_returns, confidence_level=confidence_level)
es_b = ESCalculator.historical_es(portfolio_b_returns, confidence_level=confidence_level)

var_c = VaRCalculator.parametric_var(portfolio_c_returns, confidence_level=confidence_level)
es_c = ESCalculator.parametric_es(portfolio_c_returns, confidence_level=confidence_level)

total_var = var_a + var_b + var_c
total_es = es_a + es_b + es_c

print(f"Portfolio A VaR: {var_a}, ES: {es_a}")
print(f"Portfolio B VaR: {var_b}, ES: {es_b}")
print(f"Portfolio C VaR: {var_c}, ES: {es_c}")
print(f"Total VaR: {total_var}, Total ES: {total_es}")
