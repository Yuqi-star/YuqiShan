#problem 1 - library

#problem 2
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import minimize

file_path = 'problem1.csv'
data = pd.read_csv(file_path)

#a
confidence_level = 0.05
z_score_95 = 1.645 

returns = data['x']

lambda_factor = 0.97

weights = np.array([(1 - lambda_factor) * (lambda_factor ** i) for i in range(len(returns))])[::-1]
weighted_returns = returns * weights
ew_variance = np.var(weighted_returns, ddof=1)
ew_std_dev = np.sqrt(ew_variance)

VaR_normal = z_score_95 * ew_std_dev

ES_normal = -ew_std_dev * (np.exp(-0.5 * z_score_95**2) / (np.sqrt(2 * np.pi) * (1 - confidence_level)))

#b
def t_log_likelihood(params, data):
    df, loc, scale = params
    return -np.sum(t.logpdf(data, df=df, loc=loc, scale=scale))

initial_params = [10, np.mean(returns), np.std(returns)]

result = minimize(t_log_likelihood, initial_params, args=(returns,), bounds=[(2, None), (None, None), (0.0001, None)])
df_mle, loc_mle, scale_mle = result.x

VaR_t = t.ppf(confidence_level, df=df_mle, loc=loc_mle, scale=scale_mle)

ES_t = (df_mle + VaR_t**2) / (df_mle - 1) * t.pdf(VaR_t, df=df_mle, loc=loc_mle, scale=scale_mle)

#c
sorted_returns = returns.sort_values()
VaR_historical = sorted_returns.quantile(confidence_level)

ES_historical = sorted_returns[sorted_returns <= VaR_historical].mean()

print("VaR (Normal):", VaR_normal)
print("ES (Normal):", ES_normal)

print("VaR (MLE-fitted T-distribution):", VaR_t)
print("ES (MLE-fitted T-distribution):", ES_t)

print("VaR (Historical Simulation):", VaR_historical)
print("ES (Historical Simulation):", ES_historical)

#problem 3-please find it in library folder