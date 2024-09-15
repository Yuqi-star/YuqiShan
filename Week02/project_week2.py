import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

data = pd.read_csv('problem1.csv')

#Problem 1
mean_manual = data['x'].mean()
variance_manual = data['x'].var(ddof=1)  
std_dev = np.sqrt(variance_manual)
skewness_manual = (data['x'] - mean_manual).apply(lambda x: x**3).sum() / ((len(data) - 1) * std_dev**3)
kurtosis_manual = (data['x'] - mean_manual).apply(lambda x: x**4).sum() / ((len(data) - 1) * std_dev**4) - 3

print(f"Mean: {mean_manual}")
print(f"Variance: {variance_manual}")
print(f"Skewness: {skewness_manual}")
print(f"Kurtosis: {kurtosis_manual}")

skewness_package = skew(data['x'], bias=False)
kurtosis_package = kurtosis(data['x'], bias=False) 

print("Skewness calculated by the package:", skewness_package)
print("Kurtosis calculated by the package:", kurtosis_package)

#Problem 2
import statsmodels.api as sm
from scipy.stats import norm, t

data = pd.read_csv('problem2.csv')

X = sm.add_constant(data['x'])  
y = data['y']

ols_model = sm.OLS(y, X).fit()

print(ols_model.summary())

def neg_log_likelihood(params):
    intercept, beta, sigma = params[0], params[1], params[2]
    y_pred = intercept + beta * data['x']
    return -np.sum(norm.logpdf(data['y'], loc=y_pred, scale=sigma))

from scipy.optimize import minimize

initial_guesses = [0, 0, 1]

mle_results = minimize(neg_log_likelihood, initial_guesses, method='L-BFGS-B', bounds=[(None, None), (None, None), (0.01, None)])

print(f"MLE Estimates: Intercept = {mle_results.x[0]}, Beta = {mle_results.x[1]}, Sigma = {mle_results.x[2]}")

print(f"OLS Beta: {ols_model.params['x']}, MLE Beta: {mle_results.x[1]}")
print(f"Standard Deviation of OLS errors: {np.std(ols_model.resid)}, MLE Sigma: {mle_results.x[2]}")

def mle_t_distribution(params):
    intercept, slope, sigma, nu = params
    y_pred = intercept + slope * data['x']
    likelihood = -t.logpdf(data['y'], df=nu, loc=y_pred, scale=sigma).sum()
    return likelihood

initial_params_t = [0, 0, 1, 10]  
bounds_t = [(None, None), (None, None), (0, None), (2, None)] 

mle_results_t = minimize(mle_t_distribution, initial_params_t, method='L-BFGS-B', bounds=bounds_t)

mle_params_t = mle_results_t.x
mle_intercept_t, mle_slope_t, mle_sigma_t, mle_nu_t = mle_params_t

print(f"Intercept (β₀): {mle_intercept_t}")
print(f"Slope (β₁): {mle_slope_t}")
print(f"Scale (σ): {mle_sigma_t}")
print(f"Degrees of Freedom (ν): {mle_nu_t}")

from scipy.special import gammaln

x = data['x']
y = data['y']
n = len(y)  

beta0_normal, beta1_normal, sigma_normal = -0.08738447443979024, 0.7752741738871116, 1.003756269318408

beta0_t, beta1_t, sigma_t, nu_t = -0.09726796971842995, 0.6750102349179323, 0.8551027650098203, 7.159886081005919

log_likelihood_normal = -n/2 * np.log(2 * np.pi * sigma_normal**2) - (1/(2 * sigma_normal**2)) * np.sum((y - (beta0_normal + beta1_normal * x))**2)

log_likelihood_t = np.sum(
    gammaln((nu_t + 1) / 2) - gammaln(nu_t / 2) - 0.5 * np.log(nu_t * np.pi * sigma_t**2) -
    ((nu_t + 1) / 2) * np.log(1 + (1 / nu_t) * ((y - (beta0_t + beta1_t * x)) / sigma_t)**2)
)

print("Log-Likelihood for Normal Distribution:", log_likelihood_normal)
print("Log-Likelihood for T Distribution:", log_likelihood_t)

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

data = pd.read_csv('problem2_x.csv')
mean_vector = data.mean()
covariance_matrix = data.cov()

mu_x1, mu_x2 = mean_vector['x1'], mean_vector['x2']
sigma_x1, sigma_x2 = covariance_matrix.at['x1', 'x1'], covariance_matrix.at['x2', 'x2']
sigma_x1x2 = covariance_matrix.at['x1', 'x2']

data['conditional_mean_x2'] = mu_x2 + (sigma_x1x2 / sigma_x1) * (data['x1'] - mu_x1)
conditional_variance_x2 = sigma_x2 - (sigma_x1x2**2 / sigma_x1)

z_value = 1.96  
data['lower_bound_x2'] = data['conditional_mean_x2'] - z_value * np.sqrt(conditional_variance_x2)
data['upper_bound_x2'] = data['conditional_mean_x2'] + z_value * np.sqrt(conditional_variance_x2)

data_sorted = data.sort_values(by='x1')

x1_sorted = data_sorted['x1'].values
conditional_mean_x2_sorted = data_sorted['conditional_mean_x2'].values
lower_bound_x2_sorted = data_sorted['lower_bound_x2'].values
upper_bound_x2_sorted = data_sorted['upper_bound_x2'].values

plt.figure(figsize=(10, 6))
plt.scatter(x1_sorted, data_sorted['x2'], color='blue', label='Observed $X_2$')
plt.plot(x1_sorted, conditional_mean_x2_sorted, 'r-', label='Expected $X_2$')
plt.fill_between(x1_sorted, lower_bound_x2_sorted, upper_bound_x2_sorted, color='gray', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Conditional Distribution of $X_2$ given $X_1$')
plt.legend()
plt.show()

#Problem 3
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('problem3.csv')

plt.figure(figsize=(12, 4))
plt.plot(data['x'], label='Time Series')
plt.title('Time Series Data')
plt.xlabel('Time Index')
plt.ylabel('Values')
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 4))

plot_acf(data['x'], ax=ax[0], lags=40, alpha=0.05)
ax[0].set_title('Autocorrelation Function')

plot_pacf(data['x'], ax=ax[1], lags=40, alpha=0.05, method='ywm')
ax[1].set_title('Partial Autocorrelation Function')

plt.show()

def fit_arima(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.aic

ar_orders = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
ma_orders = [(0, 0, 1), (0, 0, 2), (0, 0, 3)]

aic_values = {}
for order in ar_orders + ma_orders:
    aic_values[order] = fit_arima(data['x'], order)

for order, aic in aic_values.items():
    print(f"Model {order}: AIC = {aic:.2f}")