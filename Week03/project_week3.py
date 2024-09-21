import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file_path = 'DailyReturn.csv'
data = pd.read_csv(file_path)

#problem 1
def exponentially_weighted_covariance_matrix(returns, lambd):
    T, N = returns.shape
    weighted_cov_matrix = np.zeros((N, N))
    weights = np.array([lambd**(T-t) for t in range(1, T+1)])
    weights /= weights.sum()  

    weighted_mean = np.average(returns, axis=0, weights=weights)
    
    for i in range(N):
        for j in range(N):
            cov_ij = np.sum(weights * (returns[:, i] - weighted_mean[i]) * (returns[:, j] - weighted_mean[j]))
            weighted_cov_matrix[i, j] = cov_ij
            
    return weighted_cov_matrix

stock_returns = data.drop(columns=['SPY']).values

lambda_values = [0.94, 0.97, 0.99]

cumulative_variances = {}

for lambd in lambda_values:
    cov_matrix = exponentially_weighted_covariance_matrix(stock_returns, lambd)
    
    pca = PCA()
    pca.fit(cov_matrix)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    cumulative_variances[lambd] = cumulative_variance

plt.figure(figsize=(10, 6))
for lambd, cumulative_variance in cumulative_variances.items():
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=f'λ = {lambd}')
    
plt.title('Cumulative Variance Explained by Principal Components for Different λ Values')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.legend()
plt.grid(True)
plt.show()

#problem 2
import numpy as np
import time

def is_psd(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)

def chol_psd(matrix):
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        pass
    
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < 0] = 0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def near_psd(matrix, epsilon=1e-8):
    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_psd(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(matrix))
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_psd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def higham_psd(matrix, tol=1e-8):
    n = matrix.shape[0]
    Y = matrix.copy()
    for _ in range(100):  
        R = Y - np.minimum(0, np.linalg.eigvals(Y)).min() * np.eye(n)
        Y = (R + R.T) / 2
        if np.linalg.norm(Y - matrix, 'fro') < tol:
            break
    return Y

n = 500
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1)
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

chol_result = chol_psd(sigma)
near_psd_result = near_psd(sigma)
higham_result = higham_psd(sigma)

print(f"Cholesky PSD matrix is PSD: {is_psd(chol_result)}")
print(f"Near PSD matrix is PSD: {is_psd(near_psd_result)}")
print(f"Higham PSD matrix is PSD: {is_psd(higham_result)}")

frobenius_chol = np.linalg.norm(chol_result - sigma, 'fro')
frobenius_near_psd = np.linalg.norm(near_psd_result - sigma, 'fro')
frobenius_higham = np.linalg.norm(higham_result - sigma, 'fro')

print(f"Frobenius Norm (Cholesky): {frobenius_chol}")
print(f"Frobenius Norm (Near PSD): {frobenius_near_psd}")
print(f"Frobenius Norm (Higham): {frobenius_higham}")

start_time = time.time()
chol_result = chol_psd(sigma)
chol_runtime = time.time() - start_time

start_time = time.time()
near_psd_result = near_psd(sigma)
near_psd_runtime = time.time() - start_time

start_time = time.time()
higham_result = higham_psd(sigma)
higham_runtime = time.time() - start_time

print(f"Cholesky Runtime: {chol_runtime} seconds")
print(f"Near PSD Runtime: {near_psd_runtime} seconds")
print(f"Higham Runtime: {higham_runtime} seconds")

#problem 3
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
import time

data = pd.read_csv('DailyReturn.csv')
returns = data.values

pearson_corr = np.corrcoef(returns, rowvar=False)
pearson_var = np.var(returns, axis=0)

lambda_ = 0.97
ewma_corr = np.corrcoef(returns, rowvar=False)
ewma_var = np.var(returns, axis=0)

for i in range(1, len(returns)):
    ewma_corr = lambda_ * ewma_corr + (1 - lambda_) * np.corrcoef(returns[:i+1], rowvar=False)
    ewma_var = lambda_ * ewma_var + (1 - lambda_) * np.var(returns[:i+1], axis=0)

cov_pearson_pearson = np.diag(pearson_var) @ pearson_corr @ np.diag(pearson_var)
cov_pearson_ewma = np.diag(ewma_var) @ pearson_corr @ np.diag(ewma_var)
cov_ewma_pearson = np.diag(pearson_var) @ ewma_corr @ np.diag(pearson_var)
cov_ewma_ewma = np.diag(ewma_var) @ ewma_corr @ np.diag(ewma_var)

cov_matrices = [cov_pearson_pearson, cov_pearson_ewma, cov_ewma_pearson, cov_ewma_ewma]
labels = ['Pearson-Pearson', 'Pearson-EWMA', 'EWMA-Pearson', 'EWMA-EWMA']

n_draws = 25000
results = []

for i, cov_matrix in enumerate(cov_matrices):
    start_time = time.time()
    simulated_data_direct = np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix, size=n_draws)
    runtime_direct = time.time() - start_time
    cov_direct = np.cov(simulated_data_direct, rowvar=False)
    frobenius_direct = np.linalg.norm(cov_direct - cov_matrix, 'fro')
    
    pca = PCA()
    pca.fit(cov_matrix)
    
    start_time = time.time()
    pca_100 = pca.transform(np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix, size=n_draws))
    pca_100_inv = pca.inverse_transform(pca_100)
    runtime_pca_100 = time.time() - start_time
    cov_pca_100 = np.cov(pca_100_inv, rowvar=False)
    frobenius_pca_100 = np.linalg.norm(cov_pca_100 - cov_matrix, 'fro')
    
    pca_75 = PCA(n_components=int(0.75 * cov_matrix.shape[0]))
    pca_75.fit(cov_matrix)
    start_time = time.time()
    pca_75_data = pca_75.transform(np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix, size=n_draws))
    pca_75_inv = pca_75.inverse_transform(pca_75_data)
    runtime_pca_75 = time.time() - start_time
    cov_pca_75 = np.cov(pca_75_inv, rowvar=False)
    frobenius_pca_75 = np.linalg.norm(cov_pca_75 - cov_matrix, 'fro')
    
    pca_50 = PCA(n_components=int(0.5 * cov_matrix.shape[0]))
    pca_50.fit(cov_matrix)
    start_time = time.time()
    pca_50_data = pca_50.transform(np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix, size=n_draws))
    pca_50_inv = pca_50.inverse_transform(pca_50_data)
    runtime_pca_50 = time.time() - start_time
    cov_pca_50 = np.cov(pca_50_inv, rowvar=False)
    frobenius_pca_50 = np.linalg.norm(cov_pca_50 - cov_matrix, 'fro')
    
    results.append({
        'Label': labels[i],
        'Frobenius Direct': frobenius_direct,
        'Runtime Direct': runtime_direct,
        'Frobenius PCA 100%': frobenius_pca_100,
        'Runtime PCA 100%': runtime_pca_100,
        'Frobenius PCA 75%': frobenius_pca_75,
        'Runtime PCA 75%': runtime_pca_75,
        'Frobenius PCA 50%': frobenius_pca_50,
        'Runtime PCA 50%': runtime_pca_50
    })

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv('simulation_results.csv', index=False)
