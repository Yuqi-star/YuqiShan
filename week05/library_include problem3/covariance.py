import numpy as np

class CovarianceEstimator:
    @staticmethod
    def sample_covariance(returns):
        """Compute the sample covariance matrix."""
        return np.cov(returns, rowvar=False)

    @staticmethod
    def ewma_covariance(returns, lambda_factor=0.97):
        """Exponentially Weighted Moving Average (EWMA) covariance."""
        n = returns.shape[1]
        weights = np.array([(1 - lambda_factor) * lambda_factor ** i for i in range(returns.shape[0])])[::-1]
        weighted_cov = np.zeros((n, n))
        weighted_returns = returns * weights[:, np.newaxis]
        for i in range(n):
            for j in range(n):
                weighted_cov[i, j] = np.dot(weighted_returns[:, i], weighted_returns[:, j])
        return weighted_cov
