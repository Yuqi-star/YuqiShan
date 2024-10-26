import numpy as np
from scipy.stats import multivariate_normal

class SimulationMethods:
    @staticmethod
    def multivariate_normal_simulation(mean, cov, num_samples):
        """Simulate multivariate normal distribution."""
        return multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)
    
    @staticmethod
    def historical_simulation(returns, num_samples):
        """Resample historical returns with replacement."""
        return returns[np.random.choice(returns.shape[0], num_samples, replace=True), :]
