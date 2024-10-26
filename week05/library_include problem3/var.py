import numpy as np
from scipy.stats import norm, t

class VaRCalculator:
    @staticmethod
    def parametric_var(returns, confidence_level=0.95):
        """Delta-Normal (Parametric) VaR."""
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = norm.ppf(1 - confidence_level)
        return -(mean + z_score * std)

    @staticmethod
    def t_distribution_var(returns, confidence_level=0.95):
        """VaR using T-distribution."""
        df, loc, scale = t.fit(returns)
        return t.ppf(1 - confidence_level, df, loc=loc, scale=scale)

    @staticmethod
    def historical_var(returns, confidence_level=0.95):
        """Historical simulation VaR."""
        sorted_returns = np.sort(returns)
        return -sorted_returns[int((1 - confidence_level) * len(sorted_returns))]
    
    @staticmethod
    def monte_carlo_var(returns, confidence_level=0.95, num_samples=10000):
        """Monte Carlo simulation for VaR."""
        simulated_returns = np.random.normal(np.mean(returns), np.std(returns), num_samples)
        return -np.percentile(simulated_returns, 100 * (1 - confidence_level))
