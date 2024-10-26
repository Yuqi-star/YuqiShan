import numpy as np
from scipy.stats import norm

class ESCalculator:
    @staticmethod
    def parametric_es(returns, confidence_level=0.95):
        """Parametric ES for normal distribution."""
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = norm.ppf(1 - confidence_level)
        pdf_value = norm.pdf(z_score)
        return -(mean + std * pdf_value / (1 - confidence_level))

    @staticmethod
    def historical_es(returns, confidence_level=0.95):
        """Historical ES."""
        sorted_returns = np.sort(returns)
        var_threshold = int((1 - confidence_level) * len(sorted_returns))
        return -np.mean(sorted_returns[:var_threshold])

    @staticmethod
    def monte_carlo_es(returns, confidence_level=0.95, num_samples=10000):
        """Monte Carlo ES."""
        simulated_returns = np.random.normal(np.mean(returns), np.std(returns), num_samples)
        var_value = np.percentile(simulated_returns, 100 * (1 - confidence_level))
        return -np.mean(simulated_returns[simulated_returns <= var_value])
