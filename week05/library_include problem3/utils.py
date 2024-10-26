import numpy as np

class Utils:
    @staticmethod
    def random_correlation_matrix(size):
        """Generate a random correlation matrix."""
        A = np.random.rand(size, size)
        P = np.dot(A, A.T)
        D = np.diag(1 / np.sqrt(np.diag(P)))
        return D @ P @ D
