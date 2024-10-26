import numpy as np

class CorrelationFixer:
    @staticmethod
    def nearest_psd(matrix):
        """Find the nearest positive semi-definite matrix."""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = 0
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
