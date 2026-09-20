"""Centering and the sample covariance matrix."""

import numpy as np


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the column means. Returns (X_centered, mean)."""
    mean = X.mean(axis=0)
    return X - mean, mean


def covariance(X_centered: np.ndarray) -> np.ndarray:
    """C = (1 / (n - 1)) X^T X for already-centered X of shape (n, d).

    C_jk is the sample covariance between feature j and feature k.
    The diagonal holds the variance of each feature, so trace(C) is the
    total variance in the data. C is symmetric and positive semi-definite,
    which is what guarantees real, non-negative eigenvalues and orthogonal
    eigenvectors.
    """
    n = X_centered.shape[0]
    return X_centered.T @ X_centered / (n - 1)