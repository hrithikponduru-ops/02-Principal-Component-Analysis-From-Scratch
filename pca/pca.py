"""PCA: fit, project, reconstruct."""

from dataclasses import dataclass

import numpy as np

from .covariance import center, covariance
from .power_iteration import top_k_eigenpairs


@dataclass(frozen=True)
class PCA:
    mean: np.ndarray  # shape (d,)
    components: np.ndarray  # shape (d, k), orthonormal columns, descending variance
    explained_variance: np.ndarray  # shape (k,), the eigenvalues
    total_variance: float  # trace of the covariance matrix
    iterations: np.ndarray  # power-iteration steps per component

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.explained_variance / self.total_variance


def fit(X: np.ndarray, k: int, rng: np.random.Generator) -> PCA:
    """Find the k directions of greatest variance in X (n, d)."""
    X_centered, mean = center(X)
    C = covariance(X_centered)
    eig = top_k_eigenpairs(C, k, rng)
    return PCA(
        mean=mean,
        components=eig.vectors,
        explained_variance=eig.values,
        total_variance=float(np.trace(C)),
        iterations=eig.iterations,
    )


def transform(model: PCA, X: np.ndarray, k: int | None = None) -> np.ndarray:
    """Coordinates of X in the principal basis: (X - mean) V_k, shape (n, k)."""
    V = model.components if k is None else model.components[:, :k]
    return (X - model.mean) @ V


def inverse_transform(model: PCA, scores: np.ndarray) -> np.ndarray:
    """Map scores (n, k) back to the original space: scores V_k^T + mean."""
    k = scores.shape[1]
    return scores @ model.components[:, :k].T + model.mean


def reconstruction_error(model: PCA, X: np.ndarray, k: int) -> float:
    """Mean squared error per sample when keeping k components."""
    X_hat = inverse_transform(model, transform(model, X, k))
    return float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))