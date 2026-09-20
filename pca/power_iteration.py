"""A hand-written eigensolver for symmetric positive semi-definite matrices.

Power iteration finds the eigenvector with the largest eigenvalue by
repeatedly multiplying a vector by the matrix and renormalising. Deflation
then removes that direction so the next-largest can be found the same way.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenResult:
    values: np.ndarray  # shape (k,), descending
    vectors: np.ndarray  # shape (d, k), unit-length columns
    iterations: np.ndarray  # shape (k,), iterations used per eigenpair


def power_iteration(
    A: np.ndarray,
    rng: np.random.Generator,
    orthogonal_to: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[float, np.ndarray, int]:
    """Return (lambda_1, v_1, iterations) for the dominant eigenpair of A.

    Start from a random unit vector v. Each step computes A v and rescales
    to unit length. Writing v in the eigenbasis, v = sum_i c_i u_i, gives
    A^t v = sum_i c_i lambda_i^t u_i, so after many steps the term with the
    largest |lambda_i| dominates. Convergence is geometric with ratio
    |lambda_2 / lambda_1|: close eigenvalues mean slow convergence.

    orthogonal_to: (d, m) matrix Q of already-found eigenvectors. Each step
    projects them out of the iterate (one Gram-Schmidt sweep). Deflation
    alone makes those directions have eigenvalue 0, but floating-point
    rounding re-introduces tiny components along them every multiply, and
    for slowly converging eigenpairs those components grow into visible
    non-orthogonality. Projecting them out explicitly fixes that.

    The eigenvalue estimate is the Rayleigh quotient v^T A v, which for a
    unit vector is exactly the variance of the data along v.

    Stopping rule: v is an eigenvector exactly when A v = lambda v, so the
    residual ||A v - lambda v|| measures how far v is from being one. Stop
    when it is below tol relative to lambda. (Comparing successive iterates
    instead saturates around 1e-8 accuracy because 1 - cos(angle) is
    quadratic in the angle and hits machine epsilon early.)
    """
    d = A.shape[0]
    v = rng.normal(size=d)
    if orthogonal_to is not None:
        v = v - orthogonal_to @ (orthogonal_to.T @ v)
    v /= np.linalg.norm(v)
    eigenvalue = 0.0
    for it in range(1, max_iter + 1):
        w = A @ v
        if orthogonal_to is not None:
            w = w - orthogonal_to @ (orthogonal_to.T @ w)
        eigenvalue = float(v @ w)  # Rayleigh quotient, since ||v|| = 1
        residual = np.linalg.norm(w - eigenvalue * v)
        if residual <= tol * max(abs(eigenvalue), np.finfo(float).tiny):
            break
        w_norm = np.linalg.norm(w)
        if w_norm == 0.0:
            return 0.0, v, it
        v = w / w_norm
    return eigenvalue, v, it


def deflate(A: np.ndarray, eigenvalue: float, v: np.ndarray) -> np.ndarray:
    """Remove one eigen-direction: A' = A - lambda v v^T.

    For symmetric A with orthonormal eigenvectors, A = sum_i lambda_i u_i u_i^T.
    Subtracting the lambda_1 term leaves a matrix with the same eigenvectors
    but eigenvalue 0 in place of lambda_1, so lambda_2 is now dominant.
    """
    return A - eigenvalue * np.outer(v, v)


def top_k_eigenpairs(
    A: np.ndarray, k: int, rng: np.random.Generator, tol: float = 1e-12, max_iter: int = 10_000
) -> EigenResult:
    """Largest k eigenpairs of symmetric PSD A by power iteration + deflation."""
    values, vectors, iterations = [], [], []
    A_current = A
    for _ in range(k):
        Q = np.column_stack(vectors) if vectors else None
        lam, v, it = power_iteration(A_current, rng, Q, tol, max_iter)
        values.append(lam)
        vectors.append(v)
        iterations.append(it)

        A_current = deflate(A_current, lam, v)
    return EigenResult(
        values=np.array(values),
        vectors=np.column_stack(vectors),
        iterations=np.array(iterations),
    )