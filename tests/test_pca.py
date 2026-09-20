"""Verify the hand-written eigensolver and PCA against NumPy's LAPACK routines.

Each test encodes a mathematical fact that must hold if the derivation in
README.md is correct. If any of them fails, the code may still produce
plausible-looking pictures while being wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pca.covariance import center, covariance  # noqa: E402
from pca.pca import fit, inverse_transform, reconstruction_error, transform  # noqa: E402
from pca.power_iteration import top_k_eigenpairs  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def random_data(rng):
    """200 samples in 12 dimensions with a clear low-rank structure."""
    latent = rng.normal(size=(200, 4)) * np.array([5.0, 3.0, 2.0, 1.0])
    mixing = rng.normal(size=(4, 12))
    return latent @ mixing + 0.05 * rng.normal(size=(200, 12))


def _same_up_to_sign(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    """Eigenvectors are defined only up to sign, so v and -v are both correct."""
    return np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol)


def test_power_iteration_matches_lapack_eigenvalues(random_data, rng):
    _, C = _centered_cov(random_data)
    ours = top_k_eigenpairs(C, k=6, rng=rng)
    reference = np.sort(np.linalg.eigvalsh(C))[::-1][:6]
    np.testing.assert_allclose(ours.values, reference, rtol=1e-8)


def test_power_iteration_matches_lapack_eigenvectors(random_data, rng):
    _, C = _centered_cov(random_data)
    ours = top_k_eigenpairs(C, k=4, rng=rng)
    ref_vals, ref_vecs = np.linalg.eigh(C)
    ref_vecs = ref_vecs[:, ::-1]
    for i in range(4):
        assert _same_up_to_sign(ours.vectors[:, i], ref_vecs[:, i], atol=1e-6)


def test_eigenvalues_are_descending_and_nonnegative(random_data, rng):
    _, C = _centered_cov(random_data)
    ours = top_k_eigenpairs(C, k=6, rng=rng)
    assert np.all(np.diff(ours.values) <= 1e-12)
    assert np.all(ours.values >= -1e-12)


def test_components_are_orthonormal(random_data, rng):
    model = fit(random_data, k=6, rng=rng)
    gram = model.components.T @ model.components
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-8)


def test_pca_matches_svd(random_data, rng):
    """PCA via covariance eigenvectors equals PCA via SVD of the centered data."""
    model = fit(random_data, k=4, rng=rng)
    X_centered, _ = center(random_data)
    _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n = random_data.shape[0]
    np.testing.assert_allclose(model.explained_variance, s[:4] ** 2 / (n - 1), rtol=1e-8)
    for i in range(4):
        assert _same_up_to_sign(model.components[:, i], Vt[i], atol=1e-6)


def test_projected_coordinates_are_uncorrelated(random_data, rng):
    """The covariance of the scores is diagonal with the eigenvalues on it."""
    model = fit(random_data, k=4, rng=rng)
    scores = transform(model, random_data)
    score_cov = covariance(scores)
    np.testing.assert_allclose(score_cov, np.diag(model.explained_variance), atol=1e-8)


def test_full_rank_reconstruction_is_exact(random_data, rng):
    d = random_data.shape[1]
    model = fit(random_data, k=d, rng=rng)
    X_hat = inverse_transform(model, transform(model, random_data))
    np.testing.assert_allclose(X_hat, random_data, atol=1e-8)


def test_reconstruction_error_equals_dropped_eigenvalues(random_data, rng):
    """MSE with k components = sum of eigenvalues k+1..d (scaled by (n-1)/n)."""
    d, n = random_data.shape[1], random_data.shape[0]
    model = fit(random_data, k=d, rng=rng)
    for k in (1, 3, 7):
        predicted = model.explained_variance[k:].sum() * (n - 1) / n
        np.testing.assert_allclose(reconstruction_error(model, random_data, k), predicted, rtol=1e-8)


def test_explained_variance_sums_to_total(random_data, rng):
    d = random_data.shape[1]
    model = fit(random_data, k=d, rng=rng)
    np.testing.assert_allclose(model.explained_variance.sum(), model.total_variance, rtol=1e-10)


def _centered_cov(X):
    Xc, _ = center(X)
    return Xc, covariance(Xc)