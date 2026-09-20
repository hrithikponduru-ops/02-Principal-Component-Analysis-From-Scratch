"""PCA on five 2-D points, every number printed so it can be checked by hand.

Run from this folder: python worked_example.py
"""

import numpy as np

from pca.covariance import center, covariance
from pca.power_iteration import power_iteration

np.set_printoptions(precision=4, suppress=True)

X = np.array([[2.0, 1.0], [3.0, 3.0], [4.0, 3.0], [5.0, 5.0], [6.0, 5.0]])

def show(name, value):
    print(f"{name:<22}= {np.array2string(np.asarray(value), prefix=' ' * 24)}")

print("=== DATA ===")
show("X (5 points, 2 dims)", X)

print("\n=== STEP 1: CENTER ===")
Xc, mean = center(X)
show("mean", mean)
show("X - mean", Xc)

print("\n=== STEP 2: COVARIANCE  C = Xc^T Xc / (n-1) ===")
C = covariance(Xc)
show("C", C)
print(f"{'total variance':<22}= trace(C) = {np.trace(C):.4f}")

print("\n=== STEP 3: EIGENVECTORS BY HAND ===")
a, b, d = C[0, 0], C[0, 1], C[1, 1]
tr, det = a + d, a * d - b * b
disc = np.sqrt(tr**2 / 4 - det)
lam1, lam2 = tr / 2 + disc, tr / 2 - disc
print(f"characteristic polynomial: lambda^2 - {tr:.4f} lambda + {det:.4f} = 0")
print(f"lambda_1 = {lam1:.4f}   lambda_2 = {lam2:.4f}   (sum = {lam1 + lam2:.4f} = trace, as it must)")
v1 = np.array([b, lam1 - a])
v1 /= np.linalg.norm(v1)
show("v_1 (solve (C - lambda_1 I) v = 0)", v1)

print("\n=== STEP 4: SAME ANSWER BY POWER ITERATION ===")
rng = np.random.default_rng(0)
lam_pi, v_pi, its = power_iteration(C, rng)
print(f"lambda_1 = {lam_pi:.4f}   after {its} iterations")
show("v_pi", v_pi)
print(f"{'agreement':<22}= |v_hand . v_power| = {abs(v1 @ v_pi):.6f}  (1 means identical up to sign)")

print("\n=== STEP 5: PROJECT ONTO v_1 ===")
scores = Xc @ v_pi
show("1-D coordinates", scores)
print(f"{'variance of scores':<22}= {scores.var(ddof=1):.4f}  (= lambda_1: the eigenvalue IS the variance along v_1)")
X_hat = np.outer(scores, v_pi) + mean
show("reconstruction", X_hat)
err = np.mean(np.sum((X - X_hat) ** 2, axis=1))
print(f"{'mean sq. error':<22}= {err:.4f}  (= lambda_2 * (n-1)/n = {lam2 * 4 / 5:.4f}: what you drop is what you lose)")
print(f"{'variance kept':<22}= {lam1 / (lam1 + lam2):.1%}")