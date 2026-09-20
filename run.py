"""Fit PCA to MNIST with the hand-written eigensolver and save the figures.

Run from this folder: python run.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca.data import load_mnist
from pca.pca import fit, inverse_transform, reconstruction_error, transform

K = 100
SEED = 0
RECON_KS = (2, 10, 50, 100)
N_SCATTER = 5000

def main() -> None:
    rng = np.random.default_rng(SEED)
    X, y = load_mnist(train=True)

    t0 = time.perf_counter()
    model = fit(X, K, rng)
    print(f"fit {K} components in {time.perf_counter() - t0:.1f}s")
    print(f"power-iteration steps per component: min {model.iterations.min()}, "
          f"median {int(np.median(model.iterations))}, max {model.iterations.max()}")

    ratio = model.explained_variance_ratio
    cum = np.cumsum(ratio)
    for k in (1, 2, 10, 50, 100):
        print(f"k={k:>3} cumulative variance explained {cum[k - 1]:.1%} "
              f"| reconstruction MSE {reconstruction_error(model, X, k):.3f}")

    fig_explained_variance(ratio, cum)
    fig_eigen_digits(model)
    fig_reconstructions(model, X, y)
    fig_scatter(model, X, y, rng)
    fig_mean_digit(model)


def fig_explained_variance(ratio, cum):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.bar(np.arange(1, 31), ratio[:30], color="#2563eb")
    ax1.set_xlabel("component")
    ax1.set_ylabel("fraction of variance")
    ax1.set_title("Variance explained by each component (eigenvalue / trace)")
    ax2.plot(np.arange(1, len(cum) + 1), cum, color="#2563eb", lw=2)
    for level in (0.5, 0.8, 0.9):
        k = int(np.argmax(cum >= level)) + 1
        ax2.axhline(level, color="#9ca3af", lw=0.8, ls="--")
        ax2.annotate(f"{level:.0%}", xy=(k, level), xytext=(k + 8, level - 0.07), fontsize=9)
    ax2.set_xlabel("number of components k")
    ax2.set_ylabel("cumulative fraction")
    ax2.set_title("Cumulative variance explained")
    ax2.set_ylim(0, 1)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/explained_variance.png", dpi=150)
    plt.close(fig)

def fig_eigen_digits(model):
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(model.components[:, i].reshape(28, 28), cmap="RdBu_r",
                  vmin=-0.15, vmax=0.15)
        ax.set_title(f"PC {i + 1}\n{model.explained_variance_ratio[i]:.1%}", fontsize=9)
        ax.axis("off")
    fig.suptitle("The first 16 principal components, reshaped to 28x28 ('eigen-digits')", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/eigen_digits.png", dpi=150)
    plt.close(fig)

def fig_reconstructions(model, X, y):
    idx = [int(np.argmax(y == digit)) for digit in range(10)]
    rows = 1 + len(RECON_KS)
    fig, axes = plt.subplots(rows, 10, figsize=(12, 1.3 * rows))
    for col, i in enumerate(idx):
        axes[0, col].imshow(X[i].reshape(28, 28), cmap="gray")
        for row, k in enumerate(RECON_KS, start=1):
            X_hat = inverse_transform(model, transform(model, X[i : i + 1], k))
            axes[row, col].imshow(X_hat.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_ylabel("original", fontsize=9, rotation=0, ha="right", va="center")
    for row, k in enumerate(RECON_KS, start=1):
        axes[row, 0].set_ylabel(f"k = {k}", fontsize=9, rotation=0, ha="right", va="center")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Reconstruction from the first k principal components (784 pixels originally)", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/reconstructions.png", dpi=150)
    plt.close(fig)

def fig_scatter(model, X, y, rng):
    idx = rng.choice(len(X), N_SCATTER, replace=False)
    scores = transform(model, X[idx], k=2)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=y[idx], cmap="tab10", s=6, alpha=0.7)
    ax.legend(*sc.legend_elements(), title="digit", fontsize=8, loc="upper right")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"{N_SCATTER} digits projected from 784 dimensions to 2")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/scatter_2d.png", dpi=150)
    plt.close(fig)

def fig_mean_digit(model):
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(model.mean.reshape(28, 28), cmap="gray")
    ax.set_title("mean digit", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("figures/mean_digit.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()