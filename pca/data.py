"""Load MNIST as flat NumPy arrays. Shares the download cache with project 01."""

from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

DATA_ROOT = Path(__file__).resolve().parents[2] / ".data"


def load_mnist(train: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with X of shape (n, 784) in [0, 1] and y of shape (n,)."""
    dataset = MNIST(root=str(DATA_ROOT), train=train, download=True)
    X = dataset.data.numpy().reshape(len(dataset), -1).astype(np.float64) / 255.0
    y = dataset.targets.numpy().astype(np.int64)
    return X, y