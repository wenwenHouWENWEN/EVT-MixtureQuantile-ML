# distance.py
"""
Distance computation module
---------------------------
This module provides functions to:
1. Compute pairwise distances within training data.
2. Compute distances between test samples and training data.
Supported metrics: euclidean, manhattan, cosine.
"""

import numpy as np
import config


def compute_distance_matrix(X, Y=None, metric=config.DISTANCE_METRIC):
    """
    Compute distance matrix between two datasets.

    Parameters
    ----------
    X : ndarray
        Data matrix (n_samples, n_features).
    Y : ndarray or None
        If None, compute pairwise distances within X.
        If ndarray, compute distances between X and Y.
    metric : str
        Distance metric: "euclidean", "manhattan", "cosine".

    Returns
    -------
    D : ndarray
        Distance matrix, shape (n_samples_X, n_samples_Y).
    """
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)

    if metric == "euclidean":
        # ||x - y||_2
        D = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))

    elif metric == "manhattan":
        # ||x - y||_1
        D = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)

    elif metric == "cosine":
        # 1 - cos similarity
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        D = 1 - np.dot(X_norm, Y_norm.T)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return D


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(5, 3))
    X_test = rng.normal(size=(2, 3))

    # Distances within training set
    D_train = compute_distance_matrix(X_train, metric="euclidean")
    print("Training pairwise distances:\n", D_train)

    # Distances between test and train
    D_test = compute_distance_matrix(X_test, X_train, metric="euclidean")
    print("Test-to-train distances:\n", D_test)
