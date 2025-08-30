# data_loader.py
"""
Data loading and preprocessing module
-------------------------------------
This module provides functions to:
1. Load training and testing data from files.
2. Perform basic preprocessing (NaN removal, normalization).
3. Return numpy arrays for downstream modules.
"""

import os
import numpy as np
import config


def load_data(train_path=config.TRAIN_DATA_PATH,
              test_path=config.TEST_DATA_PATH,
              normalize=True):
    """
    Load training and testing data.

    Parameters
    ----------
    train_path : str
        Path to training data file (.npy or .csv).
    test_path : str
        Path to testing data file (.npy or .csv).
    normalize : bool
        If True, apply z-score normalization.

    Returns
    -------
    X_train : ndarray
        Training data, shape (n_samples, n_features).
    X_test : ndarray
        Testing data, shape (m_samples, n_features).
    """
    # Helper function to load single file
    def _load_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".csv"):
            return np.loadtxt(path, delimiter=",")
        else:
            raise ValueError(f"Unsupported file format: {path}")

    # Load data
    X_train = _load_file(train_path)
    X_test = _load_file(test_path)

    # Handle NaN values
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # Normalize (z-score)
    if normalize:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    return X_train, X_test


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    # Example: random synthetic data
    rng = np.random.default_rng(42)
    np.save("train.npy", rng.normal(size=(100, 5)))
    np.save("test.npy", rng.normal(size=(20, 5)))

    # Load and preprocess
    X_train, X_test = load_data("train.npy", "test.npy", normalize=True)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
