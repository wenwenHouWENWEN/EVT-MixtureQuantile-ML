# quantile.py
"""
Empirical quantile estimation module
------------------------------------
This module provides functions to:
1. Compute empirical quantiles of a dataset.
2. Extract top-k extreme values for GEVD fitting.
"""

import numpy as np
import config


def empirical_quantile(data, alpha=config.ALPHA):
    """
    Compute empirical quantile at level (1 - alpha).

    Parameters
    ----------
    data : array-like
        Input 1D data (e.g., distances).
    alpha : float
        Significance level (e.g., 0.05).

    Returns
    -------
    q : float
        Empirical quantile value.
    """
    data = np.sort(np.asarray(data))
    n = len(data)
    # position of quantile (nearest-rank method)
    k = int(np.ceil((1 - alpha) * n)) - 1
    k = np.clip(k, 0, n - 1)
    return data[k]


def top_k_extremes(data, k):
    """
    Extract top-k extreme values (largest values).

    Parameters
    ----------
    data : array-like
        Input 1D data (e.g., negated distances).
    k : int
        Number of extreme values to extract.

    Returns
    -------
    extremes : ndarray
        Top-k extreme values.
    """
    data = np.sort(np.asarray(data))
    return data[-k:]


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    values = rng.normal(size=100)

    # Empirical 95% quantile
    q95 = empirical_quantile(values, alpha=0.05)
    print("Empirical 95% quantile:", q95)

    # Top 10 extremes
    extremes = top_k_extremes(values, k=10)
    print("Top-10 extreme values:", extremes)
