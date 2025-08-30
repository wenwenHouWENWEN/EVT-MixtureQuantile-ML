# gevd_fit.py
"""
GEVD fitting and parameter selection module
-------------------------------------------
This module provides functions to:
1. Fit Generalized Extreme Value Distribution (GEVD).
2. Select optimal tail size k using goodness-of-fit.
3. Estimate quantiles from the fitted model.
"""

import numpy as np
from scipy.stats import genextreme, kstest
import config


def fit_gevd(data):
    """
    Fit GEVD to data using MLE.

    Parameters
    ----------
    data : array-like
        Input extreme values (1D).

    Returns
    -------
    params : dict
        Fitted parameters {shape, loc, scale}.
    """
    data = np.asarray(data)
    shape, loc, scale = genextreme.fit(data)
    return {"shape": shape, "loc": loc, "scale": scale}


def gevd_quantile(params, alpha=config.ALPHA):
    """
    Estimate quantile from fitted GEVD.

    Parameters
    ----------
    params : dict
        GEVD parameters {shape, loc, scale}.
    alpha : float
        Significance level.

    Returns
    -------
    q : float
        Quantile at (1 - alpha).
    """
    return genextreme.ppf(1 - alpha,
                          c=params["shape"],
                          loc=params["loc"],
                          scale=params["scale"])


def select_k_and_fit(data, k_min=config.K_MIN, k_max=config.K_MAX, step=5):
    """
    Try multiple k values, fit GEVD, and select the best using KS test.

    Parameters
    ----------
    data : array-like
        Input negated & sorted distances (1D).
    k_min : int
        Minimum number of tail samples.
    k_max : int
        Maximum number of tail samples.
    step : int
        Step size for k search.

    Returns
    -------
    best_k : int
        Chosen number of tail samples.
    best_params : dict
        Best fitted GEVD parameters.
    best_stat : float
        KS statistic (smaller is better).
    """
    data = np.asarray(data)
    n = len(data)

    best_k, best_params, best_stat = None, None, np.inf

    for k in range(k_min, min(k_max, n) + 1, step):
        tail = data[-k:]  # pick top-k extremes
        params = fit_gevd(tail)

        # Compute KS test statistic
        c, loc, scale = params["shape"], params["loc"], params["scale"]
        D, _ = kstest(tail, 'genextreme', args=(c, loc, scale))

        if D < best_stat:
            best_k, best_params, best_stat = k, params, D

    return best_k, best_params, best_stat


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    synthetic = genextreme.rvs(c=0.1, loc=0, scale=1, size=500, random_state=rng)

    # Sort & take negatives (simulate arrange.py output)
    arranged_neg = -np.sort(synthetic)

    # Select best k and fit
    best_k, best_params, best_stat = select_k_and_fit(arranged_neg, 20, 100, 10)
    print("Best k:", best_k)
    print("Best params:", best_params)
    print("Best KS stat:", best_stat)

    # Estimate quantile
    q95 = gevd_quantile(best_params, alpha=0.05)
    print("GEVD 95% quantile:", q95)
