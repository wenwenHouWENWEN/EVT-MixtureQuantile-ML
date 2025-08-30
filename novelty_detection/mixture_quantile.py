# mixture_quantile.py
"""
Mixture quantile estimation module
----------------------------------
This module provides functions to:
1. Combine empirical quantile and GEVD-based quantile.
2. Output absolute value as novelty detection threshold.
"""

import numpy as np
import config


def mixture_quantile(emp_q, gevd_q, weight=0.5):
    """
    Compute mixture quantile.

    Parameters
    ----------
    emp_q : float
        Empirical quantile (from quantile.py).
    gevd_q : float
        GEVD-based quantile (from gevd_fit.py).
    weight : float
        Mixing weight in [0, 1].
        0 -> only empirical quantile
        1 -> only GEVD quantile
        default 0.5 means equal weighting.

    Returns
    -------
    q_mix : float
        Absolute value of the mixture quantile.
    """
    q_mix = (1 - weight) * emp_q + weight * gevd_q
    return np.abs(q_mix)


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    # Suppose from quantile.py and gevd_fit.py we got:
    emp_q = 1.8
    gevd_q = 2.5

    q_mix = mixture_quantile(emp_q, gevd_q, weight=0.5)
    print("Empirical quantile:", emp_q)
    print("GEVD quantile:", gevd_q)
    print("Mixture quantile (abs):", q_mix)
