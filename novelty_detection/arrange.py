# arrange.py
"""
Arrange and negate distance values
----------------------------------
This module provides functions to:
1. Sort distance values.
2. Negate sorted distances (as required by GEVD fitting in the framework).
"""

import numpy as np


def arrange_and_negate(distances, ascending=True):
    """
    Sort distances and negate them.

    Parameters
    ----------
    distances : array-like
        Input distance values (1D array).
    ascending : bool
        If True, sort in ascending order (default).
        If False, sort in descending order.

    Returns
    -------
    arranged_neg : ndarray
        Sorted and negated distance values.
    """
    distances = np.asarray(distances).flatten()
    if ascending:
        arranged = np.sort(distances)       # ascending
    else:
        arranged = np.sort(distances)[::-1] # descending
    arranged_neg = -arranged
    return arranged_neg


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    distances = rng.uniform(low=0, high=10, size=10)

    print("Original distances:", distances)
    arranged_neg = arrange_and_negate(distances, ascending=True)
    print("Arranged & negated:", arranged_neg)
