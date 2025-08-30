# main_test.py
"""
Testing phase (Framework B)
---------------------------
Pipeline:
1. Load test data
2. Compute distances between test samples and training data
3. Arrange and negate distances
4. Empirical quantile estimation
5. GEVD fitting with k-selection
6. Mixture quantile estimation
7. Compare with training threshold for novelty detection
"""

import numpy as np
import config
from data_loader import load_data
from distance import compute_distance_matrix
from arrange import arrange_and_negate
from quantile import empirical_quantile
from gevd_fit import select_k_and_fit, gevd_quantile
from mixture_quantile import mixture_quantile


def test_pipeline(train_threshold=None):
    """
    Full testing pipeline (Framework B).

    Parameters
    ----------
    train_threshold : float
        Final mixture quantile from training phase.

    Returns
    -------
    results : dict
        Dictionary containing novelty scores and decisions.
    """
    # 1. Load training and test data
    X_train, X_test = load_data()

    # 2. Compute distances between test and train
    D_test = compute_distance_matrix(X_test, X_train)
    distances = D_test.flatten()

    # 3. Arrange & negate distances
    arranged_neg = arrange_and_negate(distances, ascending=True)

    # 4. Empirical quantile
    emp_q = empirical_quantile(distances, alpha=config.ALPHA)

    # 5. GEVD fitting with k-selection
    best_k, best_params, best_stat = select_k_and_fit(arranged_neg,
                                                     k_min=config.K_MIN,
                                                     k_max=config.K_MAX,
                                                     step=config.K_STEP)
    gevd_q = gevd_quantile(best_params, alpha=config.ALPHA)

    # 6. Mixture quantile
    final_q = mixture_quantile(emp_q, gevd_q, weight=0.5)

    # 7. Novelty decision (if threshold provided)
    if train_threshold is not None:
        is_novel = final_q > train_threshold
    else:
        is_novel = None

    results = {
        "empirical_quantile": emp_q,
        "best_k": best_k,
        "gevd_params": best_params,
        "gevd_quantile": gevd_q,
        "mixture_quantile": final_q,
        "ks_statistic": best_stat,
        "train_threshold": train_threshold,
        "is_novel": is_novel
    }

    return results


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    # Simulate: training threshold = 2.0
    train_threshold = 2.0
    results = test_pipeline(train_threshold=train_threshold)

    print("=== Testing Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")
