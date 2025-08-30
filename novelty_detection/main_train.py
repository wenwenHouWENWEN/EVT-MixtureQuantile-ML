# main_train.py
"""
Training phase (Framework A)
----------------------------
Pipeline:
1. Load training data
2. Compute distance matrix
3. Arrange and negate distances
4. Empirical quantile estimation
5. GEVD fitting with k-selection
6. Mixture quantile estimation
7. Save final threshold
"""

import numpy as np
import config
from data_loader import load_data
from distance import compute_distance_matrix
from arrange import arrange_and_negate
from quantile import empirical_quantile
from gevd_fit import select_k_and_fit, gevd_quantile
from mixture_quantile import mixture_quantile


def train_pipeline():
    """
    Full training pipeline (Framework A).

    Returns
    -------
    result : dict
        Dictionary with all intermediate and final results.
    """
    # 1. Load training data
    X_train, _ = load_data()

    # 2. Compute pairwise distances in training data
    D_train = compute_distance_matrix(X_train)
    distances = D_train.flatten()

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

    # 7. Return all results
    result = {
        "empirical_quantile": emp_q,
        "best_k": best_k,
        "gevd_params": best_params,
        "gevd_quantile": gevd_q,
        "mixture_quantile": final_q,
        "ks_statistic": best_stat
    }

    return result


# ---------------------------
# Example usage (for testing)
# ---------------------------
if __name__ == "__main__":
    result = train_pipeline()
    print("=== Training Results ===")
    for k, v in result.items():
        print(f"{k}: {v}")
