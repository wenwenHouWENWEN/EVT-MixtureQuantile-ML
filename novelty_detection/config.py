# config.py
"""
Configuration file for the probabilistic novelty detection framework.
Based on the paper's framework (GEVD-based approach).
"""

# -----------------------------
# General parameters
# -----------------------------
RANDOM_SEED = 42

# Significance level (α)
ALPHA = 0.05   # 5% significance level, can adjust based on application

# -----------------------------
# Distance metric
# -----------------------------
# Options: "euclidean", "manhattan", "cosine"
DISTANCE_METRIC = "euclidean"

# -----------------------------
# GEVD fitting parameters
# -----------------------------
# Range of k values to try for selecting best GEVD fit
K_MIN = 10
K_MAX = 50
K_STEP = 5

# -----------------------------
# Data paths
# -----------------------------
TRAIN_DATA_PATH = "data/train.npy"   # Placeholder, adjust to your dataset
TEST_DATA_PATH = "data/test.npy"
RESULTS_PATH = "results/"

# -----------------------------
# Debugging / Logging
# -----------------------------
VERBOSE = True
