"""
config.py
---------
Central configuration for SpaceRL Version 1 (Taxi-v3 Q-learning).
All hyperparameters and path constants are defined here.
Import this module in any script to keep settings consistent across the project.
"""

# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
ENV_NAME = "Taxi-v3"

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
N_EPISODES      = 10_000   # total training episodes
MAX_STEPS       = 200      # max steps per episode (Taxi-v3 default is 200)
LEARNING_RATE   = 0.1      # alpha
DISCOUNT_FACTOR = 0.99     # gamma
EPSILON_START   = 1.0      # initial exploration rate
EPSILON_MIN     = 0.01     # floor for epsilon
EPSILON_DECAY   = 0.9995   # multiplicative decay per episode
LOG_EVERY       = 500      # console log frequency during training

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
N_EVAL_EPS  = 100   # episodes for greedy evaluation
DEMO_EPS    = 3     # episodes rendered in human mode
RENDER_DEMO = False # set True to watch the agent play

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
WINDOW_SIZE  = 200   # moving-average window (episodes)
FIGURE_DPI   = 150
RAW_ALPHA    = 0.25  # transparency of raw curves

# ──────────────────────────────────────────────
# Paths  (relative to v1/ root)
# ──────────────────────────────────────────────
QTABLE_PATH      = "results/models/qtable.npy"
METRICS_NPZ_PATH = "results/metrics/metrics.npz"
METRICS_CSV_PATH = "results/metrics/metrics.csv"
EVAL_CSV_PATH    = "results/metrics/evaluation.csv"
FIGURE_PATH      = "results/figures/training_metrics.png"