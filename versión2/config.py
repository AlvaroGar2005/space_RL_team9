"""
config.py
---------
Central configuration for SpaceRL Version 2 (Taxi-v3 + Wrappers).
All hyperparameters, wrapper settings and path constants are defined here.
Import this module in any script to keep settings consistent across the project.
"""

# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
ENV_NAME = "Taxi-v3"

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
N_EPISODES      = 10_000
MAX_STEPS       = 200
LEARNING_RATE   = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.9995
LOG_EVERY       = 500

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
N_EVAL_EPS  = 100
DEMO_EPS    = 3
RENDER_DEMO = True

# ──────────────────────────────────────────────
# Wrapper settings
# ──────────────────────────────────────────────

# RewardWrapper — SpaceRL penalty/bonus values
REWARD_INVALID_ACTION_PENALTY = -5.0   # extra penalty for invalid moves (operacional error)
REWARD_STEP_PENALTY           = -0.5   # extra penalty per step (fuel consumption)
REWARD_DELIVERY_BONUS         = +10.0  # extra bonus for successful delivery
REWARD_EFFICIENCY_THRESHOLD   = 15     # steps below this = efficiency bonus
REWARD_EFFICIENCY_BONUS       = +5.0   # bonus for very fast deliveries

# ObservationWrapper — decode Taxi state into components
# Taxi-v3 encodes state as integer: decode into (taxi_row, taxi_col, passenger, destination)
# We group into a compact tuple and re-index as a discrete integer for the Q-table
OBS_GRID_SIZE   = 5   # 5x5 grid
OBS_N_PASSENGER = 5   # 4 locations + 1 (in taxi)
OBS_N_DEST      = 4   # 4 destinations

# ActionWrapper — restrict/reinterpret actions
# Taxi-v3 actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
# We allow all 6 but add tracking of semantically "risky" actions
RISKY_ACTIONS = [4, 5]   # Pickup and Dropoff — high-cost ops in SpaceRL narrative

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
WINDOW_SIZE = 200
FIGURE_DPI  = 150
RAW_ALPHA   = 0.25

# ──────────────────────────────────────────────
# Paths (relative to v2/ root)
# ──────────────────────────────────────────────
QTABLE_PATH          = "results/models/qtable_v2.npy"
METRICS_NPZ_PATH     = "results/metrics/metrics_v2.npz"
METRICS_CSV_PATH     = "results/metrics/metrics_v2.csv"
EVAL_CSV_PATH        = "results/metrics/evaluation_v2.csv"
FIGURE_PATH          = "results/figures/training_metrics_v2.png"
COMPARISON_CSV_PATH  = "results/comparisons/v1_vs_v2_comparison.csv"
WRAPPER_ANALYSIS_CSV = "results/comparisons/wrapper_analysis.csv"

# Path to V1 metrics for comparison (adjust if your V1 folder differs)
V1_METRICS_CSV = "../v1/results/metrics/metrics.csv"
V1_EVAL_CSV    = "../v1/results/metrics/evaluation.csv"