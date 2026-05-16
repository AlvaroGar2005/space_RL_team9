"""
config.py
---------
Central configuration for SpaceRL Version 3 — SpaceMissionEnv.
All hyperparameters, environment settings and paths are defined here.
"""

# ──────────────────────────────────────────────
# Environment — SpaceMissionEnv
# ──────────────────────────────────────────────
ENV_NAME = "SpaceMissionEnv-v3"

# State variable levels (each variable has 3 discrete levels: 0=low, 1=mid, 2=high)
N_LEVELS = 3

# State variables
STATE_VARS = ["energy", "oxygen", "food", "fuel", "hull", "event"]

# Event types
EVENT_NORMAL    = 0
EVENT_STORM     = 1   # solar storm
EVENT_ASTEROID  = 2   # asteroid impact
EVENT_SAFE_ZONE = 3   # safe zone

N_EVENTS  = 4
N_STATES  = N_LEVELS ** 5 * N_EVENTS   # 3^5 * 4 = 972 discrete states
N_ACTIONS = 6

# Action indices
ACTION_LIFE_SUPPORT    = 0
ACTION_PROPULSION      = 1
ACTION_GATHER          = 2
ACTION_REPAIR          = 3
ACTION_POWER_SAVING    = 4
ACTION_SAFE_ZONE       = 5

ACTION_NAMES = [
    "Life Support",
    "Propulsion",
    "Gather Resources",
    "Repair Ship",
    "Power Saving",
    "Move to Safe Zone",
]

# Max steps per episode
MAX_STEPS = 200

# Mission success: survive this many steps with all resources balanced
MISSION_SUCCESS_STEPS = 50

# ──────────────────────────────────────────────
# Reward shaping (inherited and refined from V2)
# ──────────────────────────────────────────────
REWARD_SURVIVE          = +1.0    # base: survive one step
REWARD_BALANCED         = +2.0    # all resources at mid or high level
REWARD_SAFE_ZONE        = +5.0    # good use of safe zone
REWARD_CRITICAL_LOW     = -5.0    # any critical resource at 0
REWARD_SEVERE_DAMAGE    = -10.0   # hull at 0 (severe damage)
REWARD_MISSION_FAILED   = -100.0  # total mission failure
REWARD_MISSION_SUCCESS  = +50.0   # mission completed successfully

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
N_EPISODES      = 15_000
LEARNING_RATE   = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.9997
LOG_EVERY       = 1000

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
N_EVAL_EPS  = 200
RENDER_DEMO = True
DEMO_EPS    = 10

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
WINDOW_SIZE = 300
FIGURE_DPI  = 150
RAW_ALPHA   = 0.2

# ──────────────────────────────────────────────
# Paths (relative to v3/ root)
# ──────────────────────────────────────────────
QTABLE_PATH      = "results/models/qtable_v3.npy"
METRICS_CSV_PATH = "results/metrics/metrics_v3.csv"
EVAL_CSV_PATH    = "results/metrics/evaluation_v3.csv"
FIGURE_PATH      = "results/figures/training_metrics_v3.png"
COMPARISON_CSV   = "results/comparisons/v1_v2_v3_comparison.csv"

# Paths to previous versions (for comparison)
V1_METRICS_CSV = "../versión1/results/metrics/metrics.csv"
V2_METRICS_CSV = "../versión2/results/metrics/metrics_v2.csv"
V1_EVAL_CSV    = "../versión1/results/metrics/evaluation.csv"
V2_EVAL_CSV    = "../versión2/results/metrics/evaluation_v2.csv"
