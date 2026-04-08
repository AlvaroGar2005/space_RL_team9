"""
plot_metrics.py
===============
SpaceRL – Version 1: Standard Gymnasium Environment
----------------------------------------------------
Loads the metrics saved by train_taxi_qlearning.py and produces a three-panel
figure showing how the agent's performance evolved over training.

Plots:
    1. Total reward per episode  (raw + moving average)
    2. Steps per episode         (raw + moving average)
    3. Mean TD error per episode (raw + moving average)

Output:
    training_metrics.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# Configuration
# =============================================================================
METRICS_PATH  = "metrics.npz"
OUTPUT_PATH   = "training_metrics.png"
WINDOW_SIZE   = 200   # Window for the moving average smoother

# =============================================================================
# Load metrics
# =============================================================================
if not os.path.exists(METRICS_PATH):
    print(
        "[ERROR] metrics.npz not found.\n"
        "        Please run train_taxi_qlearning.py first."
    )
    sys.exit(1)

data     = np.load(METRICS_PATH)
rewards  = data["rewards"]
steps    = data["steps"]
td_errors = data["td_errors"]

n_episodes = len(rewards)
episodes   = np.arange(1, n_episodes + 1)

print(f"[INFO] Loaded metrics for {n_episodes} episodes from '{METRICS_PATH}'")

# =============================================================================
# Helper: compute moving average
# =============================================================================
def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a 1-D array of equal length with a centred moving average.

    Edges are padded with NaN so the curve length matches the input.
    """
    if window >= len(values):
        return np.full_like(values, np.mean(values), dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    # Pad with NaN on both sides to keep the same length
    pad_left  = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return np.concatenate([
        np.full(pad_left,  np.nan),
        smoothed,
        np.full(pad_right, np.nan),
    ])

rewards_ma   = moving_average(rewards,   WINDOW_SIZE)
steps_ma     = moving_average(steps,     WINDOW_SIZE)
td_errors_ma = moving_average(td_errors, WINDOW_SIZE)

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("SpaceRL – Version 1: Q-learning on Taxi-v3\nTraining Metrics",
             fontsize=14, fontweight="bold", y=0.98)

RAW_COLOR    = "#a8c7fa"   # light blue for raw curve
SMOOTH_COLOR = "#1a73e8"   # strong blue for moving average
ALPHA_RAW    = 0.35

# --- Panel 1: Total reward per episode ---
ax = axes[0]
ax.plot(episodes, rewards,    color=RAW_COLOR,    alpha=ALPHA_RAW, linewidth=0.6, label="Raw")
ax.plot(episodes, rewards_ma, color=SMOOTH_COLOR, linewidth=1.8,   label=f"MA({WINDOW_SIZE})")
ax.set_ylabel("Total reward", fontsize=11)
ax.set_title("Reward per episode", fontsize=11, loc="left")
ax.legend(fontsize=9, loc="lower right")
ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.grid(True, alpha=0.3)

# --- Panel 2: Steps per episode ---
ax = axes[1]
ax.plot(episodes, steps,    color=RAW_COLOR,    alpha=ALPHA_RAW, linewidth=0.6, label="Raw")
ax.plot(episodes, steps_ma, color=SMOOTH_COLOR, linewidth=1.8,   label=f"MA({WINDOW_SIZE})")
ax.set_ylabel("Steps", fontsize=11)
ax.set_title("Steps per episode", fontsize=11, loc="left")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

# --- Panel 3: Mean TD error per episode ---
ax = axes[2]
ax.plot(episodes, td_errors,    color=RAW_COLOR,    alpha=ALPHA_RAW, linewidth=0.6, label="Raw")
ax.plot(episodes, td_errors_ma, color=SMOOTH_COLOR, linewidth=1.8,   label=f"MA({WINDOW_SIZE})")
ax.set_ylabel("Mean |TD error|", fontsize=11)
ax.set_title("TD error per episode", fontsize=11, loc="left")
ax.set_xlabel("Episode", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

# Shared x-axis formatting
axes[2].xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"[INFO] Figure saved → {OUTPUT_PATH}")
plt.close()
