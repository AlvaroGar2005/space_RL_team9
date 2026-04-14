"""
plot_metrics.py
---------------
Loads training metrics saved by train_taxi_qlearning.py and produces
a three-panel figure showing:
    1. Total reward per episode
    2. Steps per episode
    3. Mean TD error per episode

Both the raw curve and a smoothed moving average are displayed.
The figure is saved as training_metrics.png.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
METRICS_FILE   = "metrics.npz"
OUTPUT_FILE    = "training_metrics.png"
WINDOW_SIZE    = 200    # window for the moving average (in episodes)
FIGURE_DPI     = 150    # resolution of the saved figure
RAW_ALPHA      = 0.25   # transparency of the raw curve


def load_metrics(path: str) -> tuple:
    """
    Load training metrics from a .npz file.
    Raises a clear error if the file is not found.
    """
    try:
        data = np.load(path)
        rewards   = data["rewards"]
        steps     = data["steps"]
        td_errors = data["td_errors"]
        print(f"[INFO] Loaded '{path}'  ({len(rewards)} episodes)")
        return rewards, steps, td_errors
    except FileNotFoundError:
        print(
            f"\n[ERROR] File '{path}' not found.\n"
            "  → You must train the agent first.\n"
            "  → Run:  python train_taxi_qlearning.py\n"
        )
        sys.exit(1)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a centered moving average using convolution.
    Edges are handled with 'valid' mode so the returned array is shorter;
    we pad both ends to preserve the original length.
    """
    if window >= len(values):
        return np.full_like(values, np.mean(values), dtype=float)
    kernel = np.ones(window) / window
    smooth = np.convolve(values, kernel, mode="valid")
    # Pad to match original length
    pad_left  = (len(values) - len(smooth)) // 2
    pad_right = len(values) - len(smooth) - pad_left
    return np.pad(smooth, (pad_left, pad_right), mode="edge")


def plot_panel(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
):
    """
    Draw a single subplot with the raw curve and its moving average.
    """
    episodes = np.arange(1, len(values) + 1)
    smooth   = moving_average(values, WINDOW_SIZE)

    # Raw curve (transparent)
    ax.plot(episodes, values, color=color, alpha=RAW_ALPHA, linewidth=0.6, label="Raw")

    # Smoothed curve (solid)
    ax.plot(
        episodes, smooth,
        color=color, linewidth=1.8,
        label=f"Moving avg (window={WINDOW_SIZE})",
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)


def build_figure(rewards, steps, td_errors):
    """Create the three-panel figure and save it."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    fig.suptitle(
        "SpaceRL – Version 1: Taxi-v3 Q-learning Training Metrics",
        fontsize=13, fontweight="bold",
    )

    plot_panel(
        axes[0], rewards,
        title="Total Reward per Episode",
        ylabel="Cumulative reward",
        color="#2196F3",
    )
    plot_panel(
        axes[1], steps,
        title="Steps per Episode",
        ylabel="Number of steps",
        color="#4CAF50",
    )
    plot_panel(
        axes[2], td_errors,
        title="Mean TD Error per Episode",
        ylabel="Mean |TD error|",
        color="#FF5722",
    )

    fig.savefig(OUTPUT_FILE, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[INFO] Figure saved as '{OUTPUT_FILE}'")
    plt.show()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SpaceRL – Version 1: Taxi-v3 Training Metrics Plot")
    print("=" * 55)

    rewards, steps, td_errors = load_metrics(METRICS_FILE)
    build_figure(rewards, steps, td_errors)
    print("[DONE] Plotting complete.")