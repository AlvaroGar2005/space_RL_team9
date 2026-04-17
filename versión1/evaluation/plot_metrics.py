"""
evaluation/plot_metrics.py
--------------------------
Generates training and evaluation plots from saved metrics.

Used by:
    plot_metrics.py             (CLI entry point)
    evaluate_taxi_qlearning.py  (optional plot after evaluation)
    notebooks/SpaceRL_v1.ipynb  (inline plots in Jupyter)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import METRICS_NPZ_PATH, METRICS_CSV_PATH, FIGURE_PATH, WINDOW_SIZE, RAW_ALPHA, FIGURE_DPI
from utils.metrics_io import load_training_metrics


def load_metrics_df(prefer_csv: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load training metrics as a DataFrame."""
    return load_training_metrics(
        npz_path=METRICS_NPZ_PATH,
        csv_path=METRICS_CSV_PATH,
        prefer_csv=prefer_csv,
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a moving average using convolution and pad back to original length.
    """
    if window >= len(values):
        return np.full_like(values, np.mean(values), dtype=float)
    kernel     = np.ones(window) / window
    smooth     = np.convolve(values, kernel, mode="valid")
    pad_left   = (len(values) - len(smooth)) // 2
    pad_right  = len(values) - len(smooth) - pad_left
    return np.pad(smooth, (pad_left, pad_right), mode="edge")


def _panel(ax, values, title, ylabel, color):
    """Draw one subplot (raw curve + moving average)."""
    eps    = np.arange(1, len(values) + 1)
    smooth = moving_average(values, WINDOW_SIZE)

    ax.plot(eps, values, color=color, alpha=RAW_ALPHA, linewidth=0.6, label="Raw")
    ax.plot(
        eps, smooth,
        color=color, linewidth=1.8,
        label=f"Moving avg (w={WINDOW_SIZE})",
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)


def build_figure(
    df: pd.DataFrame,
    save_path: str = FIGURE_PATH,
    show: bool = True,
):
    """
    Build a 4-panel figure:
        1. Reward per episode
        2. Steps per episode
        3. Mean TD error per episode
        4. Epsilon decay curve

    Args:
        df        : training metrics DataFrame (output of load_metrics_df)
        save_path : where to save the PNG
        show      : call plt.show() at the end (False for headless/CI)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    fig.suptitle(
        "SpaceRL – Version 1: Taxi-v3 Q-learning  |  Training Metrics",
        fontsize=13, fontweight="bold",
    )

    _panel(axes[0], df["reward"].values,   "Total Reward per Episode",     "Cumulative reward",  "#2196F3")
    _panel(axes[1], df["steps"].values,    "Steps per Episode",             "Steps",              "#4CAF50")
    _panel(axes[2], df["td_error"].values, "Mean TD Error per Episode",     "Mean |TD error|",    "#FF5722")

    # Panel 4: epsilon decay (no moving average needed, already smooth)
    eps_vals = df["epsilon"].values
    episodes = np.arange(1, len(eps_vals) + 1)
    axes[3].plot(episodes, eps_vals, color="#9C27B0", linewidth=1.8)
    axes[3].set_title("Epsilon Decay", fontsize=11, fontweight="bold", pad=6)
    axes[3].set_xlabel("Episode", fontsize=9)
    axes[3].set_ylabel("ε (epsilon)", fontsize=9)
    axes[3].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[3].grid(True, linestyle="--", alpha=0.4)
    axes[3].tick_params(labelsize=8)

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[PLOT] Figure saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)