"""
evaluation/plot_metrics_v2.py
------------------------------
Generates V2 training plots and V1 vs V2 comparison figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import FIGURE_PATH, WINDOW_SIZE, RAW_ALPHA, FIGURE_DPI, COMPARISON_CSV_PATH


def _smooth(values, window):
    if window >= len(values):
        return np.full_like(values, np.mean(values), dtype=float)
    kernel = np.ones(window) / window
    s = np.convolve(values, kernel, mode="valid")
    pl = (len(values) - len(s)) // 2
    pr = len(values) - len(s) - pl
    return np.pad(s, (pl, pr), mode="edge")


def _panel(ax, values, title, ylabel, color):
    eps = np.arange(1, len(values) + 1)
    sm  = _smooth(values, WINDOW_SIZE)
    ax.plot(eps, values, color=color, alpha=RAW_ALPHA, linewidth=0.6, label="Raw")
    ax.plot(eps, sm, color=color, linewidth=1.8, label=f"Avg (w={WINDOW_SIZE})")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)


def build_training_figure(df: pd.DataFrame,
                           save_path: str = FIGURE_PATH,
                           show: bool = True):
    """
    6-panel training figure:
    top row: reward, steps, TD error, epsilon
    bottom row: invalid_actions, wrapper_penalty, wrapper_bonus, useful_actions_ratio
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("SpaceRL V2 — Wrapped Taxi-v3 | Training Metrics", fontsize=13, fontweight="bold")

    _panel(axes[0,0], df["reward"].values,               "Reward per Episode",           "Reward",             "#2196F3")
    _panel(axes[0,1], df["steps"].values,                "Steps per Episode",             "Steps",              "#4CAF50")
    _panel(axes[0,2], df["td_error"].values,             "Mean TD Error",                 "TD Error",           "#FF5722")
    _panel(axes[0,3], df["epsilon"].values,              "Epsilon Decay",                 "ε",                  "#9C27B0")
    _panel(axes[1,0], df["invalid_actions"].values,      "Invalid Actions per Episode",   "Count",              "#F44336")
    _panel(axes[1,1], df["wrapper_penalty"].values,      "Wrapper Penalty per Episode",   "Total penalty",      "#FF9800")
    _panel(axes[1,2], df["wrapper_bonus"].values,        "Wrapper Bonus per Episode",     "Total bonus",        "#8BC34A")
    _panel(axes[1,3], df["useful_actions_ratio"].values, "Useful Actions Ratio",          "Ratio",              "#00BCD4")

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[PLOT] V2 training figure saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def build_comparison_figure(df_v1: pd.DataFrame,
                              df_v2: pd.DataFrame,
                              save_path: str = None,
                              show: bool = True):
    """
    Side-by-side comparison of V1 and V2 on reward, steps and TD error.
    """
    if save_path is None:
        save_path = FIGURE_PATH.replace("training_metrics_v2", "v1_vs_v2_comparison")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(len(df_v1), len(df_v2))
    eps = np.arange(1, n + 1)

    metrics = [
        ("reward",   "Reward per Episode",  "#2196F3", "#FF5722"),
        ("steps",    "Steps per Episode",   "#4CAF50", "#FF9800"),
        ("td_error", "Mean TD Error",       "#9C27B0", "#F44336"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle("SpaceRL — V1 vs V2 Comparison", fontsize=13, fontweight="bold")

    for ax, (col, title, c1, c2) in zip(axes, metrics):
        if col not in df_v1.columns or col not in df_v2.columns:
            continue
        v1s = _smooth(df_v1[col].values[:n], WINDOW_SIZE)
        v2s = _smooth(df_v2[col].values[:n], WINDOW_SIZE)
        ax.plot(eps, v1s, color=c1, linewidth=1.8, label="V1 (baseline)")
        ax.plot(eps, v2s, color=c2, linewidth=1.8, linestyle="--", label="V2 (wrapped)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[PLOT] Comparison figure saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
