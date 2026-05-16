"""
evaluation/plot_metrics_v3.py
------------------------------
Training and comparison plots for SpaceRL V3.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import FIGURE_PATH, WINDOW_SIZE, RAW_ALPHA, FIGURE_DPI, COMPARISON_CSV


def _smooth(values, window=WINDOW_SIZE):
    if window >= len(values):
        return np.full_like(values, np.mean(values), dtype=float)
    k = np.ones(window) / window
    s = np.convolve(values, k, mode="valid")
    pl = (len(values) - len(s)) // 2
    pr = len(values) - len(s) - pl
    return np.pad(s, (pl, pr), mode="edge")


def _panel(ax, values, title, ylabel, color):
    eps = np.arange(1, len(values) + 1)
    ax.plot(eps, values, color=color, alpha=RAW_ALPHA, linewidth=0.5, label="Raw")
    ax.plot(eps, _smooth(values), color=color, linewidth=1.8, label=f"Avg (w={WINDOW_SIZE})")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)


def build_training_figure(df: pd.DataFrame,
                           save_path: str = FIGURE_PATH,
                           show: bool = True):
    """5-panel training figure: reward, steps, TD error, epsilon, success rate."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("SpaceRL V3 — SpaceMissionEnv | Training Metrics", fontsize=13, fontweight="bold")

    _panel(axes[0,0], df["reward"].values,             "Reward per Episode",      "Reward",    "#2196F3")
    _panel(axes[0,1], df["steps"].values,              "Steps per Episode",        "Steps",     "#4CAF50")
    _panel(axes[0,2], df["td_error"].values,           "Mean TD Error",            "TD Error",  "#FF5722")
    _panel(axes[1,0], df["epsilon"].values,            "Epsilon Decay",            "ε",         "#9C27B0")

    # Success rate (rolling)
    success_smooth = _smooth(df["success"].values.astype(float) * 100)
    axes[1,1].plot(np.arange(1, len(success_smooth)+1), success_smooth, color="#FF9800", linewidth=1.8)
    axes[1,1].set_title("Mission Success Rate (%)", fontsize=10, fontweight="bold")
    axes[1,1].set_xlabel("Episode", fontsize=8)
    axes[1,1].set_ylabel("Success %", fontsize=8)
    axes[1,1].set_ylim(0, 105)
    axes[1,1].grid(True, linestyle="--", alpha=0.4)
    axes[1,1].tick_params(labelsize=7)

    # Termination reasons pie (last 2000 episodes)
    last = df.tail(2000)
    if "term_reason" in df.columns:
        counts = last["term_reason"].value_counts()
        axes[1,2].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                      colors=["#4CAF50","#F44336","#FF9800","#2196F3"])
        axes[1,2].set_title("Termination Reasons\n(last 2000 episodes)", fontsize=10, fontweight="bold")
    else:
        axes[1,2].axis("off")

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[PLOT] V3 training figure saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def build_three_way_figure(df_v1, df_v2, df_v3,
                            save_path: str = None,
                            show: bool = True):
    """V1 vs V2 vs V3 comparison figure."""
    if save_path is None:
        save_path = FIGURE_PATH.replace("training_metrics_v3", "v1_v2_v3_comparison")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(
        len(df_v1) if df_v1 is not None else 99999,
        len(df_v2) if df_v2 is not None else 99999,
        len(df_v3),
    )
    eps = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    fig.suptitle("SpaceRL — V1 vs V2 vs V3 Comparison", fontsize=13, fontweight="bold")

    metrics = [
        ("reward", "Reward per Episode"),
        ("steps",  "Steps per Episode"),
    ]
    colors = {"V1": "#2196F3", "V2": "#FF9800", "V3": "#4CAF50"}

    for ax, (col, title) in zip(axes[:2], metrics):
        for df, name in [(df_v1,"V1"),(df_v2,"V2"),(df_v3,"V3")]:
            if df is not None and col in df.columns:
                ax.plot(eps, _smooth(df[col].values[:n]), color=colors[name],
                        linewidth=1.8, label=name)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(labelsize=7)

    # Success rate comparison
    ax = axes[2]
    for df, name in [(df_v1,"V1"),(df_v2,"V2"),(df_v3,"V3")]:
        if df is not None and "success" in df.columns:
            ax.plot(eps, _smooth(df["success"].values[:n].astype(float)*100),
                    color=colors[name], linewidth=1.8, label=name)
    ax.set_title("Mission Success Rate (%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Success %", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[PLOT] Comparison figure saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
