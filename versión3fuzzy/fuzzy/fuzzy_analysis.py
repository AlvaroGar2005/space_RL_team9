"""
fuzzy/fuzzy_analysis.py
------------------------
Batch fuzzy analysis tools for SpaceRL V3.

Provides:
    - Evaluation of the full Q-table through the fuzzy system
    - Per-state MHI heatmap data
    - CSV export of fuzzy results
    - Comparison of fuzzy risk vs Q-learning reward
    - Visualisation helpers for the notebook
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from fuzzy.fuzzy_mission import evaluate_mission_state, get_memberships
from environment.space_mission_env import SpaceMissionEnv
from config import N_STATES, N_ACTIONS, ACTION_NAMES


# ── State decoding helper ────────────────────────────────────────────────────

def decode_all_states() -> pd.DataFrame:
    """
    Decode all 972 discrete states of SpaceMissionEnv into their
    resource components and compute the fuzzy MHI for each.

    Returns:
        DataFrame with columns:
        state_id, energy, oxygen, food, fuel, hull, event,
        mhi, crisp_risk, status
    """
    rows = []
    for s in range(N_STATES):
        desc   = SpaceMissionEnv.decode_state(s)
        # Map label to numeric for fuzzy evaluation
        level_map = {"critical/low": 0, "stable/mid": 1, "optimal/high": 2}
        e = level_map.get(desc["energy"], 1)
        o = level_map.get(desc["oxygen"], 1)
        f = level_map.get(desc["food"],   1)
        u = level_map.get(desc["fuel"],   1)
        h = level_map.get(desc["hull"],   1)

        result = evaluate_mission_state(e, o, f, u, h)

        rows.append({
            "state_id"   : s,
            "energy"     : desc["energy"],
            "oxygen"     : desc["oxygen"],
            "food"       : desc["food"],
            "fuel"       : desc["fuel"],
            "hull"       : desc["hull"],
            "event"      : desc["event"],
            "energy_lvl" : e,
            "oxygen_lvl" : o,
            "food_lvl"   : f,
            "fuel_lvl"   : u,
            "hull_lvl"   : h,
            "mhi"        : result["mhi"],
            "crisp_risk" : result["crisp_risk"],
            "status"     : result["status"],
        })

    return pd.DataFrame(rows)


# ── Q-table fuzzy analysis ────────────────────────────────────────────────────

def qtable_fuzzy_analysis(qtable: np.ndarray) -> pd.DataFrame:
    """
    Cross-reference the learned Q-table with the fuzzy MHI of each state.

    For each state, computes:
        - The fuzzy MHI
        - The Q-value of the greedy action
        - The greedy action name
        - Whether the greedy action is "appropriate" given the fuzzy status

    Returns:
        DataFrame with one row per state.
    """
    df_states = decode_all_states()

    greedy_actions   = np.argmax(qtable, axis=1)
    greedy_qvalues   = qtable[np.arange(N_STATES), greedy_actions]
    greedy_names     = [ACTION_NAMES[a] for a in greedy_actions]

    df_states["greedy_action"]  = greedy_names
    df_states["greedy_qvalue"]  = greedy_qvalues.round(3)

    # Appropriateness heuristic:
    # In CRITICAL states, the best action should be life_support, power_saving or repair
    # In OPTIMAL states, any action is appropriate
    appropriate_in_critical = {"Life Support", "Power Saving", "Repair Ship"}

    def is_appropriate(row):
        if row["status"] == "CRITICAL":
            return row["greedy_action"] in appropriate_in_critical
        return True

    df_states["appropriate_action"] = df_states.apply(is_appropriate, axis=1)

    return df_states


# ── CSV export ────────────────────────────────────────────────────────────────

def save_fuzzy_analysis(df: pd.DataFrame,
                         path: str = "results/comparisons/fuzzy_analysis.csv"):
    """Save fuzzy analysis DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Fuzzy] Analysis saved → {path}")
    return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_membership_functions(save_path: str = None):
    """
    Plot the three triangular membership functions (LOW, MEDIUM, HIGH)
    over the resource universe [0, 2].
    Useful for the notebook to explain the fuzzy system.
    """
    universe = np.linspace(0, 2, 300)
    low_vals    = [max(0, 1 - x) if x <= 1 else 0 for x in universe]
    medium_vals = [x if x <= 1 else 2 - x for x in universe]
    medium_vals = [max(0, v) for v in medium_vals]
    high_vals   = [max(0, x - 1) for x in universe]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(universe, low_vals,    color="#F44336", linewidth=2.5, label="LOW (critical)")
    ax.plot(universe, medium_vals, color="#FF9800", linewidth=2.5, label="MEDIUM (stable)")
    ax.plot(universe, high_vals,   color="#4CAF50", linewidth=2.5, label="HIGH (optimal)")

    ax.set_title("Fuzzy Membership Functions — SpaceRL Resource Variables",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Resource level (0=critical, 1=stable, 2=optimal)", fontsize=10)
    ax.set_ylabel("Degree of membership μ(x)", fontsize=10)
    ax.set_xlim(0, 2); ax.set_ylim(-0.05, 1.1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0\n(Critical)", "1\n(Stable)", "2\n(Optimal)"])
    ax.legend(fontsize=10, loc="upper center")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fuzzy] Membership functions figure saved → {save_path}")
    plt.show()
    return fig


def plot_mhi_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot the distribution of Mission Health Index across all 972 states.
    Shows how many states fall into each fuzzy risk category.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Fuzzy Mission Health Index — Distribution across all 972 states",
                 fontsize=12, fontweight="bold")

    # Histogram of MHI values
    axes[0].hist(df["mhi"], bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["mhi"].mean(), color="#FF5722", linestyle="--", linewidth=2,
                    label=f"Mean MHI: {df['mhi'].mean():.3f}")
    axes[0].set_title("MHI Distribution")
    axes[0].set_xlabel("Mission Health Index (MHI)")
    axes[0].set_ylabel("Number of states")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.4)

    # Status pie chart
    status_counts = df["status"].value_counts()
    colors = {"OPTIMAL": "#4CAF50", "STABLE": "#2196F3",
              "WARNING": "#FF9800", "CRITICAL": "#F44336"}
    pie_colors = [colors.get(s, "#9E9E9E") for s in status_counts.index]
    axes[1].pie(status_counts.values, labels=status_counts.index,
                autopct="%1.1f%%", colors=pie_colors,
                startangle=90, textprops={"fontsize": 10})
    axes[1].set_title("States by Fuzzy Status")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fuzzy] MHI distribution figure saved → {save_path}")
    plt.show()
    return fig


def plot_mhi_vs_qvalue(df: pd.DataFrame, save_path: str = None):
    """
    Scatter plot: MHI (fuzzy) vs greedy Q-value (learned).
    Shows the correlation between what the fuzzy system considers healthy
    and what the Q-learning agent has learned to value.
    """
    if "greedy_qvalue" not in df.columns:
        print("[Fuzzy] No Q-value data found. Run qtable_fuzzy_analysis() first.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    status_colors = {"OPTIMAL": "#4CAF50", "STABLE": "#2196F3",
                     "WARNING": "#FF9800", "CRITICAL": "#F44336"}

    for status, grp in df.groupby("status"):
        ax.scatter(grp["mhi"], grp["greedy_qvalue"],
                   color=status_colors.get(status, "grey"),
                   label=status, alpha=0.5, s=20)

    # Correlation line
    corr = df["mhi"].corr(df["greedy_qvalue"])
    ax.set_title(f"Fuzzy MHI vs Q-learning Greedy Q-value  (r = {corr:.3f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Mission Health Index (fuzzy)", fontsize=10)
    ax.set_ylabel("Greedy Q-value (Q-learning)", fontsize=10)
    ax.legend(title="Fuzzy Status", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fuzzy] MHI vs Q-value figure saved → {save_path}")
    plt.show()
    return fig


def plot_fuzzy_rules_example(save_path: str = None):
    """
    Visual example of fuzzy rule evaluation for three representative states:
        - All resources at 0 (critical)
        - All resources at 1 (stable)
        - All resources at 2 (optimal)
    Shows the membership degrees and the resulting MHI.
    """
    from fuzzy.fuzzy_mission import evaluate_mission_state

    cases = [
        {"label": "All CRITICAL (0,0,0,0,0)", "vals": (0, 0, 0, 0, 0)},
        {"label": "All STABLE   (1,1,1,1,1)", "vals": (1, 1, 1, 1, 1)},
        {"label": "All OPTIMAL  (2,2,2,2,2)", "vals": (2, 2, 2, 2, 2)},
        {"label": "Mixed: E=0,O=2,F=1,U=1,H=2", "vals": (0, 2, 1, 1, 2)},
    ]

    fig, axes = plt.subplots(1, len(cases), figsize=(15, 4))
    fig.suptitle("Fuzzy Rule Evaluation — Representative States",
                 fontsize=12, fontweight="bold")

    resources = ["Energy", "Oxygen", "Food", "Fuel", "Hull"]
    colors_mf = {"low": "#F44336", "medium": "#FF9800", "high": "#4CAF50"}

    for ax, case in zip(axes, cases):
        result = evaluate_mission_state(*case["vals"])
        memberships = result["memberships"]

        x = np.arange(len(resources))
        low_vals    = [memberships[r.lower()]["low"]    for r in resources]
        medium_vals = [memberships[r.lower()]["medium"] for r in resources]
        high_vals   = [memberships[r.lower()]["high"]   for r in resources]

        width = 0.25
        ax.bar(x - width, low_vals,    width, label="LOW",    color="#F44336", alpha=0.8)
        ax.bar(x,         medium_vals, width, label="MEDIUM", color="#FF9800", alpha=0.8)
        ax.bar(x + width, high_vals,   width, label="HIGH",   color="#4CAF50", alpha=0.8)

        ax.set_title(f"{case['label']}\nMHI = {result['mhi']:.3f}  [{result['status']}]",
                     fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(resources, fontsize=7, rotation=20)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("μ(x)", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fuzzy] Rules example figure saved → {save_path}")
    plt.show()
    return fig
