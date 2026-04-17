"""
train_taxi_qlearning.py
-----------------------
Entry point for training a Q-learning agent on Taxi-v3.

Usage:
    python train_taxi_qlearning.py

Outputs (written to results/):
    results/models/qtable.npy          – learned Q-table
    results/metrics/metrics.npz        – binary metrics archive
    results/metrics/metrics.csv        – human-readable metrics (pandas)
    results/figures/training_metrics.png  – learning curves

Module layout:
    config.py              → all hyperparameters & paths
    agent/                 → QLearningAgent (Q-table, policy, Bellman update)
    environment/           → make_env factory
    utils/                 → metrics I/O (npz + CSV via pandas)
    training/              → train_loop (this script calls it)
"""

import sys
import os

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import (
    N_EPISODES, MAX_STEPS, LOG_EVERY,
    QTABLE_PATH, METRICS_NPZ_PATH, METRICS_CSV_PATH, FIGURE_PATH,
)
from agent import QLearningAgent
from environment import make_env, env_info
from utils import save_training_metrics, training_summary
from training.train_loop import run_training
from evaluation.plot_metrics import build_figure, load_metrics_df


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SpaceRL – Version 1: Taxi-v3  |  Q-learning Training")
    print("=" * 60)

    # 1. Create environment and agent
    env   = make_env()
    info  = env_info(env)
    print(f"\n[ENV] {info['env_name']}  |  states: {info['n_states']}  |  actions: {info['n_actions']}")

    agent = QLearningAgent(env)

    # 2. Train
    print(f"\n[TRAIN] Starting {N_EPISODES:,} episodes...\n")
    rewards, steps, td_errors, epsilon_history = run_training(env, agent)
    env.close()

    # 3. Save Q-table
    agent.save(QTABLE_PATH)

    # 4. Save metrics (npz + CSV)
    df_train = save_training_metrics(
        rewards, steps, td_errors, epsilon_history,
        npz_path=METRICS_NPZ_PATH,
        csv_path=METRICS_CSV_PATH,
    )

    # 5. Print summary
    summary = training_summary(df_train, last_n=500)
    print("\n" + "=" * 60)
    print("  Training Summary  (last 500 episodes)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)

    # 6. Generate and save plots
    print(f"\n[PLOT] Generating training curves → {FIGURE_PATH}")
    df_plot = load_metrics_df()          # re-loads from disk to confirm saved correctly
    build_figure(df_plot, save_path=FIGURE_PATH, show=False)

    print("\n[DONE] Training complete.")
    print(f"  Q-table  : {QTABLE_PATH}")
    print(f"  Metrics  : {METRICS_CSV_PATH}")
    print(f"  Figure   : {FIGURE_PATH}")