"""
train_taxi_v2.py
----------------
Entry point for training the Q-learning agent on the wrapped SpaceRL V2 environment.

Usage:
    python train_taxi_v2.py

Outputs:
    results/models/qtable_v2.npy
    results/metrics/metrics_v2.npz
    results/metrics/metrics_v2.csv
    results/figures/training_metrics_v2.png
    results/comparisons/v1_vs_v2_comparison.csv  (if V1 metrics exist)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import QTABLE_PATH, METRICS_NPZ_PATH, METRICS_CSV_PATH, FIGURE_PATH
from agent import QLearningAgent
from environment import make_env_v2, env_info_v2
from training import run_training_v2
from utils import (
    save_training_metrics_v2, training_summary_v2,
    load_v1_metrics, build_comparison_csv,
)
from evaluation.plot_metrics_v2 import build_training_figure, build_comparison_figure
from utils.metrics_io_v2 import load_training_metrics_v2


if __name__ == "__main__":
    print("=" * 65)
    print("  SpaceRL – Version 2: Wrapped Taxi-v3  |  Q-learning Training")
    print("=" * 65)

    env  = make_env_v2()
    info = env_info_v2(env)
    print(f"\n[ENV] {info['env_name']}  |  wrappers: {info['wrappers']}")
    print(f"      states: {info['n_states']}  |  actions: {info['n_actions']}")

    agent = QLearningAgent(env)

    print(f"\n[TRAIN] Starting training...\n")
    metrics = run_training_v2(env, agent)
    env.close()

    agent.save(QTABLE_PATH)

    df_train = save_training_metrics_v2(
        metrics, npz_path=METRICS_NPZ_PATH, csv_path=METRICS_CSV_PATH
    )

    summary = training_summary_v2(df_train, last_n=500)
    print("\n" + "=" * 65)
    print("  Training Summary (last 500 episodes)")
    print("=" * 65)
    print(summary.to_string(index=False))

    print(f"\n[PLOT] Generating training figure...")
    build_training_figure(df_train, save_path=FIGURE_PATH, show=False)

    # V1 vs V2 comparison (if V1 results exist)
    df_v1 = load_v1_metrics()
    if df_v1 is not None:
        df_cmp = build_comparison_csv(df_v1, df_train)
        cmp_fig = FIGURE_PATH.replace("training_metrics_v2", "v1_vs_v2_comparison")
        build_comparison_figure(df_v1, df_train, save_path=cmp_fig, show=False)
        print(f"[PLOT] Comparison figure saved.")

    print("\n[DONE] V2 training complete.")
