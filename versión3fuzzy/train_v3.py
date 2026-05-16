"""
train_v3.py
-----------
Entry point for training the Q-learning agent on SpaceMissionEnv.

Usage:
    python train_v3.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import QTABLE_PATH, METRICS_CSV_PATH, FIGURE_PATH, V1_METRICS_CSV, V2_METRICS_CSV, COMPARISON_CSV
from agent import QLearningAgent
from environment import SpaceMissionEnv
from training import run_training_v3
from utils import (
    save_training_metrics_v3, training_summary_v3,
    load_version_metrics, build_three_way_comparison,
)
from evaluation.plot_metrics_v3 import build_training_figure, build_three_way_figure


if __name__ == "__main__":
    print("=" * 65)
    print("  SpaceRL – Version 3: SpaceMissionEnv  |  Q-learning Training")
    print("=" * 65)

    env   = SpaceMissionEnv()
    agent = QLearningAgent(env)

    print(f"\n[ENV] SpaceMissionEnv")
    print(f"      States: {env.observation_space.n}  |  Actions: {env.action_space.n}")

    print(f"\n[TRAIN] Starting training...\n")
    metrics = run_training_v3(env, agent)
    env.close()

    agent.save(QTABLE_PATH)
    df_train = save_training_metrics_v3(metrics, csv_path=METRICS_CSV_PATH)

    summary = training_summary_v3(df_train, last_n=1000)
    print("\n" + "=" * 65)
    print("  Training Summary (last 1000 episodes)")
    print("=" * 65)
    print(summary.to_string(index=False))

    print(f"\n[PLOT] Generating training figure...")
    build_training_figure(df_train, save_path=FIGURE_PATH, show=False)

    # Three-way comparison
    df_v1 = load_version_metrics(V1_METRICS_CSV, "V1")
    df_v2 = load_version_metrics(V2_METRICS_CSV, "V2")
    if df_v1 is not None or df_v2 is not None:
        build_three_way_comparison(df_v1, df_v2, df_train, out_path=COMPARISON_CSV)
        cmp_fig = FIGURE_PATH.replace("training_metrics_v3", "v1_v2_v3_comparison")
        build_three_way_figure(df_v1, df_v2, df_train, save_path=cmp_fig, show=False)
        print("[PLOT] Three-way comparison figure saved.")

    print("\n[DONE] V3 training complete.")
