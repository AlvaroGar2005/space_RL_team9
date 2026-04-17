"""
evaluate_taxi_qlearning.py
--------------------------
Entry point for evaluating the trained Q-learning agent on Taxi-v3.

Usage:
    python evaluate_taxi_qlearning.py

Outputs:
    results/metrics/evaluation.csv  – per-episode evaluation metrics
    (optionally) terminal rendering if RENDER_DEMO = True in config.py

Change RENDER_DEMO and N_EVAL_EPS in config.py to adjust behaviour.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import N_EVAL_EPS, RENDER_DEMO, QTABLE_PATH, EVAL_CSV_PATH
from agent import QLearningAgent
from environment import make_env
from evaluation.eval_runner import evaluate_agent, run_demo
from utils import save_evaluation_metrics, evaluation_summary


if __name__ == "__main__":
    print("=" * 60)
    print("  SpaceRL – Version 1: Taxi-v3  |  Q-learning Evaluation")
    print("=" * 60)

    # 1. Load trained agent
    env   = make_env()
    agent = QLearningAgent(env)
    env.close()
    agent.load(QTABLE_PATH)

    # 2. Greedy evaluation
    rewards, steps, successes = evaluate_agent(agent, n_episodes=N_EVAL_EPS)

    # 3. Save evaluation metrics to CSV
    df_eval = save_evaluation_metrics(rewards, steps, successes, csv_path=EVAL_CSV_PATH)

    # 4. Print summary
    summary = evaluation_summary(df_eval)
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(summary.to_string(index=False))
    print("=" * 50)

    # 5. Optional visual demo
    if RENDER_DEMO:
        run_demo(agent)
    else:
        print("\n[TIP] Set RENDER_DEMO = True in config.py to watch the agent play.")

    print(f"\n[DONE] Results saved → {EVAL_CSV_PATH}")