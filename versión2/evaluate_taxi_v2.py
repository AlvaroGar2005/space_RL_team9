"""
evaluate_taxi_v2.py
--------------------
Entry point for evaluating the trained V2 Q-learning agent.

Usage:
    python evaluate_taxi_v2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import QTABLE_PATH, EVAL_CSV_PATH, N_EVAL_EPS
from agent import QLearningAgent
from environment import make_env_v2
from evaluation.eval_runner_v2 import evaluate_agent_v2
from utils import save_evaluation_metrics_v2, evaluation_summary_v2


if __name__ == "__main__":
    print("=" * 65)
    print("  SpaceRL – Version 2: Wrapped Taxi-v3  |  Evaluation")
    print("=" * 65)

    env   = make_env_v2()
    agent = QLearningAgent(env)
    env.close()
    agent.load(QTABLE_PATH)

    results = evaluate_agent_v2(agent, n_episodes=N_EVAL_EPS)

    df_eval = save_evaluation_metrics_v2(
        results["rewards"], results["steps"], results["successes"],
        results["invalid_actions"], results["wrapper_penalty"],
        results["wrapper_bonus"], results["blocked_actions"],
        results["useful_actions_ratio"],
        csv_path=EVAL_CSV_PATH,
    )

    summary = evaluation_summary_v2(df_eval)
    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)
    print(summary.to_string(index=False))
    print("=" * 55)
    print(f"\n[DONE] Results saved → {EVAL_CSV_PATH}")
