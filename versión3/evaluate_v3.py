"""
evaluate_v3.py
--------------
Entry point for evaluating the trained V3 agent on SpaceMissionEnv.

Usage:
    python evaluate_v3.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import QTABLE_PATH, EVAL_CSV_PATH, N_EVAL_EPS, RENDER_DEMO, DEMO_EPS
from agent import QLearningAgent
from environment import SpaceMissionEnv
from evaluation.eval_runner_v3 import evaluate_agent_v3, run_demo
from utils import save_evaluation_metrics_v3, evaluation_summary_v3


if __name__ == "__main__":
    print("=" * 65)
    print("  SpaceRL – Version 3: SpaceMissionEnv  |  Evaluation")
    print("=" * 65)

    agent = QLearningAgent()
    agent.load(QTABLE_PATH)

    results = evaluate_agent_v3(agent, n_episodes=N_EVAL_EPS)

    df_eval = save_evaluation_metrics_v3(
        results["rewards"], results["steps"], results["successes"],
        csv_path=EVAL_CSV_PATH,
    )

    summary = evaluation_summary_v3(df_eval)
    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)
    print(summary.to_string(index=False))
    print("=" * 55)

    if RENDER_DEMO:
        run_demo(agent, n_demo=DEMO_EPS)
    else:
        print("\n[TIP] Set RENDER_DEMO = True in config.py to watch the agent.")

    print(f"\n[DONE] Results saved → {EVAL_CSV_PATH}")